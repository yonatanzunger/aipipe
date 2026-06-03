from typing import Any

import pytest

from dag import dag as dag_mod
from dag.dag import ProviderRegistry, make
from dag.llm_stage import LLMStage
from llm.chat import ClientChatBackend
from llm.client import LLMClient
from llm.types import LLMResponse, TextBlock


class FakeClient(LLMClient):
    """Echoes the rendered prompt and resolved model — no network."""

    TIER_DEFAULTS = {"default": "fake-default"}

    async def _create_message(self, *, messages, model, max_tokens=4096, system=None, tools=None):
        return LLMResponse(content=[TextBlock(text=f"[{model}] {messages[-1]['content']}")])


@pytest.fixture
def reg(monkeypatch):
    """Fresh global registry so make() sees only this test's providers."""
    r = ProviderRegistry()
    monkeypatch.setattr(dag_mod, "registry", r)
    return r


@pytest.fixture
def backend():
    be = ClientChatBackend(FakeClient(), tiers={"default": "fake-default"})
    be.connect()
    yield be
    be.disconnect()


# --------------------------------------------------------------------------- #
# Pure unit tests (no registry / backend)
# --------------------------------------------------------------------------- #


def test_variable_discovery_dedup_and_whitespace():
    stage = LLMStage("s", "{{a}} then {{ b }} then {{a}} and {{c_2}}")
    assert stage.variables == ["a", "b", "c_2"]


def test_render_stringifies_values():
    stage = LLMStage("s", "n={{n}}, name={{name}}")
    assert stage.render({"n": 42, "name": "bob"}) == "n=42, name=bob"


def test_as_provider_wiring():
    p = LLMStage("summary", "sum {{doc}} for {{user}}").as_provider()
    assert p.name == "summary"
    assert p.provides is str
    assert set(p.requires) == {"doc", "user"}
    assert all(t is Any for t in p.requires.values())
    assert "model" in p.optionally_requires  # reserved model resource


def test_model_not_duplicated_when_a_template_var():
    # If a template literally uses {{model}}, it's a required var, not the
    # reserved optional.
    p = LLMStage("s", "use {{model}}").as_provider()
    assert "model" in p.requires
    assert "model" not in p.optionally_requires


# --------------------------------------------------------------------------- #
# End-to-end via make()
# --------------------------------------------------------------------------- #


def test_make_runs_stage(reg, backend):
    LLMStage("summary", "Summarize: {{document}}", backend=backend).register(reg)
    out = make("summary", document="hello world")
    assert out["summary"] == "[fake-default] Summarize: hello world"


def test_make_chains_stages(reg, backend):
    LLMStage("draft", "Draft from {{topic}}", backend=backend).register(reg)
    # second stage consumes the first stage's output as a {{draft}} variable
    LLMStage("polished", "Polish: {{draft}}", backend=backend).register(reg)
    out = make("polished", topic="cats")
    assert out["draft"] == "[fake-default] Draft from cats"
    assert out["polished"] == "[fake-default] Polish: [fake-default] Draft from cats"


def test_model_resource_drives_stage(reg, backend):
    LLMStage("answer", "Q: {{q}}", backend=backend).register(reg)
    out = make("answer", q="why", model="claude-x")
    assert out["answer"] == "[claude-x] Q: why"


def test_stage_model_overrides_resource(reg, backend):
    LLMStage("answer", "Q: {{q}}", model="stage-model", backend=backend).register(reg)
    out = make("answer", q="why", model="resource-model")
    assert out["answer"] == "[stage-model] Q: why"
