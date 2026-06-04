import pytest

from dag import dag as dag_mod
from dag.dag import ProviderRegistry, make
from dag.llm_stage import stage_from_file
from llm.chat import ClientChatBackend
from llm.client import LLMClient
from llm.types import LLMResponse, TextBlock


class FakeClient(LLMClient):
    TIER_DEFAULTS = {"default": "fake-default"}

    async def _create_message(self, *, messages, model, max_tokens=4096, system=None, tools=None):
        return LLMResponse(content=[TextBlock(text=f"[{model}|sys={system}] {messages[-1]['content']}")])


@pytest.fixture
def reg(monkeypatch):
    r = ProviderRegistry()
    monkeypatch.setattr(dag_mod, "registry", r)
    return r


@pytest.fixture
def backend():
    be = ClientChatBackend(FakeClient(), tiers={"default": "fake-default"})
    be.connect()
    yield be
    be.disconnect()


def test_name_defaults_to_stem(tmp_path):
    p = tmp_path / "summary.md"
    p.write_text("Summarize {{document}}")
    stage = stage_from_file(p)
    assert stage.name == "summary"
    assert stage.variables == ["document"]
    assert stage.model is None and stage.system is None


def test_front_matter_overrides(tmp_path):
    p = tmp_path / "ignored.md"
    p.write_text("---\nname: refined\nmodel: claude-x\nsystem: Be brief.\n---\nDo {{thing}}")
    stage = stage_from_file(p, model="fallback-model")
    assert stage.name == "refined"
    assert stage.model == "claude-x"   # front matter beats the model arg
    assert stage.system == "Be brief."
    assert stage.template == "Do {{thing}}"


def test_model_arg_used_when_no_front_matter_model(tmp_path):
    p = tmp_path / "ask.md"
    p.write_text("Ask {{q}}")
    stage = stage_from_file(p, model="arg-model")
    assert stage.model == "arg-model"


def test_non_identifier_stem_raises(tmp_path):
    p = tmp_path / "my-summary.md"
    p.write_text("hi {{x}}")
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        stage_from_file(p)


def test_non_identifier_stem_rescued_by_front_matter(tmp_path):
    p = tmp_path / "my-summary.md"
    p.write_text("---\nname: my_summary\n---\nhi {{x}}")
    assert stage_from_file(p).name == "my_summary"


def test_end_to_end_via_make(tmp_path, reg, backend):
    p = tmp_path / "summary.md"
    p.write_text("---\nsystem: Be brief.\n---\nSummarize: {{document}}")
    stage_from_file(p, output=str, backend=backend).register(reg)
    out = make("summary", document="hello")
    assert out["summary"] == "[fake-default|sys=Be brief.] Summarize: hello"
