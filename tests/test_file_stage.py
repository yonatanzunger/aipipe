from pathlib import Path

import pytest

from dag import dag as dag_mod
from dag.dag import ProviderRegistry, make
from dag.llm_stage import LLMStage, stage_from_file
from llm.chat import ClientChatBackend
from llm.client import LLMClient
from llm.types import LLMResponse, TextBlock


class FakeClient(LLMClient):
    TIER_DEFAULTS = {"default": "fake-default"}

    async def _create_message(self, *, messages, model, max_tokens=4096, system=None, tools=None):
        return LLMResponse(content=[TextBlock(text=f"OUT[{messages[-1]['content']}]")])


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


def test_file_output_writes_and_returns_path(reg, backend, tmp_path):
    LLMStage("summary", "Sum: {{document}}", output=Path, backend=backend).register(reg)
    result = make("summary", document="hello", workdir=tmp_path)

    out = result["summary"]
    assert isinstance(out, Path)
    assert out == tmp_path / "summary.md"
    assert out.read_text() == "OUT[Sum: hello]"


def test_as_provider_declares_path_and_workdir():
    p = LLMStage("s", "x {{a}}", output=Path).as_provider()
    assert p.provides is Path
    assert "workdir" in p.requires  # needs the dir to write into


def test_downstream_sees_the_path_not_the_text(reg, backend, tmp_path):
    # A file-output producer feeds a consumer; the consumer's {{draft}} should
    # render as the *path* to the file, not its contents.
    LLMStage("draft", "Write {{topic}}", output=Path, backend=backend).register(reg)
    LLMStage("review", "Review {{draft}}", output=str, backend=backend).register(reg)

    result = make("review", topic="cats", workdir=tmp_path)
    draft_path = tmp_path / "draft.md"
    assert result["draft"] == draft_path
    # review's prompt embedded the path string, not the draft's text
    assert result["review"] == f"OUT[Review {draft_path}]"


def test_string_output_leaves_no_workdir(reg, backend, tmp_path):
    # A str-output stage shouldn't touch the workdir at all (lazy creation).
    empty = tmp_path / "run"
    LLMStage("answer", "Q {{q}}", output=str, backend=backend).register(reg)
    result = make("answer", q="why", workdir=empty)
    assert result["answer"] == "OUT[Q why]"
    assert not empty.exists()


def test_stage_from_file_front_matter_output(tmp_path):
    p = tmp_path / "summary.md"
    p.write_text("---\noutput: file\nextension: .txt\n---\nSum {{doc}}")
    stage = stage_from_file(p)
    assert stage.output is Path
    assert stage.extension == ".txt"


def test_stage_from_file_output_default_is_path(tmp_path):
    p = tmp_path / "plain.md"
    p.write_text("Just {{x}}")
    assert stage_from_file(p).output is Path  # file output by default
    assert stage_from_file(p, output=str).output is str
