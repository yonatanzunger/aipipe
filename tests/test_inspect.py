import pytest

from dag.dag import Plan, Provider, ProviderRegistry
from dag.inspect import (
    categorize,
    describe,
    dry_run,
    layered,
    mermaid,
    mermaid_live_url,
    overview,
)


def fn(func):
    return Provider.from_function(func)


@pytest.fixture
def diamond():
    """a (input) → b → c → d, with c also depending directly on a."""
    r = ProviderRegistry()

    def b(a: str) -> str:
        return a

    def c(a: str, b: str) -> str:
        return a + b

    def d(c: str) -> str:
        return c

    for f in (b, c, d):
        r.add(fn(f))
    return r


# --------------------------------------------------------------------------- #
# plan / Plan
# --------------------------------------------------------------------------- #


def test_plan_returns_namedtuple(diamond):
    plan = diamond.plan(["d"], [])
    assert isinstance(plan, Plan)
    assert plan.missing == ["a"]  # the one true input
    names = [s.name for s in plan.steps]
    assert names.index("b") < names.index("c") < names.index("d")


def test_plan_reports_missing_without_raising(diamond):
    # recipe() raises; plan() reports.
    assert diamond.plan(["d"], []).missing == ["a"]
    with pytest.raises(ValueError, match="No provider"):
        diamond.recipe(["d"], [])


def test_plan_supplied_input_has_no_missing(diamond):
    plan = diamond.plan(["d"], ["a"])
    assert plan.missing == []


# --------------------------------------------------------------------------- #
# kind field
# --------------------------------------------------------------------------- #


def test_function_provider_kind():
    assert fn(lambda: 1).kind == "function"  # noqa: E731


def test_llm_stage_kind():
    from dag.llm_stage import LLMStage

    assert LLMStage("s", "{{x}}", output=str).as_provider().kind == "llm stage"


# --------------------------------------------------------------------------- #
# categorize
# --------------------------------------------------------------------------- #


def test_categorize(diamond):
    targets, inputs, reserved = categorize(diamond)
    assert targets == ["b", "c", "d"]
    assert inputs == ["a"]
    assert reserved == []  # no framework resources used here


# --------------------------------------------------------------------------- #
# text views (captured via capsys)
# --------------------------------------------------------------------------- #


def test_overview(diamond, capsys):
    overview(diamond)
    out = capsys.readouterr().out
    assert "Targets (build these)" in out
    assert "Inputs (supply these)" in out
    assert "--a" in out and "→ b, c" in out  # the shared input + its consumers


def test_describe_input(diamond, capsys):
    describe(diamond, "a")
    out = capsys.readouterr().out
    assert "input you supply" in out
    assert "required by: b, c" in out


def test_describe_provider(diamond, capsys):
    describe(diamond, "c")
    out = capsys.readouterr().out
    assert "provided by: function" in out
    assert "requires:" in out and "a" in out and "b" in out


def test_layered(diamond, capsys):
    layered(diamond, ["d"])
    out = capsys.readouterr().out
    assert "inputs" in out and "a" in out
    # each provider appears once, in build order
    assert out.index("b") < out.index("c") < out.index("d")


def test_dry_run_supplied(diamond, capsys):
    dry_run(diamond, ["d"], ["a"])
    out = capsys.readouterr().out
    assert "Dry run" in out and "would build: d" in out
    assert "Supplied: a" in out
    assert "Missing" not in out


def test_dry_run_missing(diamond, capsys):
    dry_run(diamond, ["d"], [])
    out = capsys.readouterr().out
    assert "Missing:" in out and "a (--a)" in out


# --------------------------------------------------------------------------- #
# mermaid
# --------------------------------------------------------------------------- #


def test_mermaid(diamond):
    text = mermaid(diamond, ["d"])
    assert text.startswith("graph TD")
    assert "a --> b" in text
    assert "a --> c" in text  # the diamond edge
    assert "b --> c" in text
    assert "c --> d" in text
    assert 'a["a"]:::input' in text  # input node tagged


def test_mermaid_live_url(diamond):
    url = mermaid_live_url(mermaid(diamond, ["d"]))
    assert url.startswith("https://mermaid.live/edit#pako:")
    assert "=" not in url.split("pako:", 1)[1]  # padding stripped
