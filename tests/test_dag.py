from typing import Any

import pytest

from dag.dag import Providers


def test_make_resolves_chain():
    """A simple linear chain builds in dependency order."""
    ps = Providers()

    def first_output(input: str) -> str:
        return input + "-first"

    def second_output(first_output: str) -> str:
        return first_output + "-second"

    ps.add(first_output)
    ps.add(second_output)

    recipe = ps.recipe(["second_output"], {"input": "x"})
    values: dict[str, Any] = {"input": "x"}
    for stage in recipe:
        values[stage.name] = stage.call(**values)

    assert values["second_output"] == "x-first-second"


def test_cyclic_dependency_is_caught():
    """A -> B -> A is reported as a cycle, not silently looped or mis-flagged."""
    ps = Providers()

    def a(b: str) -> str:
        return b

    def b(a: str) -> str:
        return a

    ps.add(a)
    ps.add(b)

    with pytest.raises(ValueError, match="Cyclic dependency"):
        ps.recipe(["a"], {})


def test_incompatible_type_declarations_are_caught():
    """Declaring the same resource as two unrelated types is rejected."""
    ps = Providers()

    def produces_int() -> int:
        return 1

    def consumes_str(produces_int: str) -> str:
        return produces_int

    ps.add(produces_int)
    with pytest.raises(TypeError, match="produces_int"):
        ps.add(consumes_str)


def test_compatible_type_declarations_work():
    """A dict[str, int] producer satisfies a dict[str, Any] consumer."""
    ps = Providers()

    def produces() -> dict[str, int]:
        return {"answer": 42}

    def consumes(produces: dict[str, Any]) -> int:
        return produces["answer"]

    # Neither registration should raise: dict[str, int] <: dict[str, Any].
    ps.add(produces)
    ps.add(consumes)

    recipe = ps.recipe(["consumes"], {})
    assert [p.name for p in recipe] == ["produces", "consumes"]


def test_overbroad_provider_is_rejected():
    """A producer broader than its consumer is unsafe and rejected.

    dict[str, Any] is *not* a subtype of dict[str, int], so a provider that
    only promises dict[str, Any] cannot satisfy a consumer requiring
    dict[str, int].
    """
    ps = Providers()

    def produces() -> dict[str, Any]:
        return {"answer": "not an int"}

    def consumes(produces: dict[str, int]) -> int:
        return produces["answer"]

    ps.add(produces)
    with pytest.raises(TypeError, match="produces"):
        ps.add(consumes)


def test_type_check_is_order_independent():
    """Incompatibility is caught even when the consumer is registered first."""
    ps = Providers()

    def consumes(produces: str) -> str:
        return produces

    def produces() -> int:
        return 1

    ps.add(consumes)  # consumer first, before any producer exists
    with pytest.raises(TypeError, match="produces"):
        ps.add(produces)


def test_diamond_dependency_is_handled():
    """A shared dependency is built exactly once, not flagged as a cycle."""
    ps = Providers()

    def base(seed: int) -> int:
        return seed + 1

    def left(base: int) -> int:
        return base * 10

    def right(base: int) -> int:
        return base * 100

    def top(left: int, right: int) -> int:
        return left + right

    for f in (base, left, right, top):
        ps.add(f)

    recipe = ps.recipe(["top"], {"seed": 1})
    names = [p.name for p in recipe]

    # base appears once, and before both of its consumers; both consumers
    # precede top.
    assert names.count("base") == 1
    assert names.index("base") < names.index("left")
    assert names.index("base") < names.index("right")
    assert names.index("left") < names.index("top")
    assert names.index("right") < names.index("top")

    values: dict[str, Any] = {"seed": 1}
    for stage in recipe:
        values[stage.name] = stage.call(**values)
    assert values["top"] == 20 + 200  # base=2 -> left=20, right=200


def test_missing_dependency_raises_no_provider():
    """Requiring a resource that nothing provides is an error."""
    ps = Providers()

    def needs_missing(missing: str) -> str:
        return missing

    ps.add(needs_missing)

    with pytest.raises(ValueError, match="No provider registered"):
        ps.recipe(["needs_missing"], {})


def test_explicitly_supplied_resource_skips_its_provider():
    """If a resource is passed in as an input, its provider is not run."""
    ps = Providers()
    calls: list[str] = []

    def first_output(input: str) -> str:
        calls.append("first_output")
        return input + "-first"

    def second_output(first_output: str) -> str:
        return first_output + "-second"

    ps.add(first_output)
    ps.add(second_output)

    # Supply first_output directly; its provider should be skipped.
    recipe = ps.recipe(["second_output"], {"first_output": "given"})
    names = [p.name for p in recipe]
    assert names == ["second_output"]

    values: dict[str, Any] = {"first_output": "given"}
    for stage in recipe:
        values[stage.name] = stage.call(**values)

    assert values["second_output"] == "given-second"
    assert calls == []  # first_output provider never invoked
