import argparse
from typing import Any

import pytest

from dag import dag as dag_mod
from dag.dag import (
    Provider,
    ProviderRegistry,
    Source,
    VLevel,
    make,
    provider,
)
from dag.dag import _as_arg


def fn(func):
    """Shorthand: build a Provider from a function."""
    return Provider.from_function(func)


@pytest.fixture
def reg(monkeypatch):
    """A fresh global registry for tests that go through make()/@provider."""
    r = ProviderRegistry()
    monkeypatch.setattr(dag_mod, "registry", r)
    return r


# --------------------------------------------------------------------------- #
# recipe()
# --------------------------------------------------------------------------- #


def test_recipe_linear_chain():
    r = ProviderRegistry()

    def first_output(input: str) -> str:
        return input + "-first"

    def second_output(first_output: str) -> str:
        return first_output + "-second"

    r.add(fn(first_output))
    r.add(fn(second_output))

    recipe = r.recipe(["second_output"], ["input"])
    assert [p.name for p in recipe] == ["first_output", "second_output"]


def test_cyclic_dependency_is_caught():
    r = ProviderRegistry()

    def a(b: str) -> str:
        return b

    def b(a: str) -> str:
        return a

    r.add(fn(a))
    r.add(fn(b))

    with pytest.raises(ValueError, match="Cyclic dependency"):
        r.recipe(["a"], [])


def test_diamond_dependency_is_handled():
    r = ProviderRegistry()

    def base(seed: int) -> int:
        return seed + 1

    def left(base: int) -> int:
        return base * 10

    def right(base: int) -> int:
        return base * 100

    def top(left: int, right: int) -> int:
        return left + right

    for f in (base, left, right, top):
        r.add(fn(f))

    names = [p.name for p in r.recipe(["top"], ["seed"])]
    assert names.count("base") == 1
    assert names.index("base") < names.index("left")
    assert names.index("base") < names.index("right")
    assert names.index("left") < names.index("top")
    assert names.index("right") < names.index("top")


def test_missing_dependency_raises_no_provider():
    r = ProviderRegistry()

    def needs_missing(missing: str) -> str:
        return missing

    r.add(fn(needs_missing))

    with pytest.raises(ValueError, match="No provider registered"):
        r.recipe(["needs_missing"], [])


def test_explicitly_supplied_resource_skips_its_provider():
    r = ProviderRegistry()

    def first_output(input: str) -> str:
        return input + "-first"

    def second_output(first_output: str) -> str:
        return first_output + "-second"

    r.add(fn(first_output))
    r.add(fn(second_output))

    # first_output is supplied directly, so its provider should be skipped.
    names = [p.name for p in r.recipe(["second_output"], ["first_output"])]
    assert names == ["second_output"]


# --------------------------------------------------------------------------- #
# Type compatibility (order independent)
# --------------------------------------------------------------------------- #


def test_incompatible_types_caught():
    r = ProviderRegistry()

    def produces_int() -> int:
        return 1

    def consumes_str(produces_int: str) -> str:
        return produces_int

    r.add(fn(produces_int))
    with pytest.raises(TypeError, match="produces_int"):
        r.add(fn(consumes_str))


def test_compatible_types_work():
    r = ProviderRegistry()

    def produces() -> dict[str, int]:
        return {"answer": 42}

    def consumes(produces: dict[str, Any]) -> int:
        return produces["answer"]

    r.add(fn(produces))
    r.add(fn(consumes))  # dict[str, int] <: dict[str, Any]
    assert [p.name for p in r.recipe(["consumes"], [])] == ["produces", "consumes"]


def test_overbroad_provider_rejected():
    r = ProviderRegistry()

    def produces() -> dict[str, Any]:
        return {"answer": "not an int"}

    def consumes(produces: dict[str, int]) -> int:
        return produces["answer"]

    r.add(fn(produces))
    with pytest.raises(TypeError, match="produces"):
        r.add(fn(consumes))


def test_type_check_is_order_independent():
    r = ProviderRegistry()

    def consumes(produces: str) -> str:
        return produces

    def produces() -> int:
        return 1

    r.add(fn(consumes))  # consumer first, declares 'produces' implicitly as str
    with pytest.raises(TypeError, match="produces"):
        r.add(fn(produces))


# --------------------------------------------------------------------------- #
# Identifier enforcement
# --------------------------------------------------------------------------- #


def test_declare_resource_rejects_non_identifier():
    r = ProviderRegistry()
    for bad in ("a.b", "1abc", "with-dash", "has space"):
        with pytest.raises(ValueError, match="not a valid Python identifier"):
            r.declare_resource(bad, int)


def test_requirement_non_identifier_rejected():
    r = ProviderRegistry()
    bad = Provider(
        name="thing",
        func=lambda **kw: 1,
        provides=int,
        requires={"bad.name": int},
        optionally_requires={},
    )
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        r.add(bad)


# --------------------------------------------------------------------------- #
# Source priority: explicit > provider > implicit
# --------------------------------------------------------------------------- #


def test_explicit_beats_provider_and_any_never_clobbers():
    r = ProviderRegistry()
    r.declare_resource("x", int)  # EXPLICIT

    def x():  # no annotation -> provides Any
        return 5

    r.add(fn(x))
    assert r.resources["x"].type is int
    assert r.resources["x"].source is Source.EXPLICIT


def test_provider_beats_implicit():
    r = ProviderRegistry()

    def consumer(x: object) -> int:  # declares 'x' implicitly as object
        return 0

    def x() -> int:  # provider for x
        return 1

    r.add(fn(consumer))
    assert r.resources["x"].source is Source.IMPLICIT
    assert r.resources["x"].type is object

    r.add(fn(x))  # int <: object, so this is compatible and provider wins
    assert r.resources["x"].type is int
    assert r.resources["x"].source is Source.PROVIDER


def test_explicit_redeclare_after_provider_triggers_conformance():
    r = ProviderRegistry()

    def y() -> bool:
        return True

    r.add(fn(y))
    # bool <: int, so an explicit narrowing to int is fine and wins.
    r.declare_resource("y", int)
    assert r.resources["y"].type is int
    assert r.resources["y"].source is Source.EXPLICIT

    # But declaring an incompatible explicit type must fail (provider yields bool).
    with pytest.raises(TypeError):
        r.declare_resource("y", str)


# --------------------------------------------------------------------------- #
# VLevel
# --------------------------------------------------------------------------- #


def test_vlevel_default_zero():
    assert VLevel({})("anything") == 0


def test_vlevel_reads_verbosity():
    assert VLevel({"verbosity": 3})("stage") == 3


def test_vlevel_vmodule_overrides():
    v = VLevel({"verbosity": 1, "vmodule": "a:5,b:2"})
    assert v("a") == 5
    assert v("b") == 2
    assert v("c") == 1  # falls back to the global level


def test_vlevel_bad_vmodule_raises():
    with pytest.raises(ValueError, match="vmodule"):
        VLevel({"vmodule": "garbage"})
    with pytest.raises(ValueError, match="vmodule"):
        VLevel({"vmodule": "a:3x"})  # fullmatch rejects trailing junk


def test_vlevel_empty_entries_skipped():
    assert VLevel({"verbosity": 0, "vmodule": "a:5,"})("a") == 5


# --------------------------------------------------------------------------- #
# make() + verbosity injection
# --------------------------------------------------------------------------- #


def test_make_chain(reg):
    @provider
    def first_output(input: str) -> str:
        return input + "-first"

    @provider
    def second_output(first_output: str) -> str:
        return first_output + "-second"

    assert make("second_output", input="x")["second_output"] == "x-first-second"


def test_make_injects_verbosity_non_optional(reg):
    @provider
    def thing(verbosity: int) -> str:  # required, not optional
        return f"v={verbosity}"

    assert make("thing")["thing"] == "v=0"
    assert make("thing", verbosity=2)["thing"] == "v=2"


def test_make_vmodule_per_provider_override(reg):
    @provider
    def a(verbosity: int) -> int:
        return verbosity

    @provider
    def b(a: int, verbosity: int) -> int:
        return verbosity

    out = make("b", verbosity=1, vmodule="a:7")
    assert out["a"] == 7  # per-provider override
    assert out["b"] == 1  # global level


# --------------------------------------------------------------------------- #
# argparse helpers
# --------------------------------------------------------------------------- #


def test_as_arg():
    assert _as_arg("v") == "-v"
    assert _as_arg("verbosity") == "--verbosity"


def test_add_arguments_builds_flags_with_alias_and_type():
    r = ProviderRegistry()
    r.declare_resource("count", int, aliases=["c"], help="how many")
    r.declare_resource("name", str)

    parser = argparse.ArgumentParser()
    r.add_arguments(parser)

    ns = parser.parse_args(["--count", "3", "-c", "5", "--name", "bob"])
    assert ns.count == 5 and isinstance(ns.count, int)  # alias maps to same dest
    assert ns.name == "bob"

    # Unset resources default to None.
    ns2 = parser.parse_args([])
    assert ns2.count is None and ns2.name is None
