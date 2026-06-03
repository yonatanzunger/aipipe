from beartype.door import is_subhint
from copy import copy
from dataclasses import dataclass
from collections.abc import Callable
from typing import Union, get_origin, get_args, Any, Iterable
from inspect import signature, Signature
from types import NoneType

"""Sample usage:

@provider
def first_output(input: Path) -> str:
    return input

@provider
def second_output(input: Path, first_output: str) -> str:
    return output

make("second_output", input="/foo/bar")
"""


@dataclass(eq=False)
class Provider:
    func: Callable[..., Any]
    name: str
    provides: Any
    requires: dict[str, Any]
    optionally_requires: dict[str, Any]

    def call(self, **kwargs: Any) -> Any:
        args = {
            name: value
            for name, value in kwargs.items()
            if name in self.requires or name in self.optionally_requires
        }
        return self.func(**args)

    @classmethod
    def build(cls, func: Callable[..., Any]) -> "Provider":
        sig = signature(func)
        provider = Provider(
            func=func,
            name=func.__name__,
            provides=_annot(sig.return_annotation),
            requires={},
            optionally_requires={},
        )
        for param in sig.parameters.values():
            t = _annot(param.annotation)
            if get_origin(t) == Union and NoneType in get_args(t):
                provider.optionally_requires[param.name] = t
            else:
                provider.requires[param.name] = t
        return provider


class Providers:
    def __init__(self):
        self.providers: dict[str, Provider] = {}
        # The type each resource is provided as (by the provider whose name
        # matches the resource), and the types it is required as by consumers.
        self.provided: dict[str, Any] = {}
        self.required: dict[str, list[tuple[str, Any]]] = {}

    def add(self, func: Callable[..., Any]) -> None:
        p = Provider.build(func)
        self._declare_provided(func.__name__, p.provides)
        for name, annot in p.requires.items():
            self._declare_required(func.__name__, name, annot)
        for name, annot in p.optionally_requires.items():
            self._declare_required(func.__name__, name, annot)
        self.providers[func.__name__] = p

    def _declare_provided(self, name: str, annot: Any) -> None:
        self.provided[name] = annot
        for consumer, required in self.required.get(name, ()):
            self._check_compatible(name, annot, consumer, required)

    def _declare_required(self, consumer: str, name: str, annot: Any) -> None:
        self.required.setdefault(name, []).append((consumer, annot))
        if name in self.provided:
            self._check_compatible(name, self.provided[name], consumer, annot)

    def _check_compatible(
        self, resource: str, provided: Any, consumer: str, required: Any
    ) -> None:
        # The produced value must be assignable to the slot that consumes it,
        # i.e. the provided type must be a subtype of the required type. `Any`
        # is treated as compatible in either direction (gradual typing): an
        # untyped provider may feed any consumer, and any value satisfies an
        # untyped consumer.
        if provided is Any or required is Any or is_subhint(provided, required):
            return
        raise TypeError(
            f"Incompatible types for resource '{resource}': provided as "
            f"{provided} but consumer '{consumer}' requires {required}."
        )

    def recipe(
        self, targets: Iterable[str], resources: dict[str, Any]
    ) -> list[Provider]:
        available = set(resources.keys())
        tasks: list[Provider] = []
        scheduled: set[str] = set()  # resources whose provider is already in tasks
        path: list[str] = []  # resources on the current DFS path
        on_path: set[str] = set()  # same, as a set for O(1) membership tests

        def visit(target: str) -> None:
            # Already supplied as an input, or already produced by another branch
            # of the search: nothing to do.
            if target in available or target in scheduled:
                return
            if target not in self.providers:
                raise ValueError(f"No provider registered that makes '{target}'")
            if target in on_path:
                cycle = " -> ".join([*path[path.index(target) :], target])
                raise ValueError(f"Cyclic dependency found: {cycle}")

            path.append(target)
            on_path.add(target)
            for dep in self.providers[target].requires:
                visit(dep)
            path.pop()
            on_path.discard(target)

            # Post-order: a provider is appended only after all of its
            # dependencies, so the recipe is already in execution order.
            scheduled.add(target)
            tasks.append(self.providers[target])

        for target in targets:
            visit(target)

        return tasks


_providers = Providers()


def _annot(x: Any) -> Any:
    return x if x is not Signature.empty else Any


def provider[T, **P](func: Callable[P, T]) -> Callable[P, T]:
    _providers.add(func)
    return func


def make(targets: str | Iterable[str], **kwargs: Any) -> dict[str, Any]:
    """Actually build one or more targets.

    target: Either the name of a target resource to provide, or an iterable of names.
    kwargs: resource=value giving all resources that are provided as inputs.

    Returns:
    Dict from target resource name to its value.
    """
    if isinstance(targets, str):
        targets = (targets,)
    resources = copy(kwargs)
    recipe = _providers.recipe(targets, resources)
    for stage in recipe:
        value = stage.call(**resources)
        resources[stage.name] = value

    return resources
