"""A library to easily run DAGs of tasks.

The idea is that we have a graph of resources. Each resource has a name, which is an identifier-safe
string, and a value, which is of any type -- although usually one that can be simply converted to
and from a string. Most resources are associated with providers -- callables that take some resources
as inputs and provide another resource as an output -- and these are stored in a registry. The make()
function lets you provide explicit values for some resources, select which resources you want to build,
and then runs the requisite providers in the right order so you get the required outputs.

The easiest way to create a Provider is by putting the @provider decorator on a function. This creates
a resource with the same name as the function (so name your function after the noun that it makes, not
the verb for creating it!), whose input resources come from its arguments: the function argument names
and types should match those of resources. Any optional argument (foo | None) is treated as an optional
dependency: that is, the resource will be resources to the function if it's available, but we won't
try to provide it unless it's already there.

For example:

    @provider
    def message(message_id: int) -> str:
    ... get a message from a store

    @provider
    def summary(message: str) -> str:
    ... summarize a message

make("summary", message_id=12345) then returns a dict:
{
  message_id: 12345,
  message: "lorem ipsum dolor sit amet, consectetur adipiscing elit",
  summary: "Onlookers were surprised by the unusual quantity of blood"
}

The returned dict contains all the resources created in the process, including your inputs, the targets
you requested, and any intermediate resources that were made.

You can also declare that a resource exists and has a certain type ahead of time using the "resource"
function:

    resource("verbosity", int)

"verbosity" is actually a special global resource managed using the LoggerFactory class below.
"""

import argparse
from beartype.door import is_subhint
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from collections.abc import Callable
from typing import Union, get_origin, get_args, Any, Iterable, NamedTuple
from inspect import signature, Signature
from pathlib import Path
from types import NoneType
from dag.logger import Logger, LoggerFactory


# TODO: It would be really nice if we had a type that could be used to describe a type signature. I'll
# give it a distinct name now for clarity.
TypeSig = Any


class Source(IntEnum):
    """Where a resource's declared type came from, in increasing priority order.

    When the same resource is declared more than once, the highest-priority
    *concrete* declaration wins as the canonical type. An untyped (Any)
    declaration never overrides a concrete one. The order encodes:

        explicit declaration > a provider's output type > inferred from a requirement
    """

    IMPLICIT = 1  # inferred because some provider takes it as an input
    PROVIDER = 2  # the output type of the provider that makes this resource
    EXPLICIT = 3  # declared directly via resource()


@dataclass(eq=False)
class Provider:
    name: str
    func: Callable[..., Any]
    provides: TypeSig
    requires: dict[str, TypeSig]
    optionally_requires: dict[str, TypeSig]
    # A short human-readable description of what kind of provider this is
    # (e.g. "function", "llm stage"). Used by introspection/`info`; extensible
    # — new provider generators set their own.
    kind: str = "function"

    def __call__(self, **kwargs: Any) -> Any:
        """You can call the provider with the *full* set of resources. Only the relevant arguments
        are passed to the underlying callable.
        """
        args = {
            name: value
            for name, value in kwargs.items()
            if name in self.requires or name in self.optionally_requires
        }
        return self.func(**args)

    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> "Provider":
        sig = signature(func)
        provider = Provider(
            name=func.__name__,
            func=func,
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


def _annot(x: TypeSig) -> TypeSig:
    return x if x is not Signature.empty else Any


class Plan(NamedTuple):
    """The result of :meth:`ProviderRegistry.plan`."""

    steps: list["Provider"]  # providers to run, in execution order
    missing: list[str]  # required resources with no provider (inputs to supply)


@dataclass
class ResourceInfo:
    name: str
    type: TypeSig
    # Where the canonical type came from; higher-priority sources win on conflict.
    source: Source

    # Aliases usable in argparse
    aliases: list[str] | None
    # Documentation for command-line flags
    help: str | None
    # True if this resource is available on the command-line.
    available_on_cl: bool = True

    def add_argument(self, parser: argparse.ArgumentParser) -> None:
        if not self.available_on_cl:
            return
        names = [_as_arg(self.name)]
        if self.aliases is not None:
            names.extend(_as_arg(alias) for alias in self.aliases)
        parser.add_argument(
            *names,
            type=self.type if callable(self.type) and self.type is not Any else str,
            required=False,
            help=self.help,
        )


def _as_arg(name: str) -> str:
    if len(name) == 1:
        return f"-{name}"
    else:
        return f"--{name}"


class ProviderRegistry:
    def __init__(self):
        self.providers: dict[str, Provider] = {}
        # The canonical type (and where it came from) for each known resource.
        self.resources: dict[str, ResourceInfo] = {}
        # Every (consumer, required_type) pair seen for each resource, used to
        # verify that the resource's type satisfies all of its consumers.
        self.required: dict[str, list[tuple[str, TypeSig]]] = {}

    def add(self, p: Provider) -> None:
        """Add a provider to the registry."""
        self.declare_resource(
            p.name, p.provides, source=Source.PROVIDER, help=p.func.__doc__
        )
        self.providers[p.name] = p
        for name, annot in p.requires.items():
            self._add_requirement(p.name, name, annot)
        for name, annot in p.optionally_requires.items():
            self._add_requirement(p.name, name, annot)
        # Now that the provider itself is registered, verify that its output
        # conforms to the resource's (possibly higher-priority) declared type.
        self._recheck(p.name)

    def declare_resource(
        self,
        name: str,
        annot: TypeSig,
        *,
        source: Source = Source.EXPLICIT,
        aliases: list[str] | None = None,
        help: str | None = None,
        available_on_cl: bool = True,
    ) -> None:
        """Declare that a resource exists and has a given type.

        The canonical type follows the priority order in Source: a higher-priority
        concrete declaration overrides a lower-priority one, but an untyped (Any)
        declaration never overrides a concrete type. Compatibility with the
        resource's provider and all of its consumers is rechecked afterwards.
        """
        self._check_name(name)
        prior = self.resources.get(name)
        if prior is None or prior.type is Any:
            chosen_type, chosen_source = annot, source
        elif annot is Any:
            chosen_type, chosen_source = prior.type, prior.source
        elif source >= prior.source:
            chosen_type, chosen_source = annot, source
        else:
            chosen_type, chosen_source = prior.type, prior.source

        self.resources[name] = ResourceInfo(
            name=name,
            type=chosen_type,
            source=chosen_source,
            aliases=aliases
            if aliases is not None
            else (prior.aliases if prior else None),
            help=help if help is not None else (prior.help if prior else None),
            available_on_cl=available_on_cl,
        )
        self._recheck(name)

    def _add_requirement(self, consumer: str, name: str, annot: TypeSig) -> None:
        """Record that `consumer` requires resource `name` with type `annot`."""
        self._check_name(name)
        self.required.setdefault(name, []).append((consumer, annot))
        if name not in self.resources:
            # A required-but-undeclared resource exists implicitly (e.g. a leaf
            # input), so it becomes a resource (and a CLI flag) with this type.
            self.declare_resource(name, annot, source=Source.IMPLICIT)
        else:
            self._recheck(name)

    def _recheck(self, name: str) -> None:
        """Verify the resource's canonical type is consistent with its provider's
        output (which must conform to the declared type) and with every consumer
        (the declared type must satisfy each requirement).
        """
        canonical = self.resources[name].type
        provider = self.providers.get(name)
        if provider is not None:
            self._check_compatible(name, provider.provides, name, canonical)
        for consumer, required in self.required.get(name, ()):
            self._check_compatible(name, canonical, consumer, required)

    def _check_name(self, resource: str) -> None:
        if not resource.isidentifier():
            raise ValueError(
                f"The resource name '{resource}' is not a valid Python identifier"
            )

    def _check_compatible(
        self,
        resource: str,
        provided_type: TypeSig,
        consumer: str,
        required_type: TypeSig,
    ) -> None:
        # The produced value must be assignable to the slot that consumes it,
        # i.e. the resources type must be a subtype of the required type. `Any`
        # is treated as compatible in either direction (gradual typing): an
        # untyped provider may feed any consumer, and any value satisfies an
        # untyped consumer.
        if (
            provided_type is Any
            or required_type is Any
            or is_subhint(provided_type, required_type)
        ):
            return
        raise TypeError(
            f"Incompatible types for resource '{resource}': provided as "
            f"{provided_type} but consumer '{consumer}' requires {required_type}."
        )

    def plan(self, targets: Iterable[str], resources: Iterable[str]) -> Plan:
        """Compute the build plan for *targets* given the available *resources*.

        Returns a :class:`Plan` (``steps``, ``missing``): ``steps`` are the
        providers to run in execution order; ``missing`` lists required resources
        that are neither supplied nor produced by any provider (i.e. inputs the
        caller must provide). Unlike :meth:`recipe`, missing inputs are reported
        rather than raised — useful for dry runs and graph views. Cycles raise.
        """
        available = set(resources)
        steps: list[Provider] = []
        scheduled: set[str] = set()  # resources whose provider is already in steps
        missing: list[str] = []
        missing_seen: set[str] = set()
        path: list[str] = []  # resources on the current DFS path
        on_path: set[str] = set()  # same, as a set for O(1) membership tests

        def visit(target: str) -> None:
            # Already supplied as an input, or already produced by another branch
            # of the search: nothing to do.
            if target in available or target in scheduled:
                return
            if target not in self.providers:
                if target not in missing_seen:
                    missing_seen.add(target)
                    missing.append(target)
                return
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
            # dependencies, so the steps are already in execution order.
            scheduled.add(target)
            steps.append(self.providers[target])

        for target in targets:
            visit(target)

        return Plan(steps, missing)

    def recipe(
        self, targets: Iterable[str], resources: Iterable[str]
    ) -> list[Provider]:
        """Compute the sequence of tasks you need to build the given targets from the
        given resources. Raises if a required resource has no provider.
        """
        result = self.plan(targets, resources)
        if result.missing:
            raise ValueError(
                f"No provider registered that makes '{result.missing[0]}'"
            )
        return result.steps

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add all our known resources to the parser."""
        for resource in self.resources.values():
            resource.add_argument(parser)


registry = ProviderRegistry()
"""The global static registry."""


def provider[T, **P](func: Callable[P, T]) -> Callable[P, T]:
    """The @provider decorator."""
    registry.add(Provider.from_function(func))
    return func


def resource(
    name: str,
    annot: TypeSig,
    *,
    aliases: list[str] | None = None,
    help: str | None = None,
    available_on_cl: bool = True,
) -> None:
    """Declare the existence of a resource."""
    registry.declare_resource(
        name, annot, aliases=aliases, help=help, available_on_cl=available_on_cl
    )


# The resources the logging system uses. `verbosity` and `vmodule` are
# user-facing inputs that LoggerFactory reads; `logger` is hand-forged by make()
# and injected into every provider call (never built by the graph), so it's the
# single ambient resource a provider declares to get logging.
resource("verbosity", int, aliases=["v"], help="Verbosity level for DAG execution")
resource(
    "vmodule",
    str,
    help="Per-provider verbosity overrides. Format as provider:value,provider:value",
)
resource("logger", annot=Logger, available_on_cl=False)


@provider
def workdir() -> Path:
    """The working directory for this run (created lazily by file stages).

    This is often overridden on the command line.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return (Path.cwd() / ".aipipe" / "runs" / stamp).resolve()


_AMBIENT_RESOURCES = ("logger",)


def make(
    targets: str | Iterable[str],
    *,
    logger_factory: LoggerFactory | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Actually build one or more targets.

    target: Either the name of a target resource to provide, or an iterable of names.
    kwargs: resource=value giving all resources that are provided as inputs.

    Returns:
    Dict from target resource name to its value.
    """
    if isinstance(targets, str):
        targets = (targets,)
    resources = copy(kwargs)

    # `logger` is hand-forged below and injected into every provider call rather
    # than built by the graph, so it's always available to providers that
    # declare it.
    logger_factory = logger_factory or LoggerFactory(resources)
    logger = logger_factory.logger()

    recipe = registry.recipe(targets, [*resources.keys(), *_AMBIENT_RESOURCES])

    logger.log(1, "Targets", sorted(list(targets)))
    if logger.verbosity == 1:
        logger.log(1, "Inputs", sorted(list(resources.keys())))
    elif logger.verbosity > 1:
        logger.header(1, "Inputs")
        for key, value in sorted(resources.items()):
            logger.log(1, key, value, indent=2)
    logger.log(1, "Build order", list(provider.name for provider in recipe))

    for provider in recipe:
        logger.log(1, "Building", provider.name)
        resources[provider.name] = provider(
            **{**resources, "logger": logger_factory.logger(provider.name)}
        )
        logger.log(
            1 if provider.name == "workdir" else 2,
            provider.name,
            resources[provider.name],
        )

    return resources
