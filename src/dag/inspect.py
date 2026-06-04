"""Introspection for a :class:`~dag.dag.ProviderRegistry`.

Human-facing views of a loaded DAG — an overview of resources, a single
resource's details, a topological "what would run" listing, a dry-run plan, and
a Mermaid diagram (plus a no-install mermaid.live link). All the *logic* lives
here; the CLI just calls these.
"""

from __future__ import annotations

import base64
import json
import sys
import zlib
from typing import TYPE_CHECKING, Any, Iterable, TextIO

from pyppin.text.tty import TTY, tty

if TYPE_CHECKING:
    from dag.dag import ProviderRegistry

# Framework resources that aren't part of a user's domain graph; excluded from
# the "supply these" list and from dependency edges so views stay about the DAG.
RESERVED = frozenset({"verbosity", "vmodule", "logger", "workdir", "model"})


def _type_name(t: Any) -> str:
    return getattr(t, "__name__", None) or str(t).replace("typing.", "")


def _kind(reg: ProviderRegistry, name: str) -> str:
    p = reg.providers.get(name)
    return p.kind if p is not None else "input"


def _domain_deps(reg: ProviderRegistry, name: str) -> list[str]:
    """The non-reserved resources a provider requires (its domain inputs)."""
    p = reg.providers.get(name)
    if p is None:
        return []
    return [d for d in p.requires if d not in RESERVED]


def _consumers(reg: ProviderRegistry, name: str) -> list[str]:
    """Providers that require *name*, de-duplicated in first-seen order."""
    seen: dict[str, None] = {}
    for consumer, _type in reg.required.get(name, ()):
        seen.setdefault(consumer, None)
    return list(seen)


def categorize(reg: ProviderRegistry) -> tuple[list[str], list[str], list[str]]:
    """Split known resources into (targets, inputs, reserved).

    - **targets**: have a provider — you can ``make`` them.
    - **inputs**: required by something but have no provider — you must supply.
    - **reserved**: framework knobs (model, verbosity, …) that are present.
    """
    provided = set(reg.providers)
    required = {n for n, lst in reg.required.items() if lst}
    present = set(reg.resources) | provided | required
    targets = sorted(provided - RESERVED)
    inputs = sorted((required - provided) - RESERVED)
    reserved = sorted(present & RESERVED)
    return targets, inputs, reserved


def overview(reg: ProviderRegistry, *, file: TextIO | None = None) -> None:
    """Print the grouped resource overview."""
    file = sys.stdout if file is None else file
    targets, inputs, reserved = categorize(reg)

    def header(text: str) -> None:
        print(tty(TTY.BRIGHT, text=text, file=file), file=file)

    if targets:
        header("Targets (build these)")
        width = max(len(t) for t in targets)
        for name in targets:
            deps = ", ".join(_domain_deps(reg, name)) or "—"
            type_str = _type_name(reg.resources[name].type)
            print(f"  {name:<{width}}  {type_str:<6} ← {deps}"
                  f"   {tty(TTY.DIM, text=_kind(reg, name), file=file)}", file=file)

    if inputs:
        print(file=file)
        header("Inputs (supply these)")
        width = max(len(i) for i in inputs)
        for name in inputs:
            into = ", ".join(_consumers(reg, name)) or "—"
            type_str = _type_name(reg.resources[name].type) if name in reg.resources else "Any"
            print(f"  --{name:<{width}}  {type_str:<6} → {into}", file=file)

    flaggable = [
        r for r in reserved
        if (ri := reg.resources.get(r)) is not None and ri.available_on_cl
    ]
    if flaggable:
        print(file=file)
        header("Reserved")
        print("  " + "  ".join(f"--{r}" for r in flaggable), file=file)


def describe(reg: ProviderRegistry, name: str, *, file: TextIO | None = None) -> None:
    """Print details about a single resource."""
    file = sys.stdout if file is None else file
    info = reg.resources.get(name)
    if info is None and name not in reg.providers:
        print(tty(TTY.RED, text=f"Unknown resource: {name!r}", file=file), file=file)
        return

    print(tty(TTY.BRIGHT, text=name, file=file), file=file)
    if info is not None:
        print(f"  type:        {_type_name(info.type)}  (from {info.source.name.lower()})", file=file)
        if info.help:
            print(f"  help:        {info.help.strip().splitlines()[0]}", file=file)

    p = reg.providers.get(name)
    if p is not None:
        print(f"  provided by: {p.kind}", file=file)
        deps = _domain_deps(reg, name)
        if deps:
            print(f"  requires:    {', '.join(deps)}", file=file)
        opt = [o for o in p.optionally_requires if o not in RESERVED]
        if opt := opt:
            print(f"  optional:    {', '.join(opt)}", file=file)
    else:
        print("  provided by: (nothing — this is an input you supply)", file=file)

    consumers = _consumers(reg, name)
    print(f"  required by: {', '.join(consumers) if consumers else '(nothing)'}", file=file)


def _scope(reg: ProviderRegistry, targets: Iterable[str] | None) -> list[str]:
    """The provider names to include — the targets' subgraph, or all of them."""
    return sorted(reg.providers) if targets is None else list(targets)


def layered(
    reg: ProviderRegistry,
    targets: Iterable[str] | None = None,
    *,
    file: TextIO | None = None,
) -> None:
    """Print the build order: each provider once, with its direct inputs inline.

    Handles diamonds/fan-out natively (nothing is duplicated). With no targets,
    shows the whole DAG.
    """
    file = sys.stdout if file is None else file
    plan = reg.plan(_scope(reg, targets), [])
    steps = [s for s in plan.steps if s.name not in RESERVED]
    if plan.missing:
        print(tty(TTY.BRIGHT, text="inputs", file=file)
              + ":  " + ", ".join(sorted(plan.missing)), file=file)
    width = max((len(s.name) for s in steps), default=0)
    for i, step in enumerate(steps, 1):
        deps = ", ".join(_domain_deps(reg, step.name)) or "—"
        print(f"  {i:>2}. {step.name:<{width}}  ← {deps}", file=file)


def dry_run(
    reg: ProviderRegistry,
    targets: list[str],
    supplied: Iterable[str],
    *,
    file: TextIO | None = None,
) -> None:
    """Print what ``make(targets, **supplied)`` would do, without running it."""
    from dag.dag import _AMBIENT_RESOURCES

    file = sys.stdout if file is None else file
    supplied = set(supplied)
    available = supplied | set(_AMBIENT_RESOURCES)
    plan = reg.plan(targets, available)
    steps = [s for s in plan.steps if s.name not in RESERVED]

    print(tty(TTY.BRIGHT, text="Dry run", file=file)
          + " — would build: " + ", ".join(targets), file=file)
    if steps:
        n = len(steps)
        print(f"\nPlan ({n} step{'s' if n != 1 else ''})", file=file)
        width = max(len(s.name) for s in steps)
        for i, step in enumerate(steps, 1):
            deps = ", ".join(_domain_deps(reg, step.name)) or "—"
            print(f"  {i:>2}. {step.name:<{width}}  [{step.kind}]  ← {deps}", file=file)
    else:
        print("\n(nothing to build — targets already supplied)", file=file)

    if supplied:
        print(f"\nSupplied: {', '.join(sorted(supplied))}", file=file)
    if plan.missing:
        print(tty(TTY.RED,
                  text="Missing:  "
                  + ", ".join(f"{m} (--{m})" for m in sorted(plan.missing)),
                  file=file), file=file)


# --------------------------------------------------------------------------- #
# Mermaid
# --------------------------------------------------------------------------- #


def mermaid(reg: ProviderRegistry, targets: Iterable[str] | None = None) -> str:
    """Return a Mermaid ``graph TD`` description of the DAG (or a target's
    subgraph). Input nodes are tagged so they can be styled distinctly."""
    plan = reg.plan(_scope(reg, targets), [])
    nodes = {s.name for s in plan.steps if s.name not in RESERVED}
    inputs = set(plan.missing)

    lines = ["graph TD"]
    for name in sorted(nodes | inputs):
        lines.append(f'    {name}["{name}"]' + (":::input" if name in inputs else ""))
    for step in plan.steps:
        for dep in _domain_deps(reg, step.name):
            lines.append(f"    {dep} --> {step.name}")
    lines.append("    classDef input fill:#fff3cd,stroke:#d39e00;")
    return "\n".join(lines)


def mermaid_live_url(diagram: str) -> str:
    """Build a https://mermaid.live edit link for *diagram*.

    The diagram is encoded into the URL *fragment* (deflate + base64), so the
    browser renders it client-side — nothing is uploaded to a server.
    """
    state = {"code": diagram, "mermaid": {"theme": "default"}}
    raw = json.dumps(state, separators=(",", ":")).encode("utf-8")
    packed = base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii").rstrip("=")
    return f"https://mermaid.live/edit#pako:{packed}"
