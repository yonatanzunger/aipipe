"""Central command-line entry point for aipipe.

All real logic lives in the libraries (``llm.setup`` for configuration, ``dag``
for pipelines); this module only wires those into an argparse command tree.

Each command group registers its subparser and a ``_handler`` via
``set_defaults`` so new groups (e.g. a future ``run`` command backed by
``dag``) can be added without touching the dispatch logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dag.dag import registry, make, LoggerFactory
from dag.loader import import_providers
from pyppin.text.tty import tty, TTY


def _run_config(args: argparse.Namespace) -> int:
    from llm.setup import config_set, config_show, config_unset, run_wizard

    if args.config_command == "setup":
        return 0 if run_wizard() else 1
    if args.config_command == "show":
        config_show()
        return 0
    if args.config_command == "set":
        config_set(args.key, args.value)
        return 0
    if args.config_command == "unset":
        config_unset(args.key)
        return 0
    return 2


def _run_build(args: argparse.Namespace) -> int:
    targets = getattr(args, "target", [])
    if not targets:
        print("Nothing to build.")
        return 0

    # Only pass flags the user actually set: a resource present in the inputs
    # (even as None) is treated by make() as already-supplied, which would skip
    # its provider.
    resources = {
        n: getattr(args, n)
        for n in registry.resources
        if getattr(args, n, None) is not None
    }
    logger_factory = LoggerFactory(resources)
    logger = logger_factory.logger()

    result = make(targets, logger_factory=logger_factory, **resources)

    for target in targets:
        if target in result:
            logger.log(0, target, result[target], truncate=False)
        else:
            logger.log(0, target, tty(TTY.RED, text="NOT PRESENT"))
    return 0


def main(argv: list[str] | None = None) -> int:
    # First pass: pull out --import and load those DAGs, since the resource
    # flags on `make` depend on what gets loaded. add_help=False so this pass
    # doesn't intercept -h/--help — the full parser below owns help.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--import",
        "-i",
        dest="imports",
        action="append",
        default=[],
        metavar="PATH",
        type=Path,
        help="Files or directories from which to read DAGs",
    )
    pre_args, extra = pre.parse_known_args(argv)
    for src in pre_args.imports:
        import_providers(src)

    # Second pass: the real parser. parents=[pre] re-declares --import so it
    # shows in help and parses cleanly on the full argv.
    parser = argparse.ArgumentParser(
        prog="aipipe",
        parents=[pre],
        description="aipipe — DAG pipelines with LLM-backed stages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    cfg = subparsers.add_parser(
        "config", help="Manage LLM provider configuration (keys, models)"
    )
    cfg.set_defaults(_handler=_run_config)
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True)
    cfg_sub.add_parser(
        "setup", help="Interactively configure a provider and test the connection"
    )
    cfg_sub.add_parser("show", help="Show the current configuration")
    set_p = cfg_sub.add_parser(
        "set", help="Set one config/credential value by its env-var name"
    )
    set_p.add_argument("key", help="e.g. ANTHROPIC_API_KEY or AIPIPE_LLM_PROVIDER")
    set_p.add_argument("value")
    unset_p = cfg_sub.add_parser(
        "unset", help="Clear one config/credential value by its env-var name"
    )
    unset_p.add_argument("key", help="e.g. ANTHROPIC_API_KEY")

    build = subparsers.add_parser("make", help="Make one or more target resources")
    build.add_argument("target", nargs="*", help="Targets to build")
    registry.add_arguments(build)
    build.set_defaults(_handler=_run_build)

    # Now actually parse the arguments!
    args = parser.parse_args(extra)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2

    return handler(args)
