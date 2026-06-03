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


def _add_config_commands(subparsers: argparse._SubParsersAction) -> None:
    """`aipipe config {setup,show,set}` — manage LLM provider configuration."""
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


def _run_config(args: argparse.Namespace) -> int:
    from llm.setup import config_set, config_show, run_wizard

    if args.config_command == "setup":
        return 0 if run_wizard() else 1
    if args.config_command == "show":
        config_show()
        return 0
    if args.config_command == "set":
        config_set(args.key, args.value)
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aipipe",
        description="aipipe — DAG pipelines with LLM-backed stages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_config_commands(subparsers)
    # Future: dag's runner registers an `aipipe run` command here.
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2
    return handler(args)
