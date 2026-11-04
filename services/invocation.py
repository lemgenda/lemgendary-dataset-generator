"""Typed argument contract for running a service entry point in its own process.

Long-running or isolated work (compiles, degradation, LLM compiles, queue
runs, Kaggle transfers) runs as ``python -m <service module> <args>``. The
arguments are built from an option table (:data:`FlagTable`), and the
service's ``main(argv)`` parses them back with a parser built from the same
table, so the job argv and the entry point cannot drift apart.

This module only describes *what* to run; ``api.jobs`` picks the interpreter
and is the one place that spawns the process.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "Flag",
    "FlagTable",
    "ServiceInvocation",
    "add_flags",
    "dispatch",
    "flag_args",
    "parsed_options",
]

#: How an option becomes arguments:
#:   ``text``   -> ``flag value`` when the value is truthy
#:   ``int`` / ``float`` -> ``flag str(value)`` when the value is not None
#:   ``switch`` -> ``flag`` when the value is truthy
FlagKind = Literal["text", "int", "float", "switch"]


@dataclass(frozen=True)
class Flag:
    """One option of a service entry point: option name, CLI flag and kind."""

    name: str
    flag: str
    kind: FlagKind


FlagTable = tuple[Flag, ...]


@dataclass(frozen=True)
class ServiceInvocation:
    """A service entry point to run in its own process: ``python -m module *args``."""

    module: str
    args: tuple[str, ...] = ()

    def argv(self, python: str) -> list[str]:
        """The full command line for interpreter ``python`` (unbuffered output)."""
        return [python, "-u", "-m", self.module, *self.args]


def flag_args(values: Mapping[str, Any], table: FlagTable) -> list[str]:
    """Translate option ``values`` into arguments, in table order.

    Raises:
        TypeError: when ``values`` holds an option the table does not define.
    """
    unknown = set(values) - {f.name for f in table}
    if unknown:
        raise TypeError(f"Unknown option(s): {', '.join(sorted(unknown))}")
    args: list[str] = []
    for spec in table:
        value = values.get(spec.name)
        if spec.kind in ("int", "float"):
            if value is not None:
                args.extend([spec.flag, str(value)])
        elif value:
            args.extend([spec.flag] if spec.kind == "switch" else [spec.flag, str(value)])
    return args


_ARG_TYPES: dict[FlagKind, type] = {"text": str, "int": int, "float": float}


def add_flags(parser: argparse.ArgumentParser, table: FlagTable) -> argparse.ArgumentParser:
    """Add one argument per table entry (``dest`` = option name, unset = ``None``/False)."""
    for spec in table:
        if spec.kind == "switch":
            parser.add_argument(spec.flag, dest=spec.name, action="store_true")
        else:
            parser.add_argument(spec.flag, dest=spec.name, type=_ARG_TYPES[spec.kind])
    return parser


def parsed_options(
    namespace: argparse.Namespace, table: FlagTable, *, keep_unset: Sequence[str] = ()
) -> dict[str, Any]:
    """The table options set in ``namespace`` (``None`` values dropped unless kept).

    Dropping unset options lets the service function apply its own defaults,
    exactly as when the CLI calls it in-process without those keywords.
    """
    values = {spec.name: getattr(namespace, spec.name) for spec in table}
    return {k: v for k, v in values.items() if v is not None or k in keep_unset}


def dispatch(
    subcommands: Mapping[str, Callable[[list[str]], int]], argv: list[str] | None = None
) -> int:
    """Run the subcommand named by ``argv[0]`` (``sys.argv`` when omitted); 2 on a bad name."""
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in subcommands:
        prog = Path(sys.argv[0]).name
        print(f"usage: {prog} {{{','.join(subcommands)}}} [args]", file=sys.stderr)
        return 2
    return subcommands[args[0]](args[1:])
