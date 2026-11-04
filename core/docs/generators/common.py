"""Shared helpers for the LemGendary document generator scripts.

Context coercion, instruction-dataset (JSONL) split discovery, line counting and
the common ``--manifold`` / ``--all`` command-line driver used by every generator.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.docs.context import ManifoldContext, resolve_manifold_context

__all__ = [
    "count_lines",
    "datasets_root",
    "ensure_context",
    "jsonl_split_files",
    "manifold_parser",
    "run_manifold_cli",
]

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def ensure_context(
    context_or_path: ManifoldContext | str | Path, **resolve_kwargs: Any
) -> ManifoldContext:
    """Return ``context_or_path`` as a ManifoldContext, resolving a directory path if needed."""
    if isinstance(context_or_path, ManifoldContext):
        return context_or_path
    return resolve_manifold_context(Path(context_or_path), **resolve_kwargs)


def count_lines(path: Path) -> int:
    """Return the number of lines of a UTF-8 text file (0 when it does not exist)."""
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def jsonl_split_files(ctx: ManifoldContext) -> tuple[Path, Path]:
    """Locate the train/val JSONL files of an instruction manifold."""
    target_dir = ctx.manifold_path
    train_file_rel = ctx.spec.unified_entry.get("train_file")
    val_file_rel = ctx.spec.unified_entry.get("val_file")
    train_file = (
        (target_dir / train_file_rel)
        if train_file_rel
        else (target_dir / "data" / f"{ctx.slug}_train.jsonl")
    )
    val_file = (
        (target_dir / val_file_rel)
        if val_file_rel
        else (target_dir / "data" / f"{ctx.slug}_val.jsonl")
    )
    for base in (target_dir / "data", target_dir):
        if not train_file.exists():
            train_file = base / "ai_helper_train.jsonl"
            val_file = base / "ai_helper_val.jsonl"
    return train_file, val_file


def datasets_root() -> Path:
    """Return the LemGendaryDatasets folder next to (or inside) the project root."""
    cand = _MODULE_ROOT.parent / "LemGendaryDatasets"
    return cand if cand.exists() else _MODULE_ROOT / "LemGendaryDatasets"


def manifold_parser(description: str, all_help: str) -> argparse.ArgumentParser:
    """Return a parser with the shared ``--manifold`` and ``--all`` options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--manifold", type=str, default=None, help="Manifold directory name or path"
    )
    parser.add_argument("--all", action="store_true", help=all_help)
    return parser


def run_manifold_cli(
    parser: argparse.ArgumentParser,
    process: Callable[[Path, argparse.Namespace], str],
    summary: str,
) -> None:
    """Run a generator over ``--manifold`` or ``--all`` manifolds and print one line each.

    Args:
        parser: Parser already configured with ``--manifold`` and ``--all``.
        process: Generates the document for a manifold directory (given the parsed
            arguments) and returns the text printed after ``[OK] <name> -> ``.
        summary: Summary prefix printed as ``<summary> for <n> manifolds.``.
    """
    args = parser.parse_args()
    root = datasets_root()

    if args.all:
        if not root.exists():
            print(f"[ERROR] Datasets directory not found at {root}")
            sys.exit(1)
        count = 0
        for m_dir in sorted(root.iterdir()):
            if m_dir.is_dir() and not m_dir.name.startswith("."):
                print(f"[OK] {m_dir.name} -> {process(m_dir, args)}")
                count += 1
        print(f"{summary} for {count} manifolds.")
    elif args.manifold:
        p = Path(args.manifold)
        if not p.is_dir() and (root / args.manifold).is_dir():
            p = root / args.manifold
        print(f"[OK] {p.name} -> {process(p, args)}")
    else:
        parser.print_help()
