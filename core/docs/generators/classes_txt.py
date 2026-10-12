#!/usr/bin/env python3
"""LemGendary classes.txt Generator Script.

Generates the authoritative newline-delimited class label dictionary (classes.txt)
for a LemGendary dataset manifold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from core.docs.context import ManifoldContext, resolve_manifold_context


def resolve_classes_list(manifold_name: str, task: str) -> list[str]:
    """Resolve the accurate list of target classes for a specific manifold and task."""
    name_lower = manifold_name.lower()

    if task == "forex" or "forex" in name_lower:
        return ["SELL", "HOLD", "BUY"]

    if task == "pose":
        return ["face"]

    return [task]


def generate_classes_txt(
    context_or_path: ManifoldContext | str | Path,
    explicit_classes: list[str] | None = None,
) -> Path:
    """Generate and write classes.txt for a manifold.

    Args:
        context_or_path: Resolved ManifoldContext or directory path.
        explicit_classes: Optional explicit override for class list.

    Returns:
        Path to the generated classes.txt file.
    """
    if isinstance(context_or_path, ManifoldContext):
        target_dir = context_or_path.manifold_path
        classes_list = explicit_classes or resolve_classes_list(
            context_or_path.manifold_name, context_or_path.task
        )
    else:
        target_dir = Path(context_or_path)
        if explicit_classes:
            classes_list = explicit_classes
        else:
            ctx = resolve_manifold_context(target_dir)
            classes_list = resolve_classes_list(ctx.manifold_name, ctx.task)

    target_file = target_dir / "classes.txt"
    content = "\n".join(c.strip() for c in classes_list) + "\n"
    target_file.write_text(content, encoding="utf-8")
    return target_file


def main() -> None:
    """CLI entrypoint for classes.txt generation."""
    parser = argparse.ArgumentParser(description="Generate classes.txt for dataset manifolds")
    parser.add_argument("--manifold", type=str, default=None, help="Manifold directory name or path")
    parser.add_argument("--all", action="store_true", help="Generate classes.txt for all manifolds")
    args = parser.parse_args()

    cand = _MODULE_ROOT.parent / "LemGendaryDatasets"
    datasets_root = cand if cand.exists() else _MODULE_ROOT / "LemGendaryDatasets"

    if args.all:
        if not datasets_root.exists():
            print(f"[ERROR] Datasets directory not found at {datasets_root}")
            sys.exit(1)
        count = 0
        for m_dir in sorted(datasets_root.iterdir()):
            if m_dir.is_dir() and not m_dir.name.startswith("."):
                out = generate_classes_txt(m_dir)
                line_count = len(out.read_text(encoding="utf-8").strip().splitlines())
                print(f"[OK] {m_dir.name} -> {out.name} ({line_count} classes)")
                count += 1
        print(f"Generated classes.txt for {count} manifolds.")
    elif args.manifold:
        p = Path(args.manifold)
        if not p.is_dir() and (datasets_root / args.manifold).is_dir():
            p = datasets_root / args.manifold
        out = generate_classes_txt(p)
        line_count = len(out.read_text(encoding="utf-8").strip().splitlines())
        print(f"[OK] {p.name} -> {out.name} ({line_count} classes)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
