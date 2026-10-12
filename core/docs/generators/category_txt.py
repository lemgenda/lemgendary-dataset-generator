#!/usr/bin/env python3
"""LemGendary category.txt Generator Script.

Generates the authoritative top-level domain classification tag file (category.txt)
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


def generate_category_txt(
    context_or_path: ManifoldContext | str | Path,
    category_str: str | None = None,
) -> Path:
    """Generate and write category.txt for a manifold.

    Args:
        context_or_path: Resolved ManifoldContext or directory path.
        category_str: Optional explicit override for category string.

    Returns:
        Path to the generated category.txt file.
    """
    if isinstance(context_or_path, ManifoldContext):
        target_dir = context_or_path.manifold_path
        cat_text = category_str or context_or_path.category
    else:
        target_dir = Path(context_or_path)
        if category_str:
            cat_text = category_str
        else:
            ctx = resolve_manifold_context(target_dir)
            cat_text = ctx.category

    target_file = target_dir / "category.txt"
    target_file.write_text(f"{cat_text.strip()}\n", encoding="utf-8")
    return target_file


def main() -> None:
    """CLI entrypoint for category.txt generation."""
    parser = argparse.ArgumentParser(description="Generate category.txt for dataset manifolds")
    parser.add_argument("--manifold", type=str, default=None, help="Manifold directory name or path")
    parser.add_argument("--all", action="store_true", help="Generate category.txt for all manifolds")
    parser.add_argument("--category", type=str, default=None, help="Explicit category override")
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
                out = generate_category_txt(m_dir, category_str=args.category)
                print(f"[OK] {m_dir.name} -> {out.name} ({out.read_text(encoding='utf-8').strip()})")
                count += 1
        print(f"Generated category.txt for {count} manifolds.")
    elif args.manifold:
        p = Path(args.manifold)
        if not p.is_dir() and (datasets_root / args.manifold).is_dir():
            p = datasets_root / args.manifold
        out = generate_category_txt(p, category_str=args.category)
        print(f"[OK] {p.name} -> {out.name} ({out.read_text(encoding='utf-8').strip()})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
