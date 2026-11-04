#!/usr/bin/env python3
"""LemGendary category.txt Generator Script.

Generates the authoritative top-level domain classification tag file (category.txt)
for a LemGendary dataset manifold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.docs.context import ManifoldContext, resolve_manifold_context
from core.docs.generators.common import manifold_parser, run_manifold_cli


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
        cat_text = category_str or context_or_path.profile.category
    else:
        target_dir = Path(context_or_path)
        if category_str:
            cat_text = category_str
        else:
            ctx = resolve_manifold_context(target_dir)
            cat_text = ctx.profile.category

    target_file = target_dir / "category.txt"
    target_file.write_text(f"{cat_text.strip()}\n", encoding="utf-8")
    return target_file


def _cli_process(manifold_dir: Path, args: argparse.Namespace) -> str:
    """Generate category.txt for one manifold and describe the result."""
    out = generate_category_txt(manifold_dir, category_str=args.category)
    return f"{out.name} ({out.read_text(encoding='utf-8').strip()})"


def main() -> None:
    """CLI entrypoint for category.txt generation."""
    parser = manifold_parser(
        "Generate category.txt for dataset manifolds", "Generate category.txt for all manifolds"
    )
    parser.add_argument("--category", type=str, default=None, help="Explicit category override")
    run_manifold_cli(parser, _cli_process, "Generated category.txt")


if __name__ == "__main__":
    main()
