#!/usr/bin/env python3
"""LemGendary classes.txt Generator Script.

Generates the authoritative newline-delimited class label dictionary (classes.txt)
for a LemGendary dataset manifold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.docs.context import ManifoldContext, resolve_manifold_context
from core.docs.generators.common import manifold_parser, run_manifold_cli
from core.source_labels import SOURCE_LABEL_SCHEMES, MaskScheme
from core.source_manifolds import registry_entry_for_folder


def _mask_class_names(manifold_name: str) -> list[str] | None:
    """The class names of the registry entry's mask scheme (``source_labels``), if any."""
    entry = registry_entry_for_folder(manifold_name) or {}
    spec = entry.get("source_labels") or {}
    scheme = SOURCE_LABEL_SCHEMES.get(str(spec.get("scheme", "")))
    return list(scheme.class_names) if isinstance(scheme, MaskScheme) else None


def resolve_classes_list(manifold_name: str, task: str) -> list[str]:
    """Resolve the accurate list of target classes for a specific manifold and task."""
    mask_classes = _mask_class_names(manifold_name)
    if mask_classes:
        return mask_classes

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


def _cli_process(manifold_dir: Path, _args: argparse.Namespace) -> str:
    """Generate classes.txt for one manifold and describe the result."""
    out = generate_classes_txt(manifold_dir)
    line_count = len(out.read_text(encoding="utf-8").strip().splitlines())
    return f"{out.name} ({line_count} classes)"


def main() -> None:
    """CLI entrypoint for classes.txt generation."""
    parser = manifold_parser(
        "Generate classes.txt for dataset manifolds", "Generate classes.txt for all manifolds"
    )
    run_manifold_cli(parser, _cli_process, "Generated classes.txt")


if __name__ == "__main__":
    main()
