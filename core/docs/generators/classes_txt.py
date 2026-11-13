#!/usr/bin/env python3
"""LemGendary classes.txt Generator Script.

Generates the authoritative newline-delimited class label dictionary (classes.txt)
for a LemGendary dataset manifold. The class names come from, in order: the
registry entry's mask scheme (``source_labels``) or class scheme
(``class_labels``), the ``classes`` a label repair recorded in
dataset_info.yaml, then the task heuristics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from core.class_labels import CLASS_SCHEMES
from core.docs.context import ManifoldContext, resolve_manifold_context
from core.docs.generators.common import manifold_parser, run_manifold_cli
from core.source_labels import SOURCE_LABEL_SCHEMES, MaskScheme
from core.source_manifolds import registry_entry_for_folder

logger = logging.getLogger(__name__)


def _scheme_class_names(manifold_name: str) -> list[str] | None:
    """The class names of the registry entry's mask or class scheme, if any."""
    entry = registry_entry_for_folder(manifold_name) or {}
    spec = entry.get("source_labels") or {}
    scheme = SOURCE_LABEL_SCHEMES.get(str(spec.get("scheme", "")))
    if isinstance(scheme, MaskScheme):
        return list(scheme.class_names)
    class_scheme = CLASS_SCHEMES.get(str((entry.get("class_labels") or {}).get("scheme", "")))
    return list(class_scheme.class_names) if class_scheme is not None else None


def _recorded_classes(manifold_dir: Path | None) -> list[str] | None:
    """The ``classes`` a label repair recorded in the manifold's dataset_info.yaml."""
    info = manifold_dir / "dataset_info.yaml" if manifold_dir is not None else None
    if info is None or not info.is_file():
        return None
    try:
        data = yaml.safe_load(info.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Cannot read classes from %s: %s", info, exc)
        return None
    classes = data.get("classes") if isinstance(data, dict) else None
    if isinstance(classes, list) and classes and all(isinstance(c, str) for c in classes):
        return [str(c) for c in classes]
    return None


def resolve_classes_list(manifold_name: str, task: str,
                         manifold_dir: Path | None = None) -> list[str]:
    """Resolve the accurate list of target classes for a specific manifold and task."""
    known = _scheme_class_names(manifold_name) or _recorded_classes(manifold_dir)
    if known:
        return known

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
            context_or_path.manifold_name, context_or_path.task, target_dir
        )
    else:
        target_dir = Path(context_or_path)
        if explicit_classes:
            classes_list = explicit_classes
        else:
            ctx = resolve_manifold_context(target_dir)
            classes_list = resolve_classes_list(ctx.manifold_name, ctx.task, target_dir)

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
