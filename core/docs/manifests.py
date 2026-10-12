"""Manifest serializers for LemGendary Dataset Compiler (Compatibility Layer).

Provides backward-compatible facades delegating to the modular generators subpackage:
- core.docs.generators.category_txt
- core.docs.generators.classes_txt
- core.docs.generators.cover_image
- core.docs.generators.dataset_info_yaml
- core.docs.generators.index_json
- core.docs.generators.kaggle_metadata
- core.docs.generators.readme_md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.docs.context import find_unified_entry, resolve_manifold_context
from core.docs.generators.category_txt import generate_category_txt
from core.docs.generators.classes_txt import generate_classes_txt
from core.docs.generators.dataset_info_yaml import generate_dataset_info_yaml
from core.docs.generators.index_json import generate_index_json
from core.docs.generators.kaggle_metadata import (
    extract_kaggle_description,
    generate_kaggle_metadata,
    introspect_root_resources,
)

# Maintain alias for backward compatibility
find_dataset_entry = find_unified_entry


def write_index_json(output_root: Path, final_index: list[dict[str, Any]] | None) -> None:
    """Serialize compiled sample index to index.json."""
    generate_index_json(output_root, final_index=final_index)


def write_dataset_info_yaml(output_root: Path, info_fields: dict[str, Any]) -> None:
    """Serialize dataset specification to dataset_info.yaml."""
    generate_dataset_info_yaml(output_root, overrides=info_fields)


def write_category_txt(output_root: Path, category_str: str) -> None:
    """Write category classification tag to category.txt."""
    generate_category_txt(output_root, category_str=category_str)


def write_classes_txt(output_root: Path, task_key: str) -> None:
    """Write class label mappings to classes.txt."""
    generate_classes_txt(output_root)


def _introspect_root_resources(
    output_root: Path,
    manifold_name: str,
    task_key: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Introspect files and subdirectories directly in output_root and return Frictionless resource descriptors."""
    ctx = resolve_manifold_context(output_root, pascal_name=manifold_name)
    return introspect_root_resources(output_root, ctx)


def write_kaggle_metadata(
    output_root: Path,
    manifold_name: str,
    cat_str: str,
    readme_content: str,
    task_key: str,
    dataset_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and write Kaggle Frictionless dataset-metadata.json."""
    ctx = resolve_manifold_context(output_root, pascal_name=manifold_name)
    return generate_kaggle_metadata(ctx, readme_content=readme_content)
