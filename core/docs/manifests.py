"""Manifest serializers for LemGendary Dataset Compiler.

Generates and serializes dataset manifests including index.json, dataset_info.yaml,
category.txt, classes.txt, and Kaggle Frictionless dataset-metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import yaml

from core.docs.metadata import FOREX_COLUMN_FIELDS


def write_index_json(output_root: Path, final_index: list[dict[str, Any]] | None) -> None:
    """Serialize compiled sample index to index.json."""
    if final_index is not None:
        with open(output_root / "index.json", "w", encoding="utf-8") as f:
            json.dump(final_index, f, indent=2)


def write_dataset_info_yaml(output_root: Path, info_fields: dict[str, Any]) -> None:
    """Serialize dataset specification to dataset_info.yaml."""
    yaml_path = output_root / "dataset_info.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            info_fields,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            explicit_start=True,
        )


def write_category_txt(output_root: Path, category_str: str) -> None:
    """Write category classification tag to category.txt."""
    with open(output_root / "category.txt", "w", encoding="utf-8") as f:
        f.write(f"{category_str.strip()}\n")


def write_classes_txt(output_root: Path, task_key: str) -> None:
    """Write class label mappings to classes.txt."""
    with open(output_root / "classes.txt", "w", encoding="utf-8") as f:
        if task_key == "forex":
            f.write("SELL\nHOLD\nBUY\n")
        else:
            class_name = "face" if task_key == "pose" else task_key
            f.write(f"{class_name}\n")


def write_kaggle_metadata(
    output_root: Path,
    manifold_name: str,
    cat_str: str,
    readme_content: str,
    task_key: str,
) -> dict[str, Any]:
    """Generate and write Kaggle Frictionless dataset-metadata.json."""
    slug = manifold_name.lower().replace("_", "")
    resources: list[dict[str, Any]] = []
    is_forex = (task_key == "forex")

    if is_forex:
        for y in range(2019, 2027):
            resources.append({
                "path": f"{manifold_name}/ForexUniverse{y}.parquet",
                "description": f"Annual OHLCV and feature tensor shards for year {y}",
                "schema": {
                    "fields": FOREX_COLUMN_FIELDS
                },
            })
            resources.append({
                "path": f"ForexUniverse{y}.parquet",
                "description": f"Annual OHLCV and feature tensor shards for year {y}",
                "schema": {
                    "fields": FOREX_COLUMN_FIELDS
                },
            })

    subtitle = f"High-fidelity manifold for {cat_str} machine learning models"
    if len(subtitle) > 80:
        subtitle = f"High-fidelity manifold for {cat_str} models"
    if len(subtitle) > 80:
        subtitle = subtitle[:77] + "..."
    if len(subtitle) < 20:
        subtitle = "High-fidelity machine learning training manifold"

    metadata_payload: dict[str, Any] = {
        "title": manifold_name.replace("Large", "").replace("LemGendized", "LemGendized "),
        "id": f"lemtreursi/{slug}",
        "subtitle": subtitle,
        "description": readme_content,
        "licenses": [{"name": "CC0-1.0"}],
        "resources": resources,
    }

    with open(output_root / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return metadata_payload
