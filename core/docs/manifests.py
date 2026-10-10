"""Manifest serializers for LemGendary Dataset Compiler.

Generates and serializes dataset manifests including index.json, dataset_info.yaml,
category.txt, classes.txt, and Kaggle Frictionless dataset-metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import yaml

from core.docs.metadata import FOREX_COLUMN_FIELDS, UNIFIED_DATA


def extract_kaggle_description(readme_content: str) -> str:
    """Extract section from '## Dataset Overview' (inclusive) up to '## Repository Structure' (exclusive)."""
    start_marker = "## Dataset Overview"
    end_marker = "## Repository Structure"

    start_idx = readme_content.find(start_marker)
    if start_idx != -1:
        end_idx = readme_content.find(end_marker, start_idx)
        if end_idx != -1:
            return readme_content[start_idx:end_idx].strip()
        return readme_content[start_idx:].strip()
    return readme_content.strip()


def find_dataset_entry(manifold_name: str) -> dict[str, Any] | None:
    """Find matching dataset entry from unified_data.yaml."""
    datasets = UNIFIED_DATA.get("datasets", {})
    if not isinstance(datasets, dict):
        return None

    norm_target = manifold_name.lower().replace("_", "").replace("large", "")

    for d_key, d_info in datasets.items():
        if not isinstance(d_info, dict):
            continue
        mod_folder = str(d_info.get("modernized_folder", "")).lower()
        d_name = str(d_info.get("name", d_key)).lower()
        if mod_folder == manifold_name.lower():
            return d_info
        if f"lemgendized{d_name}" == manifold_name.lower():
            return d_info
        if f"lemgendized{d_name}large" == manifold_name.lower():
            return d_info

    for d_key, d_info in datasets.items():
        if not isinstance(d_info, dict):
            continue
        d_name = str(d_info.get("name", d_key)).lower()
        norm_key = d_key.lower().replace("_", "")
        norm_name = d_name.replace("_", "")
        if norm_key == norm_target or norm_name == norm_target:
            return d_info
        if norm_key in norm_target or norm_target in norm_key:
            return d_info

    return None


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


def _introspect_root_resources(
    output_root: Path,
    manifold_name: str,
    task_key: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Introspect files and subdirectories directly in output_root and return Frictionless resource descriptors."""
    resources: list[dict[str, Any]] = []
    header_image: str | None = None
    is_forex = (task_key == "forex")

    if not output_root.exists() or not output_root.is_dir():
        return resources, header_image

    entries = sorted(
        [
            p
            for p in output_root.iterdir()
            if not p.name.startswith(".") and p.name != "dataset-metadata.json"
        ],
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )

    for p in entries:
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            header_image = p.name
            break

    seen_paths: set[str] = set()

    for p in entries:
        name = p.name
        rel_path = name
        is_dir = p.is_dir()

        desc = ""
        schema = None

        if is_dir:
            dir_name_lower = name.lower()
            if dir_name_lower == "mds":
                desc = "MosaicML Streaming (MDS) container directory with sharded indexed binary tensors for cloud multi-node streaming training."
            elif dir_name_lower in ["shards", "webdataset"]:
                desc = "Sharded WebDataset tar archive hierarchy containing paired image-tensor samples for high-throughput streaming."
            elif dir_name_lower == "labels":
                desc = "Pre-computed categorical class labels, bounding ground truths, and train/val split indices."
            elif dir_name_lower in ["images", "imgs"]:
                desc = "Partitioned visual imagery assets and target frames formatted for training."
            else:
                desc = f"Directory partition containing {name} assets."
        else:
            fn_lower = name.lower()
            if fn_lower == "index.json":
                desc = "Master sample inventory mapping cryptographic hashes to shard indices and class IDs."
            elif fn_lower == "dataset_info.yaml":
                desc = "Machine-readable YAML manifest detailing sample counts, sha256 checksums, and container topology."
            elif fn_lower == "classes.txt":
                desc = "Exhaustive newline-delimited class dictionary."
            elif fn_lower == "category.txt":
                desc = "Domain taxonomy descriptor defining the task archetype."
            elif fn_lower == "readme.md":
                desc = "Authoritative manifold documentation, lineage matrix, and SOTA metric targets."
            elif fn_lower.endswith(".jpg") or fn_lower.endswith(".png"):
                desc = "Canonical 564x284 architecture banner and visual sample preview."
            elif "colab" in fn_lower and fn_lower.endswith(".ipynb"):
                desc = "Standalone Google Colab GPU training notebook configured for accelerated cloud execution."
            elif fn_lower.endswith(".ipynb"):
                desc = "Standalone Kaggle GPU training notebook configured for accelerated cloud execution."
            elif fn_lower.startswith("forexuniverse") and fn_lower.endswith(".parquet"):
                shard_year = name.replace("ForexUniverse", "").replace(".parquet", "")
                desc = f"Annual OHLCV and feature tensor shards with typed column schemas ({shard_year})."
                schema = {"fields": FOREX_COLUMN_FIELDS}
            elif fn_lower == "ai_helper_train.jsonl":
                desc = "Fine-tuning prompt-completion instructions and domain training corpus."
            elif fn_lower == "ai_helper_val.jsonl":
                desc = "Evaluation validation prompt-completion instructions and benchmark test splits."
            else:
                desc = f"{name} data asset."

        res_entry: dict[str, Any] = {
            "path": rel_path,
            "description": desc,
        }
        if schema:
            res_entry["schema"] = schema

        resources.append(res_entry)
        seen_paths.add(rel_path)

    if is_forex:
        for y in range(2019, 2027):
            pq_name = f"ForexUniverse{y}.parquet"
            if pq_name not in seen_paths:
                resources.append({
                    "path": pq_name,
                    "description": f"Annual OHLCV and feature tensor shards for year {y}",
                    "schema": {"fields": FOREX_COLUMN_FIELDS},
                })
            prefixed = f"{manifold_name}/{pq_name}"
            if prefixed not in seen_paths:
                resources.append({
                    "path": prefixed,
                    "description": f"Annual OHLCV and feature tensor shards for year {y}",
                    "schema": {"fields": FOREX_COLUMN_FIELDS},
                })

    return resources, header_image


def write_kaggle_metadata(
    output_root: Path,
    manifold_name: str,
    cat_str: str,
    readme_content: str,
    task_key: str,
    dataset_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and write Kaggle Frictionless dataset-metadata.json."""
    if dataset_info is None:
        dataset_info = find_dataset_entry(manifold_name) or {}

    slug = manifold_name.lower().replace("_", "")

    resources, header_image = _introspect_root_resources(output_root, manifold_name, task_key)

    subtitle = dataset_info.get("subtitle")
    if not subtitle:
        subtitle = f"High-fidelity manifold for {cat_str} machine learning models"
        if len(subtitle) > 80:
            subtitle = f"High-fidelity manifold for {cat_str} models"
        if len(subtitle) > 80:
            subtitle = subtitle[:77] + "..."
        if len(subtitle) < 20:
            subtitle = "High-fidelity machine learning training manifold"

    title = dataset_info.get("title")
    if not title:
        title = manifold_name.replace("Large", "").replace("LemGendized", "LemGendized ")

    keywords = dataset_info.get("keywords", [])
    if not keywords:
        keywords = ["computer-vision", "deep-learning", "image", "artificial-intelligence", "benchmark"]

    description = extract_kaggle_description(readme_content)
    license_name = dataset_info.get("license", "CC-BY-NC-4.0")
    is_private = bool(dataset_info.get("is_private", False))
    update_frequency = dataset_info.get("expected_update_frequency", "never")

    author_name = dataset_info.get("author", "Lem Treursic")
    author_bio = dataset_info.get("author_bio", "Lead AI Architect & Creator of the LemGendary AI Ecosystem")

    metadata_payload: dict[str, Any] = {
        "title": title,
        "id": f"lemtreursi/{slug}",
        "subtitle": subtitle,
        "description": description,
        "keywords": keywords,
        "licenses": [{"name": license_name}],
        "isPrivate": is_private,
        "expectedUpdateFrequency": update_frequency,
        "author": author_name,
        "authors": [
            {
                "name": author_name,
                "bio": author_bio,
                "role": "Author",
            }
        ],
        "resources": resources,
    }

    if header_image:
        metadata_payload["headerImage"] = header_image
    elif dataset_info.get("header_image"):
        metadata_payload["headerImage"] = dataset_info["header_image"]

    prov_sources = dataset_info.get("provenance_sources", [])
    if prov_sources:
        metadata_payload["provenanceSources"] = prov_sources

    methodology = dataset_info.get("collection_methodology")
    if methodology:
        metadata_payload["collectionMethodology"] = methodology

    citations = dataset_info.get("citations", [])
    if citations:
        metadata_payload["citations"] = citations

    coverage = dataset_info.get("coverage", {})
    if coverage:
        metadata_payload["coverage"] = coverage

    with open(output_root / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return metadata_payload

