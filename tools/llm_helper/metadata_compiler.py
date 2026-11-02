"""Generic Metadata & Inventory Synthesizer for LLM Manifolds.

Generates:
1. dataset_info.yaml
2. index.json (sample inventory with sha256 hashes)
3. category.txt & classes.txt
4. dataset-metadata.json (Kaggle frictionless spec)
5. README.md
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger("lemdata.llm.metadata")


def generate_llm_metadata(
    manifold_dir: str | Path,
    dataset_name: str,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    category: str,
    classes: list[str],
    original_sources: list[str],
    title: str = "LemGendized LLM Instruction Dataset",
    subtitle: str = "Instruction Dataset for Causal LLM Fine-Tuning",
    author: str = "Lem Treursic",
    canonical_format: str = "jsonl",
) -> None:
    """Synchronizes all standard root metadata files for an LLM dataset manifold."""
    target_dir = Path(manifold_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    data_dir = target_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_records = train_records + val_records
    total_count = len(all_records)
    train_count = len(train_records)
    val_count = len(val_records)

    # 1. dataset_info.yaml
    info_data = {
        "name": dataset_name,
        "count": total_count,
        "task": classes[0] if classes else "llm",
        "category": category,
        "canonical_format": canonical_format,
        "original_sources": original_sources,
        "path": str(target_dir),
        "source": f"{dataset_name}-manifold",
        "last_processed": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "train_samples": train_count,
        "val_samples": val_count,
    }
    with open(target_dir / "dataset_info.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(info_data, f, sort_keys=False)

    # 2. category.txt & classes.txt
    (target_dir / "category.txt").write_text(category + "\n", encoding="utf-8")
    (target_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    # 3. index.json
    index_entries = []
    for idx, rec in enumerate(train_records):
        content_str = json.dumps(rec, sort_keys=True)
        h = hashlib.sha256(content_str.encode("utf-8")).hexdigest()[:16]
        index_entries.append({
            "id": idx,
            "name": f"{dataset_name}_train_{idx:06d}",
            "record_id": rec.get("id", f"train_{idx}"),
            "category": rec.get("category", "default"),
            "task": classes[0] if classes else "llm",
            "split": "train",
            "hash": h,
            "nima_score": 1.0,
            "caption": None,
            "style_tag": None,
            "cluster_id": -1,
        })

    for idx, rec in enumerate(val_records):
        global_idx = train_count + idx
        content_str = json.dumps(rec, sort_keys=True)
        h = hashlib.sha256(content_str.encode("utf-8")).hexdigest()[:16]
        index_entries.append({
            "id": global_idx,
            "name": f"{dataset_name}_val_{idx:06d}",
            "record_id": rec.get("id", f"val_{idx}"),
            "category": rec.get("category", "default"),
            "task": classes[0] if classes else "llm",
            "split": "val",
            "hash": h,
            "nima_score": 1.0,
            "caption": None,
            "style_tag": None,
            "cluster_id": -1,
        })

    with open(target_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_entries, f, indent=2)

    logger.info("Generated metadata files for %s (%d records).", dataset_name, total_count)

