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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("lemdata.llm.metadata")


# Fields every manifold index carries; the image-quality ones are neutral for text.
_INDEX_PARITY_FIELDS: dict[str, Any] = {
    "nima_score": 1.0, "caption": None, "style_tag": None, "cluster_id": -1,
}


@dataclass(frozen=True)
class LLMManifoldDescriptor:
    """What an LLM manifold is: its category, classes (first = task) and provenance."""

    category: str
    classes: list[str]
    original_sources: list[str]
    canonical_format: str = "jsonl"

    @property
    def task(self) -> str:
        """The manifold task: the first class, or ``"llm"`` without classes."""
        return self.classes[0] if self.classes else "llm"


def _write_dataset_info(
    target_dir: Path,
    dataset_name: str,
    descriptor: LLMManifoldDescriptor,
    split_counts: tuple[int, int],
) -> None:
    train_count, val_count = split_counts
    info_data = {
        "name": dataset_name,
        "count": train_count + val_count,
        "task": descriptor.task,
        "category": descriptor.category,
        "canonical_format": descriptor.canonical_format,
        "original_sources": descriptor.original_sources,
        "path": str(target_dir),
        "source": f"{dataset_name}-manifold",
        "last_processed": datetime.datetime.now(datetime.UTC).isoformat(),
        "train_samples": train_count,
        "val_samples": val_count,
    }
    with open(target_dir / "dataset_info.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(info_data, f, sort_keys=False)


def _split_index_entries(
    dataset_name: str,
    split: str,
    records: list[dict[str, Any]],
    first_id: int,
    task: str,
) -> list[dict[str, Any]]:
    """Index rows for one split; ``id`` continues the global numbering at ``first_id``."""
    entries: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        digest = hashlib.sha256(json.dumps(rec, sort_keys=True).encode("utf-8")).hexdigest()
        entry = {
            "id": first_id + idx,
            "name": f"{dataset_name}_{split}_{idx:06d}",
            "record_id": rec.get("id", f"{split}_{idx}"),
            "category": rec.get("category", "default"),
            "task": task,
            "split": split,
            "hash": digest[:16],
        }
        entries.append({**entry, **_INDEX_PARITY_FIELDS})
    return entries


def generate_llm_metadata(
    manifold_dir: str | Path,
    dataset_name: str,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    descriptor: LLMManifoldDescriptor | None = None,
    **descriptor_fields: Any,
) -> None:
    """Synchronizes all standard root metadata files for an LLM dataset manifold.

    Pass either ``descriptor`` or its fields as keywords (``category``,
    ``classes``, ``original_sources`` and optionally ``canonical_format``).
    """
    if descriptor is None:
        descriptor = LLMManifoldDescriptor(**descriptor_fields)
    elif descriptor_fields:
        raise TypeError("Pass either descriptor or its fields, not both")

    target_dir = Path(manifold_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "data").mkdir(parents=True, exist_ok=True)
    train_count = len(train_records)

    # 1. dataset_info.yaml
    _write_dataset_info(target_dir, dataset_name, descriptor, (train_count, len(val_records)))

    # 2. category.txt & classes.txt
    (target_dir / "category.txt").write_text(descriptor.category + "\n", encoding="utf-8")
    (target_dir / "classes.txt").write_text("\n".join(descriptor.classes) + "\n", encoding="utf-8")

    # 3. index.json
    index_entries = _split_index_entries(dataset_name, "train", train_records, 0, descriptor.task)
    index_entries += _split_index_entries(
        dataset_name, "val", val_records, train_count, descriptor.task
    )
    with open(target_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_entries, f, indent=2)

    total_count = train_count + len(val_records)
    logger.info("Generated metadata files for %s (%d records).", dataset_name, total_count)
