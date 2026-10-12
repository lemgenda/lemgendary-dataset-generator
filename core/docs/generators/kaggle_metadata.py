#!/usr/bin/env python3
"""LemGendary dataset-metadata.json Generator Script.

Generates the complete Frictionless Kaggle metadata manifest (dataset-metadata.json)
for Kaggle API synchronization, including titles, subtitles, header cover image,
provenance sources, citations, and exhaustive descriptions for each file and folder
in the dataset root directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any
from PIL import Image

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from core.docs.context import ManifoldContext, resolve_manifold_context
from core.docs.metadata import FOREX_COLUMN_FIELDS

logger = logging.getLogger(__name__)


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


def introspect_root_resources(
    output_root: Path,
    ctx: ManifoldContext,
) -> tuple[list[dict[str, Any]], str | None]:
    """Introspect every file and directory in output_root and generate accurate Frictionless descriptors."""
    resources: list[dict[str, Any]] = []
    header_image: str | None = None
    is_forex = (ctx.task == "forex")
    seen_paths: set[str] = set()

    # For Forex manifolds, prioritize annual Parquet data shards at the very top of resources
    if is_forex:
        for y in range(2019, 2027):
            pq_name = f"ForexUniverse{y}.parquet"
            resources.append({
                "path": pq_name,
                "description": f"Annual OHLCV and feature tensor shard ({y}) with 14 engineered technical indicators and causal direction targets (Zstandard compressed).",
                "schema": {"fields": FOREX_COLUMN_FIELDS},
            })
            seen_paths.add(pq_name)

    if not output_root.exists() or not output_root.is_dir():
        return resources, header_image

    entries = sorted(
        [
            p for p in output_root.iterdir()
            if not p.name.startswith(".") and p.name != "dataset-metadata.json"
        ],
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )

    # Locate header image
    for p in entries:
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            header_image = p.name
            break

    for p in entries:
        name = p.name
        rel_path = name
        if rel_path in seen_paths:
            continue
        is_dir = p.is_dir()

        desc = ""
        schema = None

        if is_dir:
            dir_name_lower = name.lower()
            if dir_name_lower in ["shards", "webdataset"]:
                tar_count = sum(1 for _ in p.glob("*.tar"))
                total_mb = sum(f.stat().st_size for f in p.glob("*.tar")) / (1024 * 1024)
                if tar_count > 0:
                    desc = f"Sharded WebDataset container directory containing {tar_count} tar archives ({total_mb:.1f} MB) with paired image-tensor samples for high-throughput streaming training."
                else:
                    desc = "Sharded WebDataset tar archive hierarchy containing paired image-tensor samples for high-throughput streaming."
            elif dir_name_lower == "mds":
                desc = "MosaicML Streaming (MDS) container directory with sharded indexed binary tensors for cloud multi-node streaming training."
            elif dir_name_lower == "labels":
                desc = "Pre-computed categorical class labels, bounding ground truths, and train/val split indices."
            elif dir_name_lower in ["images", "imgs"]:
                desc = "Partitioned visual imagery assets and target frames formatted for training."
            elif dir_name_lower == "targets":
                desc = "High-resolution clean ground truth reference targets paired identically 1-to-1 against input images."
            else:
                desc = f"Directory partition containing {name} assets."
        else:
            fn_lower = name.lower()
            if fn_lower == "index.json":
                count_str = f"{ctx.total_samples:,}" if ctx.total_samples > 0 else "compiled"
                desc = f"Master sample inventory mapping {count_str} samples with split partitions and source provenance."
            elif fn_lower == "dataset_info.yaml":
                count_str = f"{ctx.total_samples:,}" if ctx.total_samples > 0 else "manifold"
                desc = f"Machine-readable YAML manifest detailing sample counts ({count_str} samples), verified source lineage, and container topology."
            elif fn_lower == "classes.txt":
                try:
                    cls_lines = len(p.read_text(encoding="utf-8").strip().splitlines())
                    desc = f"Exhaustive newline-delimited class dictionary ({cls_lines} target classes)."
                except OSError:
                    desc = "Exhaustive newline-delimited class dictionary."
            elif fn_lower == "category.txt":
                desc = f"Top-level domain taxonomy classification descriptor ('{ctx.category}')."
            elif fn_lower == "readme.md":
                desc = "Authoritative manifold documentation, lineage matrix, and SOTA metric targets."
            elif fn_lower.endswith((".jpg", ".jpeg", ".png")):
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                    desc = f"Canonical {w}x{h} architecture banner and visual sample preview."
                except Exception:
                    desc = "Canonical 564x284 architecture banner and visual sample preview."
            elif "colab" in fn_lower and fn_lower.endswith(".ipynb"):
                desc = "Standalone Google Colab GPU training notebook configured for accelerated cloud execution."
            elif fn_lower.endswith(".ipynb"):
                desc = "Standalone Kaggle GPU training notebook configured for accelerated cloud execution."
            elif fn_lower.startswith("forexuniverse") and fn_lower.endswith(".parquet"):
                shard_year = name.replace("ForexUniverse", "").replace(".parquet", "")
                desc = f"Annual OHLCV and feature tensor shard ({shard_year}) with 14 engineered technical indicators and causal direction targets (Zstandard compressed)."
                schema = {"fields": FOREX_COLUMN_FIELDS}
            elif fn_lower == "ai_helper_train.jsonl":
                line_cnt = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
                desc = f"Fine-tuning prompt-completion instructions training corpus ({line_cnt:,} instruction pairs)."
            elif fn_lower == "ai_helper_val.jsonl":
                line_cnt = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
                desc = f"Evaluation validation prompt-completion instructions benchmark split ({line_cnt:,} evaluation pairs)."
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

    return resources, header_image


def generate_kaggle_metadata(
    context_or_path: ManifoldContext | str | Path,
    readme_content: str | None = None,
) -> dict[str, Any]:
    """Generate and write Frictionless Kaggle dataset-metadata.json.

    Args:
        context_or_path: Resolved ManifoldContext or directory path.
        readme_content: Optional pre-rendered README string.

    Returns:
        The generated metadata dictionary.
    """
    if isinstance(context_or_path, ManifoldContext):
        ctx = context_or_path
    else:
        ctx = resolve_manifold_context(Path(context_or_path))

    target_dir = ctx.manifold_path

    # Resolve README content for description
    if readme_content is None:
        readme_file = target_dir / "README.md"
        if readme_file.exists():
            readme_content = readme_file.read_text(encoding="utf-8")
        else:
            readme_content = ctx.description

    description = extract_kaggle_description(readme_content)
    resources, header_image = introspect_root_resources(target_dir, ctx)

    # Fallback to context cover image
    if not header_image and ctx.cover_image_name:
        header_image = ctx.cover_image_name

    metadata_payload: dict[str, Any] = {
        "title": ctx.title,
        "id": f"lemtreursi/{ctx.slug}",
        "subtitle": ctx.subtitle,
        "description": description,
        "keywords": ctx.keywords,
        "licenses": [{"name": ctx.license_name}],
        "isPrivate": ctx.is_private,
        "expectedUpdateFrequency": ctx.update_frequency,
        "author": ctx.author_name,
        "authors": [
            {
                "name": ctx.author_name,
                "bio": ctx.author_bio,
                "role": "Author",
            }
        ],
        "resources": resources,
    }

    if header_image:
        metadata_payload["headerImage"] = header_image

    if ctx.provenance_sources:
        metadata_payload["provenanceSources"] = ctx.provenance_sources

    if ctx.collection_methodology:
        metadata_payload["collectionMethodology"] = ctx.collection_methodology

    if ctx.citations:
        metadata_payload["citations"] = ctx.citations

    if ctx.coverage:
        metadata_payload["coverage"] = ctx.coverage

    # Tabular column schemas in 'data' array for direct Kaggle settings sync
    data_files: list[dict[str, Any]] = []
    for r in resources:
        if "schema" in r and r.get("schema", {}).get("fields"):
            data_files.append({
                "name": r["path"],
                "description": r.get("description", ""),
                "columns": r["schema"]["fields"],
            })
    if data_files:
        metadata_payload["data"] = data_files

    target_file = target_dir / "dataset-metadata.json"
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return metadata_payload


def main() -> None:
    """CLI entrypoint for dataset-metadata.json generation."""
    parser = argparse.ArgumentParser(description="Generate dataset-metadata.json for dataset manifolds")
    parser.add_argument("--manifold", type=str, default=None, help="Manifold directory name or path")
    parser.add_argument("--all", action="store_true", help="Generate dataset-metadata.json for all manifolds")
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
                out = generate_kaggle_metadata(m_dir)
                print(f"[OK] {m_dir.name} -> {out.get('title')} (resources={len(out.get('resources', []))})")
                count += 1
        print(f"Generated dataset-metadata.json for {count} manifolds.")
    elif args.manifold:
        p = Path(args.manifold)
        if not p.is_dir() and (datasets_root / args.manifold).is_dir():
            p = datasets_root / args.manifold
        out = generate_kaggle_metadata(p)
        print(f"[OK] {p.name} -> {out.get('title')} (resources={len(out.get('resources', []))})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
