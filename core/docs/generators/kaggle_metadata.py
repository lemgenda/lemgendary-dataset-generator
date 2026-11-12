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
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from core.docs.context import ManifoldContext
from core.docs.generators.common import (
    count_lines,
    ensure_context,
    manifold_parser,
    run_manifold_cli,
)
from core.docs.metadata import FOREX_COLUMN_FIELDS
from core.kaggle_identity import default_owner
from utils.shard_audit import find_shards, split_member_name

logger = logging.getLogger(__name__)


def extract_kaggle_description(readme_content: str) -> str:
    """Return the text from '## Dataset Overview' up to '## Repository Structure'."""
    start_marker = "## Dataset Overview"
    end_marker = "## Repository Structure"

    start_idx = readme_content.find(start_marker)
    if start_idx != -1:
        end_idx = readme_content.find(end_marker, start_idx)
        if end_idx != -1:
            return readme_content[start_idx:end_idx].strip()
        return readme_content[start_idx:].strip()
    return readme_content.strip()


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

_IMAGES_DESCRIPTION = "Partitioned visual imagery assets and target frames formatted for training."
_DIR_DESCRIPTIONS = {
    "mds": (
        "MosaicML Streaming (MDS) container directory with sharded indexed binary"
        " tensors for cloud multi-node streaming training."
    ),
    "labels": (
        "Pre-computed categorical class labels, bounding ground truths, and train/val"
        " split indices."
    ),
    "images": _IMAGES_DESCRIPTION,
    "imgs": _IMAGES_DESCRIPTION,
    "targets": (
        "High-resolution clean ground truth reference targets paired identically 1-to-1"
        " against input images."
    ),
}

Resource = dict[str, Any]


def _forex_shard_description(year: object) -> str:
    """Describe one annual Forex Parquet shard."""
    return (
        f"Annual OHLCV and feature tensor shard ({year}) with 14 engineered technical"
        " indicators and causal direction targets (Zstandard compressed)."
    )


def _jsonl_description(is_train: bool, line_count: int) -> str:
    """Describe a train or validation JSONL instruction split."""
    if is_train:
        return (
            "Fine-tuning prompt-completion instructions training corpus"
            f" ({line_count:,} instruction pairs)."
        )
    return (
        "Evaluation validation prompt-completion instructions benchmark split"
        f" ({line_count:,} evaluation pairs)."
    )


def _instruction_data_resources(data_dir: Path) -> list[Resource]:
    """Describe each JSONL split inside the ``data/`` folder of an instruction manifold."""
    return [
        {
            "path": f"data/{sub_f.name}",
            "description": _jsonl_description("train" in sub_f.name, count_lines(sub_f)),
        }
        for sub_f in sorted(data_dir.iterdir())
        if sub_f.is_file() and sub_f.name.endswith(".jsonl")
    ]


def _first_sample_members(shard: Path) -> list[str]:
    """Member suffixes of the first sample in ``shard`` (``[".webp"]``); empty when unreadable."""
    suffixes: list[str] = []
    first_key: str | None = None
    try:
        with tarfile.open(shard, "r:") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                key, suffix = split_member_name(member.name)
                if first_key is None:
                    first_key = key
                elif key != first_key:
                    break
                suffixes.append(suffix)
    except (OSError, tarfile.TarError) as exc:
        logger.warning("Cannot read the first sample of shard %s: %s", shard, exc)
        return []
    return suffixes


def _member_text(members: list[str]) -> str:
    """``"one .webp file per sample"`` or ``"2 files per sample (.jpg, .target.webp)"``."""
    if not members:
        return "samples grouped by key"
    if len(members) == 1:
        return f"one {members[0]} file per sample"
    return f"{len(members)} files per sample ({', '.join(members)})"


def _describe_shards(p: Path) -> str:
    """Describe a WebDataset shard folder from the shard files actually present."""
    shards = find_shards(p)
    if not shards:
        return "WebDataset shard folder (no shard files present when this was generated)."
    total_mb = sum(s.stat().st_size for s in shards) / (1024 * 1024)
    extensions = "/".join(sorted({s.suffix for s in shards}))
    splits = sorted({s.parent.name for s in shards if s.parent != p})
    split_text = f" in the {', '.join(splits)} splits" if splits else ""
    return (
        f"WebDataset shards: {len(shards)} tar-format {extensions} files ({total_mb:.1f} MB)"
        f"{split_text}, {_member_text(_first_sample_members(shards[0]))}, for streaming"
        " training."
    )


def _describe_directory(p: Path) -> str:
    """Describe a root-level folder of the manifold."""
    dir_name_lower = p.name.lower()
    if dir_name_lower in ["shards", "webdataset"]:
        return _describe_shards(p)
    return _DIR_DESCRIPTIONS.get(dir_name_lower, f"Directory partition containing {p.name} assets.")


def _count_label(total_samples: int, fallback: str) -> str:
    """Format a sample count, or ``fallback`` when the count is unknown."""
    return f"{total_samples:,}" if total_samples > 0 else fallback


def _describe_index(_p: Path, ctx: ManifoldContext) -> str:
    """Describe index.json."""
    count_str = _count_label(ctx.lineage.total_samples, "compiled")
    return (
        f"Master sample inventory mapping {count_str} samples with split partitions and"
        " source provenance."
    )


def _describe_info_yaml(_p: Path, ctx: ManifoldContext) -> str:
    """Describe dataset_info.yaml."""
    count_str = _count_label(ctx.lineage.total_samples, "manifold")
    return (
        f"Machine-readable YAML manifest detailing sample counts ({count_str} samples),"
        " verified source lineage, and container topology."
    )


def _describe_classes(p: Path, _ctx: ManifoldContext) -> str:
    """Describe classes.txt with its number of classes."""
    try:
        cls_lines = len(p.read_text(encoding="utf-8").strip().splitlines())
    except OSError:
        return "Exhaustive newline-delimited class dictionary."
    return f"Exhaustive newline-delimited class dictionary ({cls_lines} target classes)."


def _describe_category(_p: Path, ctx: ManifoldContext) -> str:
    """Describe category.txt."""
    return f"Top-level domain taxonomy classification descriptor ('{ctx.profile.category}')."


def _describe_readme(_p: Path, _ctx: ManifoldContext) -> str:
    """Describe README.md."""
    return "Authoritative manifold documentation, lineage matrix, and SOTA metric targets."


_NAMED_FILE_DESCRIPTIONS: dict[str, Callable[[Path, ManifoldContext], str]] = {
    "index.json": _describe_index,
    "dataset_info.yaml": _describe_info_yaml,
    "classes.txt": _describe_classes,
    "category.txt": _describe_category,
    "readme.md": _describe_readme,
}


def _describe_banner(p: Path) -> str:
    """Describe the cover image with its pixel size."""
    try:
        with Image.open(p) as im:
            w, h = im.size
    except (OSError, Image.DecompressionBombError) as exc:
        logger.debug("Could not read banner size of %s (%s); using the canonical size", p, exc)
        return "Canonical 564x284 architecture banner and visual sample preview."
    return f"Canonical {w}x{h} architecture banner and visual sample preview."


def _describe_file(p: Path, ctx: ManifoldContext) -> tuple[str, dict[str, Any] | None]:
    """Describe a root-level file; also return its column schema for Forex shards."""
    fn_lower = p.name.lower()
    named = _NAMED_FILE_DESCRIPTIONS.get(fn_lower)
    if named is not None:
        return named(p, ctx), None
    if fn_lower.endswith(_IMAGE_SUFFIXES):
        return _describe_banner(p), None
    if fn_lower.endswith(".ipynb"):
        platform = "Google Colab" if "colab" in fn_lower else "Kaggle"
        return (
            f"Standalone {platform} GPU training notebook configured for accelerated cloud"
            " execution."
        ), None
    if fn_lower.startswith("forexuniverse") and fn_lower.endswith(".parquet"):
        shard_year = p.name.replace("ForexUniverse", "").replace(".parquet", "")
        return _forex_shard_description(shard_year), {"fields": FOREX_COLUMN_FIELDS}
    if fn_lower.endswith(".jsonl"):
        return _jsonl_description("train" in fn_lower, count_lines(p)), None
    return f"{p.name} data asset.", None


def _sorted_root_entries(output_root: Path) -> list[Path]:
    """Return visible root entries (folders first) excluding dataset-metadata.json."""
    return sorted(
        [
            p for p in output_root.iterdir()
            if not p.name.startswith(".") and p.name != "dataset-metadata.json"
        ],
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )


def _describe_entry(p: Path, ctx: ManifoldContext) -> Resource:
    """Return the Frictionless resource descriptor of one root-level entry."""
    schema = None
    if p.is_dir():
        desc = _describe_directory(p)
    else:
        desc, schema = _describe_file(p, ctx)
    res_entry: Resource = {"path": p.name, "description": desc}
    if schema:
        res_entry["schema"] = schema
    return res_entry


def introspect_root_resources(
    output_root: Path,
    ctx: ManifoldContext,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return Frictionless resource descriptors for every file and directory in output_root."""
    resources: list[Resource] = []

    # For Forex manifolds, prioritize annual Parquet data shards at the very top of resources
    if ctx.task == "forex":
        resources = [
            {
                "path": f"ForexUniverse{y}.parquet",
                "description": _forex_shard_description(y),
                "schema": {"fields": FOREX_COLUMN_FIELDS},
            }
            for y in range(2019, 2027)
        ]
    seen_paths = {r["path"] for r in resources}

    if not output_root.exists() or not output_root.is_dir():
        return resources, None

    entries = _sorted_root_entries(output_root)
    header_image = next(
        (p.name for p in entries if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES), None
    )

    for p in entries:
        if p.name in seen_paths:
            continue
        if p.is_dir() and p.name.lower() == "data" and ctx.is_llm:
            data_resources = _instruction_data_resources(p)
        else:
            data_resources = [_describe_entry(p, ctx)]
        resources += data_resources
        seen_paths.update(r["path"] for r in data_resources)

    return resources, header_image


def _resolve_readme(ctx: ManifoldContext, readme_content: str | None) -> str:
    """Return the README text used for the Kaggle description."""
    if readme_content is not None:
        return readme_content
    readme_file = ctx.manifold_path / "README.md"
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return ctx.profile.description


def _optional_fields(ctx: ManifoldContext, header_image: str | None) -> dict[str, Any]:
    """Return the optional Kaggle fields that have a value."""
    candidates = {
        "headerImage": header_image,
        "provenanceSources": ctx.lineage.provenance_sources,
        "collectionMethodology": ctx.lineage.collection_methodology,
        "citations": ctx.lineage.citations,
        "coverage": ctx.lineage.coverage,
    }
    return {key: value for key, value in candidates.items() if value}


def _tabular_data_files(resources: list[Resource]) -> list[dict[str, Any]]:
    """Return column schemas of tabular resources for direct Kaggle settings sync."""
    return [
        {
            "name": r["path"],
            "description": r.get("description", ""),
            "columns": r["schema"]["fields"],
        }
        for r in resources
        if "schema" in r and r.get("schema", {}).get("fields")
    ]


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
    ctx = ensure_context(context_or_path)
    resources, header_image = introspect_root_resources(ctx.manifold_path, ctx)
    listing = ctx.listing

    metadata_payload: dict[str, Any] = {
        "title": listing.title,
        "id": f"{default_owner()}/{ctx.slug}",
        "subtitle": listing.subtitle,
        "description": extract_kaggle_description(_resolve_readme(ctx, readme_content)),
        "keywords": listing.keywords,
        "licenses": [{"name": listing.license_name}],
        "isPrivate": listing.is_private,
        "expectedUpdateFrequency": listing.update_frequency,
        "author": listing.author.name,
        "authors": [
            {
                "name": listing.author.name,
                "bio": listing.author.bio,
                "role": "Author",
            }
        ],
        "resources": resources,
    }
    # Header image falls back to the context cover image name
    metadata_payload.update(
        _optional_fields(ctx, header_image or ctx.inventory.cover_image_name)
    )
    data_files = _tabular_data_files(resources)
    if data_files:
        metadata_payload["data"] = data_files

    target_file = ctx.manifold_path / "dataset-metadata.json"
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return metadata_payload


def _cli_process(manifold_dir: Path, _args: argparse.Namespace) -> str:
    """Generate dataset-metadata.json for one manifold and describe the result."""
    out = generate_kaggle_metadata(manifold_dir)
    return f"{out.get('title')} (resources={len(out.get('resources', []))})"


def main() -> None:
    """CLI entrypoint for dataset-metadata.json generation."""
    parser = manifold_parser(
        "Generate dataset-metadata.json for dataset manifolds",
        "Generate dataset-metadata.json for all manifolds",
    )
    run_manifold_cli(parser, _cli_process, "Generated dataset-metadata.json")


if __name__ == "__main__":
    main()
