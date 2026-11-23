"""Catalog assembly behind the desktop GUI endpoint ``/gui/datasets/with-stats``.

Joins every compiled manifold folder with its unified_data.yaml registry entry
and dataset_info.yaml, derives display name, formats and compile status, and
splits sample counts across upstream sources (exact JSONL provenance when
available). Folder content scanning lives in :mod:`api.manifold_scan`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from api.manifold_scan import (
    DatasetFileStats,
    calculate_dataset_file_stats,
    extract_jsonl_provenance_counts,
    extract_manifold_provenance_counts,
    jsonl_files,
)
from api.models import (
    DatasetDetailStats,
    DatasetSourceInfo,
    DatasetStatsListResponse,
)
from core import paths
from core.presets import recommended_preset_for_dataset

__all__ = [
    "DatasetFileStats",
    "build_dataset_stats_response",
    "calculate_dataset_file_stats",
    "extract_jsonl_provenance_counts",
    "read_yaml_mapping",
]

logger = logging.getLogger("lemgendary.api.routes.gui.stats")

_GB = 1024**3

_READ_ERRORS = (OSError, UnicodeDecodeError, yaml.YAMLError)

_SOURCE_TYPE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("kaggle", "ava", "coco", "div2k", "flickr", "voc"), "KAGGLE"),
    (("hf", "huggingface", "tad66k", "spaq"), "HUGGINGFACE"),
    (("sub-manifold", "manifold", "multi-task"), "SUB-MANIFOLD"),
    (("terminal", "forex", "mt5"), "TERMINAL"),
    (
        ("documentation", "manual", "manifest", "whitepaper", "glossary", "guide"),
        "DOCUMENTATION",
    ),
)


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """The mapping stored in the YAML file at ``path``.

    Returns ``{}`` (and logs why) when the file is missing, unreadable, not
    UTF-8, not valid YAML, empty, or holds something other than a mapping, so
    one malformed ``dataset_info.yaml`` cannot fail a whole catalog request.
    """
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except _READ_ERRORS as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        logger.warning("Ignoring %s: top level is a %s, not a mapping", path, type(data).__name__)
        return {}
    return data


def as_count(value: Any) -> int:
    """A non-negative sample count from a metadata value (0 when it is not a number)."""
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric sample count %r", value)
        return 0


def _unified_yaml_path() -> Path:
    return paths.UNIFIED_DATA


def _load_folder_index() -> dict[str, tuple[str, dict[str, Any]]]:
    """Map lower-cased manifold folder / dataset names to (registry key, registry entry)."""
    unified_yaml = _unified_yaml_path()
    registry_datasets = read_yaml_mapping(unified_yaml).get("datasets") or {}

    folder_to_meta: dict[str, tuple[str, dict[str, Any]]] = {}
    for k, v in registry_datasets.items():
        m_folder = v.get("modernized_folder") or f"LemGendized{v.get('name', '')}"
        folder_to_meta[m_folder.lower()] = (k, v)
        if "name" in v:
            folder_to_meta[v["name"].lower()] = (k, v)
    return folder_to_meta


def _display_name(folder_name: str, reg_info: dict[str, Any]) -> str:
    display_name = reg_info.get("title")
    if display_name:
        return display_name
    if folder_name == "LemGendizedUpnV2":
        return "Unified Perceptual Net V2"
    if folder_name.startswith("LemGendized"):
        cleaned = folder_name.replace("LemGendized", "")
        return re.sub(r"([A-Z])", r" \1", cleaned).strip()
    return folder_name


@dataclass
class _ManifoldMeta:
    """Task, sample count and formats from the registry, overridden by dataset_info.yaml."""

    task: str
    sample_count: int
    canonical_format: str
    format_val: str


def _read_manifold_meta(info_file: Path, reg_info: dict[str, Any]) -> _ManifoldMeta:
    canonical_format = reg_info.get("canonical_format") or "webdataset"
    result = _ManifoldMeta(
        task=reg_info.get("task", "vision"),
        sample_count=0,
        canonical_format=canonical_format,
        format_val=reg_info.get("container", {}).get("primary") or canonical_format,
    )
    meta = read_yaml_mapping(info_file)
    result.task = meta.get("task", result.task)
    raw_cnt = meta.get("count") or meta.get("total_samples") or meta.get("samples")
    result.sample_count = as_count(raw_cnt)
    if meta.get("canonical_format"):
        result.canonical_format = str(meta["canonical_format"])
    if meta.get("format"):
        result.format_val = str(meta["format"])
    return result


def _read_original_sources(info_file: Path) -> list[Any]:
    sources = read_yaml_mapping(info_file).get("original_sources") or []
    return sources if isinstance(sources, list) else [sources]


def _count_jsonl_lines(jsonl_paths: list[Path]) -> int:
    total = 0
    for jf in jsonl_paths:
        try:
            with open(jf, encoding="utf-8") as jf_in:
                total += sum(1 for line in jf_in if line.strip())
        except OSError as exc:
            logger.debug("Failed counting lines in %s: %s", jf, exc)
    return total


def _build_ref_lookup(refs: list[Any]) -> dict[str, str]:
    ref_lookup: dict[str, str] = {}
    for r_item in refs:
        if isinstance(r_item, dict) and "ref" in r_item:
            r_val = str(r_item["ref"])
            r_name = r_val.rsplit("/", maxsplit=1)[-1]
            ref_lookup[r_name.lower()] = r_val
            ref_lookup[r_val.lower()] = r_val
    return ref_lookup


def _source_type(s_lower: str) -> str:
    for keywords, s_type in _SOURCE_TYPE_RULES:
        if any(k in s_lower for k in keywords):
            return s_type
    return "SOURCE"


def _match_ref(s_lower: str, ref_lookup: dict[str, str]) -> str | None:
    for k_ref, v_ref in ref_lookup.items():
        if k_ref in s_lower or s_lower in k_ref:
            return v_ref
    return None


def _match_provenance_count(s_name: str, provenance_counts: dict[str, int]) -> int | None:
    """Find exact sample count in provenance_counts by exact name or token overlap."""
    if not provenance_counts:
        return None
    # 1. Exact match
    if s_name in provenance_counts:
        return provenance_counts[s_name]

    # 2. Normalized alphanumeric match
    norm_target = re.sub(r"[^a-z0-9]+", "", s_name.lower())
    for p_name, count in provenance_counts.items():
        if re.sub(r"[^a-z0-9]+", "", p_name.lower()) == norm_target:
            return count

    # 3. Token-level overlap (e.g. 'SFHQ Part 3' vs 'synthetic-faces-high-quality-sfhq-part-3')
    target_tokens = set(re.findall(r"[a-z0-9]+", s_name.lower()))
    best_overlap = 0.0
    best_count: int | None = None
    for p_name, count in provenance_counts.items():
        p_tokens = set(re.findall(r"[a-z0-9]+", p_name.lower()))
        common = target_tokens & p_tokens
        if common:
            # Overlap relative to the shorter token set
            overlap = len(common) / min(len(target_tokens), len(p_tokens))
            if overlap > best_overlap and overlap >= 0.5:
                best_overlap = overlap
                best_count = count

    return best_count


def _source_count(
    s_idx: int, s_name: str, sample_count: int, provenance_counts: dict[str, int], n_sources: int
) -> int | None:
    """Exact provenance count from index.json or JSONL, else an even split of ``sample_count``."""
    matched = _match_provenance_count(s_name, provenance_counts)
    if matched is not None:
        return matched
    if sample_count > 0:
        base_count = sample_count // n_sources
        if s_idx == n_sources - 1:
            return sample_count - (base_count * (n_sources - 1))
        return base_count
    return None


def _build_sources(
    raw_sources: list[Any],
    refs: list[Any],
    sample_count: int,
    provenance_counts: dict[str, int],
) -> list[DatasetSourceInfo]:
    """Upstream sources with their type, matched registry ref and sample count."""
    ref_lookup = _build_ref_lookup(refs)
    n_sources = max(len(raw_sources), 1)
    sources: list[DatasetSourceInfo] = []
    for s_idx, s_entry in enumerate(raw_sources):
        s_name = str(s_entry).strip()
        s_lower = s_name.lower()
        sources.append(
            DatasetSourceInfo(
                name=s_name,
                ref=_match_ref(s_lower, ref_lookup),
                count=_source_count(s_idx, s_name, sample_count, provenance_counts, n_sources),
                type=_source_type(s_lower),
            )
        )
    return sources


def _is_compiled(
    stats: DatasetFileStats, shards_count: int, is_mds: bool, has_jsonl: bool, meta: _ManifoldMeta
) -> bool:
    formats = stats.formats
    has_images = formats.webp > 0 or formats.jpg > 0 or formats.png > 0
    populated_directory = meta.format_val == "directory" and meta.sample_count > 0
    return (
        shards_count > 0
        or formats.parquet > 0
        or is_mds
        or has_jsonl
        or populated_directory
        or has_images
    )


def _build_detail(
    entry: Path, folder_index: dict[str, tuple[str, dict[str, Any]]]
) -> DatasetDetailStats:
    """Assemble the catalog record for one manifold folder."""
    reg_key, reg_info = folder_index.get(entry.name.lower(), (entry.name, {}))
    info_file = entry / "dataset_info.yaml"
    meta = _read_manifold_meta(info_file, reg_info)

    stats = calculate_dataset_file_stats(entry)
    jsonl_paths = jsonl_files(entry)
    if meta.sample_count == 0:
        if jsonl_paths:
            meta.sample_count = _count_jsonl_lines(jsonl_paths)
        else:
            f = stats.formats
            meta.sample_count = f.webp + f.jpg + f.png + f.parquet

    shards_count = stats.shards_count
    is_mds = (entry / "mds").exists() or "mds" in (meta.format_val, meta.canonical_format)
    if is_mds:
        meta.format_val = "mds"
        meta.canonical_format = "mds"
        if shards_count == 0 and (entry / "mds" / "index.json").exists():
            shards_count = 1
    is_compiled = _is_compiled(stats, shards_count, is_mds, bool(jsonl_paths), meta)

    sources = _build_sources(
        _read_original_sources(info_file) or reg_info.get("provenance_sources") or [],
        reg_info.get("refs") or [],
        meta.sample_count,
        extract_manifold_provenance_counts(entry),
    )
    return DatasetDetailStats(
        name=entry.name,
        key=reg_key,
        display_name=_display_name(entry.name, reg_info),
        path=str(entry),
        task=meta.task,
        sample_count=meta.sample_count,
        size_bytes=stats.size_bytes,
        size_gb=round(stats.size_bytes / _GB, 3),
        format=meta.format_val if is_compiled else "directory",
        canonical_format=meta.canonical_format,
        formats=stats.formats,
        shards_count=shards_count,
        is_compiled=is_compiled,
        has_hardlinks=stats.has_hardlinks,
        hardlink_ratio=stats.hardlink_ratio,
        sources=sources,
        kaggle_ref=reg_info.get("kaggle_ref"),
        recommended_preset=recommended_preset_for_dataset(meta.task, meta.canonical_format),
    )


def _build_uncompiled_detail(key: str, reg_info: dict[str, Any]) -> DatasetDetailStats:
    """Build a placeholder ``DatasetDetailStats`` for a registry entry not on disk."""
    canonical_format = str(reg_info.get("canonical_format") or "webdataset")
    container = reg_info.get("container")
    fmt = (
        str(container.get("primary"))
        if isinstance(container, dict) and container.get("primary")
        else canonical_format
    )
    task = str(reg_info.get("task") or "unknown")
    display_name = str(reg_info.get("title") or reg_info.get("name") or key)
    return DatasetDetailStats(
        key=key,
        name=str(reg_info.get("name") or key),
        display_name=display_name,
        path="",
        format=fmt,
        canonical_format=canonical_format,
        task=task,
        status="uncompiled",
        sample_count=0,
        size_bytes=0,
        size_gb=0.0,
        formats={},
        has_hardlinks=False,
        hardlink_ratio=0.0,
        recommended_preset=recommended_preset_for_dataset(task, canonical_format),
    )


def build_dataset_stats_response(
    root: Path,
    include_uncompiled: bool = False,
) -> DatasetStatsListResponse:
    """Enriched catalog of every manifold folder under ``root``."""
    folder_index = _load_folder_index()
    dataset_stats: list[DatasetDetailStats] = []
    if root.exists():
        dataset_stats = [
            _build_detail(entry, folder_index)
            for entry in sorted(root.iterdir(), key=lambda p: p.name.lower())
            if entry.is_dir() and not entry.name.startswith(".")
        ]
    if include_uncompiled:
        existing_keys = {d.key for d in dataset_stats}
        for key, reg_info in folder_index.values():
            if key and key not in existing_keys:
                dataset_stats.append(_build_uncompiled_detail(key, reg_info))
                existing_keys.add(key)
        dataset_stats.sort(key=lambda d: (d.display_name or d.name).lower())
    total_storage_bytes = sum(d.size_bytes for d in dataset_stats)
    return DatasetStatsListResponse(
        datasets=dataset_stats,
        total=len(dataset_stats),
        total_size_bytes=total_storage_bytes,
        total_size_gb=round(total_storage_bytes / _GB, 3),
    )
