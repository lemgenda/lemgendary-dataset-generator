"""Per-folder content scan of a compiled manifold for the desktop GUI catalog.

Counts shards, container files and loose image formats (mtime-cached) and
attributes JSONL samples to their SSOT provenance source document family.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from api.models import DatasetFormatStats

__all__ = [
    "DatasetFileStats",
    "calculate_dataset_file_stats",
    "extract_jsonl_provenance_counts",
    "jsonl_files",
]

logger = logging.getLogger("lemgendary.api.routes.gui.stats")

_CONTAINER_DIRS = ("shards", "mds", "wds", "litdata", "data")
_SHARD_EXTS = ("data", "tar", "bin", "mds", "parquet", "zstd")
_ROOT_SHARD_EXTS = (*_SHARD_EXTS, "jsonl")
_LOOSE_DIRS = ("images", "targets", "masks", "labels")
_LOOSE_FORMATS = {"webp": "webp", "jpg": "jpg", "jpeg": "jpg", "png": "png", "parquet": "parquet"}
_LOOSE_SKIP_EXTS = ("txt", "json", "yaml", "md", "ipynb")

_MODEL_FAMILY_MARKERS = (
    "NIMA", "NAFNET", "MPRNET", "MIRNET", "FFANET", "YOLO", "FOREX", "DEGRADATION",
    "RESTORATION", "ULTRAZOOM", "FACE_SUITE", "HYBRID", "CODEFORMER", "FILM_RESTORER",
    "NSFW_CLASSIFIER", "PARSENET", "RETINAFACE", "UPN_V2", "DETECTION",
)
# First matching rule wins; documents matching none fall back to _DEFAULT_PROVENANCE.
_PROVENANCE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("ECOSYSTEM_ARCHITECTURE", "LEMGENDARY_CURRENT_STATE", "VERSIONING_POLICY",
         "GUI_CONTROL_REGISTRY"),
        "Ecosystem Architecture and Current State Manifest",
    ),
    (("GENERAL_AI_TRAINING_KNOWLEDGE",), "General AI Training Knowledge Reference"),
    (
        ("AI_HELPER_TROUBLESHOOTING", "GLOSSARY"),
        "Canonical Glossary and AI Helper Troubleshooting Knowledge Base",
    ),
    (
        ("MANUAL_AI_STUDIO_GUI", "PAPER_AI_STUDIO_GUI", "MANUAL_CLI", "MANUAL_API",
         "MANUALS_HUB"),
        "Master Operations Manuals (GUI, CLI, API)",
    ),
    (
        ("PAPER_DATASET_COMPILER", "PAPER_ENV_MANAGER"),
        "Dataset Compiler and Environment Manager Whitepapers",
    ),
    (
        ("PAPER_TRAINING_SUITE", "PAPER_TRAINING_PATHOLOGY"),
        "Training Suite and Training Pathology Documentation",
    ),
    (_MODEL_FAMILY_MARKERS, "Model-Family Architecture Whitepapers"),
    (("PAPER_FOUNDATION_MODELS_MASTER", "MANUALS_HUB"), "LemGendary AI Documentation Hub"),
)
_DEFAULT_PROVENANCE = "Curated AI LemGend instruction and validation examples"

class DatasetFileStats(NamedTuple):
    """Result of :func:`calculate_dataset_file_stats` (unpacks like the legacy 6-tuple)."""

    formats: DatasetFormatStats
    file_count: int
    size_bytes: int
    has_hardlinks: bool
    hardlink_ratio: float
    shards_count: int


@dataclass
class _ScanTally:
    """Mutable counters accumulated while scanning one manifold folder."""

    format_counts: dict[str, int] = field(
        default_factory=lambda: {"webp": 0, "jpg": 0, "png": 0, "parquet": 0, "other": 0}
    )
    total_size: int = 0
    file_count: int = 0
    shards_count: int = 0


_STATS_CACHE: dict[str, tuple[float, DatasetFileStats]] = {}

def _ext(fname: str) -> str:
    return fname.rsplit(".", 1)[-1].lower() if "." in fname else ""


def _is_container_shard(fname: str, ext: str, container_name: str) -> bool:
    if fname == "index.json":
        return False
    return (
        ext in _SHARD_EXTS
        or fname.startswith(("chunk", "shard"))
        or (container_name == "mds" and not ext)
    )


def _container_scan_dirs(c_dir: Path) -> list[Path]:
    """The container directory itself plus its immediate split sub-folders."""
    dirs_to_check = [c_dir]
    try:
        for sub in os.scandir(c_dir):
            if sub.is_dir():
                dirs_to_check.append(Path(sub.path))
    except OSError as exc:
        logger.debug("Could not scan subdirectories of %s: %s", c_dir, exc)
    return dirs_to_check


def _tally_container_file(tally: _ScanTally, entry: os.DirEntry, container_name: str) -> None:
    fname = entry.name
    ext = _ext(fname)
    if _is_container_shard(fname, ext, container_name):
        tally.shards_count += 1
    if ext == "parquet":
        tally.format_counts["parquet"] += 1
    tally.file_count += 1
    try:
        tally.total_size += entry.stat().st_size
    except OSError as stat_exc:
        logger.debug("Could not stat container file %s: %s", entry.path, stat_exc)


def _scan_containers(tally: _ScanTally, manifold_path: Path) -> None:
    """Inspect container directories (shards, mds, wds, litdata) and their split folders."""
    for container_name in _CONTAINER_DIRS:
        c_dir = manifold_path / container_name
        if not (c_dir.exists() and c_dir.is_dir()):
            continue
        for check_d in _container_scan_dirs(c_dir):
            try:
                for entry in os.scandir(check_d):
                    if entry.is_file():
                        _tally_container_file(tally, entry, container_name)
            except OSError as exc:
                logger.debug("Failed scanning container folder %s: %s", check_d, exc)


def _tally_root_file(tally: _ScanTally, entry: os.DirEntry) -> None:
    fname = entry.name
    ext = _ext(fname)
    if fname != "index.json" and (
        ext in _ROOT_SHARD_EXTS or fname.startswith(("shard", "chunk"))
    ):
        tally.shards_count += 1
    if ext == "parquet":
        tally.format_counts["parquet"] += 1
    try:
        tally.total_size += entry.stat().st_size
    except OSError as stat_exc:
        logger.debug("Could not stat root file %s: %s", entry.path, stat_exc)


def _scan_root_files(tally: _ScanTally, manifold_path: Path) -> None:
    """Check root-level files in the manifold (e.g. .parquet, dataset_info.yaml)."""
    try:
        for entry in os.scandir(manifold_path):
            if entry.is_file():
                _tally_root_file(tally, entry)
    except OSError as exc:
        logger.debug("Failed scanning root manifold path %s: %s", manifold_path, exc)


def _count_image_splits(tally: _ScanTally, manifold_path: Path) -> None:
    """Count split directories of directory-based manifolds (e.g. YOLO images/train)."""
    images_dir = manifold_path / "images"
    if not (images_dir.exists() and images_dir.is_dir()):
        return
    try:
        split_dirs = [s for s in os.scandir(images_dir) if s.is_dir()]
        if split_dirs:
            tally.shards_count = len(split_dirs)
    except OSError as exc:
        logger.debug("Failed scanning images directory %s: %s", images_dir, exc)


def _tally_loose_file(tally: _ScanTally, root_d: str, fname: str) -> None:
    ext = _ext(fname)
    if ext in _LOOSE_SKIP_EXTS:
        return
    tally.format_counts[_LOOSE_FORMATS.get(ext, "other")] += 1
    tally.file_count += 1
    try:
        tally.total_size += (Path(root_d) / fname).stat().st_size
    except OSError as stat_exc:
        logger.debug("Could not stat file %s in %s: %s", fname, root_d, stat_exc)


def _scan_loose_dirs(tally: _ScanTally, manifold_path: Path) -> None:
    """Scan loose image directories (images, targets, masks, labels)."""
    for directory in (manifold_path / name for name in _LOOSE_DIRS):
        if not directory.exists():
            continue
        try:
            for root_d, _, files in os.walk(directory):
                for fname in files:
                    _tally_loose_file(tally, root_d, fname)
        except OSError as scan_exc:
            logger.debug("Failed scanning %s: %s", directory, scan_exc)


def calculate_dataset_file_stats(manifold_path: Path) -> DatasetFileStats:
    """Fast scan of manifold directory with container prioritization and mtime caching."""
    try:
        current_mtime = manifold_path.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    cache_key = str(manifold_path.resolve())
    if cache_key in _STATS_CACHE:
        cached_mtime, cached_val = _STATS_CACHE[cache_key]
        if cached_mtime == current_mtime:
            return cached_val

    tally = _ScanTally()
    _scan_containers(tally, manifold_path)
    _scan_root_files(tally, manifold_path)
    if tally.shards_count == 0:
        _count_image_splits(tally, manifold_path)
    # Loose files are only scanned when no shards were found or they hold few files
    if tally.shards_count == 0 or tally.file_count < 10:
        _scan_loose_dirs(tally, manifold_path)

    result = DatasetFileStats(
        formats=DatasetFormatStats(**tally.format_counts),
        file_count=tally.file_count,
        size_bytes=tally.total_size,
        has_hardlinks=False,
        hardlink_ratio=0.0,
        shards_count=tally.shards_count,
    )
    _STATS_CACHE[cache_key] = (current_mtime, result)
    return result


def jsonl_files(manifold_dir: Path) -> list[Path]:
    """JSONL shards at the manifold root and in its data/ folder."""
    return list(manifold_dir.glob("*.jsonl")) + list(manifold_dir.glob("data/*.jsonl"))


def _provenance_label(doc: str) -> str:
    for markers, label in _PROVENANCE_RULES:
        if any(m in doc for m in markers):
            return label
    return _DEFAULT_PROVENANCE


def extract_jsonl_provenance_counts(manifold_dir: Path) -> dict[str, int]:
    """Parse JSONL datasets and return exact sample count per SSOT provenance source."""
    counts: Counter[str] = Counter()
    for jf in jsonl_files(manifold_dir):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    counts[_provenance_label(str(item.get("source_doc") or ""))] += 1
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
            logger.debug("Failed extracting provenance from %s: %s", jf, exc)
    return dict(counts)
