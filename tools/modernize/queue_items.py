"""Modernization queue items: registry + filesystem scan and status labels."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tools.modernize.common import LOOSE_SUBDIRS

logger = logging.getLogger(__name__)

_GB = 1024**3


@dataclass
class ManifoldRegistryEntry:
    """Registry (unified_data.yaml) identity and publication fields of a manifold."""

    key: str
    name: str
    canonical_format: str
    modernized_folder: str
    kaggle_ref: str
    title: str
    subtitle: str


@dataclass
class ManifoldSources(ManifoldRegistryEntry):
    """Registry licensing fields plus the on-disk folders and archive of a manifold."""

    license_id: str
    keywords: list[str] = field(default_factory=list)
    provenance_sources: list[str] = field(default_factory=list)

    # Filesystem locations
    target_dir: Path | None = None
    legacy_dir: Path | None = None
    zip_path: Path | None = None


@dataclass
class ManifoldQueueItem(ManifoldSources):
    """Represents a dataset manifold in the modernization queue."""

    # Filesystem state
    is_containerized: bool = False
    has_loose_files: bool = False
    container_bytes: int = 0
    file_count: int = 0
    remote_version: int = 0
    has_mds: bool = False
    mds_bytes: int = 0


def _is_tar_shard(name: str) -> bool:
    return name.endswith((".data", ".tar"))


def _fast_has_tar_shards(shards_dir: Path) -> tuple[bool, int]:
    """Quickly check if shards dir has .data or .tar files and approximate size."""
    if not shards_dir.exists():
        return False, 0
    total_bytes = 0
    found = False
    try:
        # Check root of shards or immediate subdirectories (train, val, test)
        for entry in os.scandir(shards_dir):
            if entry.is_file() and _is_tar_shard(entry.name):
                found = True
                total_bytes += entry.stat().st_size
            elif entry.is_dir():
                for sub_entry in os.scandir(entry.path):
                    if sub_entry.is_file() and _is_tar_shard(sub_entry.name):
                        found = True
                        total_bytes += sub_entry.stat().st_size
    except OSError as exc:
        logger.warning(
            "Could not scan shard directory %s (partial size %d bytes): %s",
            shards_dir,
            total_bytes,
            exc,
        )
    return found, total_bytes


def _is_mds_file(name: str) -> bool:
    return name == "index.json" or name.endswith((".mds", ".zstd")) or ".mds." in name


def _fast_has_mds(mds_dir: Path) -> tuple[bool, int]:
    """Quickly check if mds dir has .mds files or index.json and approximate size."""
    if not mds_dir.exists() or not mds_dir.is_dir():
        return False, 0
    total_bytes = 0
    found = False
    try:
        for root, _, files in os.walk(mds_dir):
            for f in files:
                if _is_mds_file(f):
                    found = True
                    total_bytes += os.path.getsize(os.path.join(root, f))
    except OSError as exc:
        logger.warning(
            "Could not scan MDS directory %s (partial size %d bytes): %s", mds_dir, total_bytes, exc
        )
    return found, total_bytes


def _find_zip(datasets_root: Path, folder_name: str, name: str) -> Path | None:
    """Return the first non-empty legacy zip archive of a manifold, if any."""
    candidates = [
        datasets_root / f"{folder_name.lower()}large.zip",
        datasets_root / f"{folder_name.lower()}.zip",
        datasets_root / f"{name.lower()}large.zip",
        datasets_root / f"{name.lower()}.zip",
    ]
    for z in candidates:
        if z.exists() and z.stat().st_size > 0:
            return z
    return None


def _directory_container_bytes(target_dir: Path) -> int:
    """Size of a directory-format manifold: recorded size, else its top-level files."""
    container_bytes = 0
    info_yaml = target_dir / "dataset_info.yaml"
    if info_yaml.exists():
        try:
            with open(info_yaml, "r", encoding="utf-8") as f_info:
                i_data = yaml.safe_load(f_info) or {}
            container_bytes = i_data.get("approx_bytes") or i_data.get("container_bytes") or 0
        except (OSError, yaml.YAMLError, AttributeError) as exc:
            logger.warning("Could not read container size from %s: %s", info_yaml, exc)
            container_bytes = 0
    if container_bytes == 0:
        try:
            container_bytes = sum(e.stat().st_size for e in os.scandir(target_dir) if e.is_file())
        except OSError as exc:
            logger.warning("Could not compute container size for %s: %s", target_dir, exc)
    return container_bytes


def _webdataset_state(target_dir: Path) -> tuple[bool, int]:
    return _fast_has_tar_shards(target_dir / "shards")


def _parquet_state(target_dir: Path) -> tuple[bool, int]:
    if not any(target_dir.glob("*.parquet")):
        return False, 0
    return True, sum(f.stat().st_size for f in target_dir.glob("*.parquet"))


def _litdata_state(target_dir: Path) -> tuple[bool, int]:
    return any(target_dir.glob("chunk-*.bin")), 0


def _directory_state(target_dir: Path) -> tuple[bool, int]:
    return True, _directory_container_bytes(target_dir)


# canonical_format -> (target_dir -> (is_containerized, container_bytes)); MDS is
# handled by the caller because its scan result is also stored on the item.
_CONTAINER_PROBES: dict[str, Callable[[Path], tuple[bool, int]]] = {
    "webdataset": _webdataset_state,
    "parquet": _parquet_state,
    "litdata": _litdata_state,
    "directory": _directory_state,
}


def _has_loose_dirs(target_dir: Path) -> bool:
    return any((target_dir / sub).exists() for sub in LOOSE_SUBDIRS)


def _scan_target(target_dir: Path, can_fmt: str) -> dict[str, Any]:
    """Return the filesystem-state fields of a manifold's modernized folder."""
    if not target_dir.exists():
        return {}
    has_mds, mds_bytes = _fast_has_mds(target_dir / "mds")
    if can_fmt == "mds":
        is_containerized, container_bytes = has_mds, (mds_bytes if has_mds else 0)
    else:
        probe = _CONTAINER_PROBES.get(can_fmt)
        is_containerized, container_bytes = probe(target_dir) if probe else (False, 0)
    return {
        "is_containerized": is_containerized,
        "container_bytes": container_bytes if is_containerized else 0,
        "has_loose_files": _has_loose_dirs(target_dir),
        "has_mds": has_mds,
        "mds_bytes": mds_bytes,
    }


def _queue_item(datasets_root: Path, key: str, ds: dict[str, Any]) -> ManifoldQueueItem:
    """Build the queue item of one registry entry."""
    name = ds.get("name", key)
    mod_folder_name = ds.get("modernized_folder", f"LemGendized{name}")
    can_fmt = ds.get("canonical_format", "webdataset")
    target_dir = datasets_root / mod_folder_name
    legacy_dir = datasets_root / f"{mod_folder_name}Large"
    return ManifoldQueueItem(
        key=key,
        name=name,
        canonical_format=can_fmt,
        modernized_folder=mod_folder_name,
        kaggle_ref=ds.get("kaggle_ref", ""),
        title=ds.get("title", f"LemGendized {name}"),
        subtitle=ds.get("subtitle", ""),
        license_id=ds.get("license", "CC-BY-NC-4.0"),
        keywords=ds.get("keywords", []),
        provenance_sources=ds.get("provenance_sources", []),
        target_dir=target_dir,
        legacy_dir=legacy_dir if legacy_dir.exists() else None,
        zip_path=_find_zip(datasets_root, mod_folder_name, name),
        **_scan_target(target_dir, can_fmt),
    )


def load_queue_manifest(
    datasets_root: Path,
    registry_yaml: Path,
) -> list[ManifoldQueueItem]:
    """Scan unified_data.yaml and LemGendaryDatasets to build the queue state."""
    with open(registry_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return [_queue_item(datasets_root, key, ds) for key, ds in cfg.get("datasets", {}).items()]


@dataclass(frozen=True)
class StatusLabelStyle:
    """How :func:`queue_status_labels` renders sizes and which states it reports."""

    gb_unit: str = "GB"
    gb_ndigits: int | None = None
    include_mds: bool = True

    def gb(self, nbytes: int) -> str:
        """Format ``nbytes`` as gigabytes with one decimal and the unit."""
        size = nbytes / _GB
        if self.gb_ndigits is not None:
            size = round(size, self.gb_ndigits)
        return f"{size:.1f}{self.gb_unit}"


# Labels printed by the queue status table.
TABLE_LABEL_STYLE = StatusLabelStyle()
# Labels returned by GET /api/modernize/queue.
API_LABEL_STYLE = StatusLabelStyle(gb_unit=" GB", gb_ndigits=2, include_mds=False)


def queue_status_labels(
    it: ManifoldQueueItem, style: StatusLabelStyle = TABLE_LABEL_STYLE
) -> tuple[str, str]:
    """Return the (status, source) labels describing the state of ``it``."""
    if it.is_containerized:
        return f"CONTAINERIZED ({style.gb(it.container_bytes)})", "Shards ready"
    if it.zip_path:
        return "PENDING STREAM", f"ZIP ({style.gb(it.zip_path.stat().st_size)})"
    if it.has_loose_files:
        return "PENDING PACK", "Loose WebP/Images"
    if it.has_mds and style.include_mds:
        return "MDS MIGRATION", f"MDS ({style.gb(it.mds_bytes)})"
    if it.legacy_dir:
        return "PENDING LEGACY", f"{it.legacy_dir.name}/"
    return "MISSING SOURCE", "None"
