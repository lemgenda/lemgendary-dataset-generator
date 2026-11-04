"""Registry I/O and discovery of legacy manifolds eligible for modernization."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from tools.modernize.common import SPLITS
from utils.container_stream import dump_yaml_file

logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

REGISTRY_YAML = ROOT / "unified_data.yaml"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

LEGACY_SUFFIX = "Large"
BRAND_PREFIX = "LemGendized"

_DATA_FILE_SUFFIXES = (".parquet", ".tar", ".data", ".mds", ".bin", ".jsonl")
_CONTAINER_DIRS = ("shards", "mds", "litdata")
_LOOSE_DIRS = ("images", "targets", "masks")


def _out_parent() -> Path:
    try:
        with open(REGISTRY_YAML, encoding="utf-8") as f:
            reg = yaml.safe_load(f) or {}
        folder = reg.get("_registry_metadata", {}).get(
            "output_folder_name", "../LemGendaryDatasets"
        )
        return (ROOT / folder).resolve()
    except (OSError, yaml.YAMLError, AttributeError, TypeError) as exc:
        logger.debug("Failed reading registry for output_folder_name, using fallback: %s", exc)
        return (ROOT.parent / "LemGendaryDatasets").resolve()


# ─── Data-Presence Check ────────────────────────────────────────────────────
def _container_dir_has_entries(sub: str) -> bool:
    try:
        with os.scandir(sub) as sit:
            return any(True for _ in sit)
    except OSError as exc:
        logger.debug("Could not scan container directory %s: %s", sub, exc)
        return False


def _split_dir_has_files(sp: str) -> bool:
    try:
        with os.scandir(sp) as sit:
            return any(sf.is_file() for sf in sit)
    except OSError as exc:
        logger.debug("Could not scan split directory %s: %s", sp, exc)
        return False


def _loose_dir_has_files(sub: str) -> bool:
    for split in SPLITS:
        sp = os.path.join(sub, split)
        if os.path.isdir(sp) and _split_dir_has_files(sp):
            return True
    return False


def _subdir_has_data(sub: str) -> bool:
    bname = os.path.basename(sub)
    if bname in _CONTAINER_DIRS:
        return _container_dir_has_entries(sub)
    if bname in _LOOSE_DIRS:
        return _loose_dir_has_files(sub)
    return False


def _has_manifold_data(path: Path) -> bool:
    """Return True if the manifold folder has actual data (not just empty dirs)."""
    try:
        subdirs: set[str] = set()
        with os.scandir(path) as it:
            for entry in it:
                if entry.name.endswith(_DATA_FILE_SUFFIXES):
                    return True
                if entry.is_dir():
                    subdirs.add(entry.path)
        return any(_subdir_has_data(sub) for sub in subdirs)
    except OSError as exc:
        logger.debug("Error checking manifold data in %s: %s", path, exc)
    return False


def _is_manifold_fully_modernized(path: Path) -> bool:
    """Return True only if the target modern manifold is fully generated with docs/index."""
    if not path.exists() or not _has_manifold_data(path):
        return False
    return (path / "dataset_info.yaml").exists() and (path / "index.json").exists()


# ─── Registry I/O ───────────────────────────────────────────────────────────
def _load_registry() -> dict:
    with open(REGISTRY_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_registry(data: dict) -> None:
    dump_yaml_file(REGISTRY_YAML, data)


def _find_dataset_entry(
    registry: dict, base_name: str, folder_name: str
) -> tuple[str | None, dict]:
    """Find dataset key and entry from registry."""
    datasets = registry.get("datasets", {})
    for key, entry in datasets.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("name", "") == base_name:
            return key, entry
        mod_folder = entry.get("modernized_folder", "")
        if mod_folder and folder_name in (mod_folder, f"{mod_folder}Large"):
            return key, entry
    return None, {}


def _find_kaggle_key(registry: dict, base_name: str) -> str | None:
    """Find the dataset key whose `name` matches the base_name (no prefix/suffix)."""
    key, _ = _find_dataset_entry(registry, base_name, "")
    return key


# ─── Eligibility Scan ───────────────────────────────────────────────────────
def _is_legacy_candidate(entry: Path) -> bool:
    return (
        entry.is_dir()
        and entry.name.endswith(LEGACY_SUFFIX)
        and entry.name.startswith(BRAND_PREFIX)
        and _has_manifold_data(entry)
    )


def _eligible_item(registry: dict, entry: Path, target_path: Path) -> dict:
    base = entry.name[len(BRAND_PREFIX): -len(LEGACY_SUFFIX)]
    kaggle_key, dataset_entry = _find_dataset_entry(registry, base, entry.name)
    return {
        "folder_name": entry.name,
        "current_path": entry,
        "target_name": target_path.name,
        "target_path": target_path,
        "kaggle_key": kaggle_key,
        "old_kaggle_ref": dataset_entry.get("kaggle_ref", ""),
        "base_name": base,
        "is_resume": target_path.exists() and _has_manifold_data(target_path),
        "canonical_format": dataset_entry.get("canonical_format", "directory"),
    }


def _enumerate_eligible(registry: dict) -> list[dict]:
    out = _out_parent()
    if not out.exists():
        return []

    eligible: list[dict] = []
    for entry in sorted(out.iterdir()):
        if not _is_legacy_candidate(entry):
            continue
        target_path = out / entry.name[: -len(LEGACY_SUFFIX)]
        # If target already exists and is fully modernized, skip it
        if _is_manifold_fully_modernized(target_path):
            continue
        eligible.append(_eligible_item(registry, entry, target_path))
    return eligible
