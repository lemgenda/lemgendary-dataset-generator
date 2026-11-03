"""Routes that list compiled datasets and report their details."""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, status

from api.dataset_stats import as_count, read_yaml_mapping
from api.models import DatasetInfoResponse, DatasetListResponse

logger = logging.getLogger("lemgendary.api.routes.datasets")
router = APIRouter(prefix="/datasets", tags=["Datasets"])


def _resolve_output_root() -> Path:
    config_file = Path("unified_data.yaml")
    if not config_file.exists():
        config_file = Path(__file__).resolve().parent.parent.parent / "unified_data.yaml"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                folder = data.get("_registry_metadata", {}).get(
                    "output_folder_name", "../LemGendaryDatasets"
                )
                p = Path(folder)
                if not p.is_absolute():
                    p = (config_file.parent / p).resolve()
                return p
        except (OSError, UnicodeDecodeError, yaml.YAMLError, AttributeError) as exc:
            logger.debug("Could not parse output_folder_name from %s: %s", config_file, exc)
    return (Path(__file__).resolve().parent.parent.parent.parent / "LemGendaryDatasets").resolve()


_DIR_FORMATS = {"mds": "mds", "litdata": "litdata", "wds": "wds", "shards": "wds"}
_REGISTRY_DB_MAX_BYTES = 50_000_000


def _read_list_meta(info_file: Path) -> tuple[str, int]:
    """Task and total_samples from a manifold's dataset_info.yaml (defaults if unreadable)."""
    meta = read_yaml_mapping(info_file)
    return str(meta.get("task", "vision")), as_count(meta.get("total_samples"))


def _scan_manifold_formats(entry: Path) -> tuple[int, list[str]]:
    """Root-level byte size and the container formats present in a manifold folder."""
    formats: list[str] = ["directory"]
    size_bytes = 0
    has_parquet = False
    has_jsonl = False
    try:
        for item in os.scandir(entry):
            if item.is_file():
                fname = item.name.lower()
                has_parquet = has_parquet or fname.endswith(".parquet")
                has_jsonl = has_jsonl or fname.endswith(".jsonl")
                try:
                    size_bytes += item.stat().st_size
                except OSError as stat_exc:
                    logger.debug("Failed stat on %s: %s", item.path, stat_exc)
            elif item.is_dir():
                dir_format = _DIR_FORMATS.get(item.name)
                if dir_format and dir_format not in formats:
                    formats.append(dir_format)
    except OSError as exc:
        logger.debug("Failed scanning manifold directory %s: %s", entry, exc)

    if has_parquet:
        formats.append("parquet")
    if has_jsonl:
        formats.append("jsonl")
    return size_bytes, formats


def _registry_db_sample_count(entry: Path) -> int:
    """Highest registry rowid of a (reasonably small) manifold_registry.db, else 0."""
    db_file = entry / "manifold_registry.db"
    try:
        if db_file.exists() and db_file.stat().st_size < _REGISTRY_DB_MAX_BYTES:
            with sqlite3.connect(db_file, timeout=0.5) as conn:
                row = conn.execute("SELECT MAX(rowid) FROM registry").fetchone()
                if row and row[0]:
                    return int(row[0])
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        logger.debug("Failed reading registry DB %s: %s", db_file, exc)
    return 0


def _dataset_info(entry: Path) -> DatasetInfoResponse:
    task, sample_count = _read_list_meta(entry / "dataset_info.yaml")
    size_bytes, formats = _scan_manifold_formats(entry)
    if sample_count == 0:
        sample_count = _registry_db_sample_count(entry)
    return DatasetInfoResponse(
        name=entry.name,
        path=str(entry),
        task=task,
        sample_count=sample_count,
        size_bytes=size_bytes,
        formats=formats,
        has_hardlinks=False,
        hardlink_ratio=0.0,
    )


@router.get("", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    """List compiled dataset manifolds found in output storage root."""
    root = _resolve_output_root()
    if not root.exists():
        return DatasetListResponse(datasets=[], total=0)

    datasets: list[DatasetInfoResponse] = [
        _dataset_info(entry)
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get("/{name}", response_model=dict[str, Any])
def get_dataset_details(name: str) -> dict[str, Any]:
    """Retrieve full metadata package for a specific compiled manifold."""
    root = _resolve_output_root()
    target = root / name
    if not target.exists() or not target.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Manifold '{name}' not found at {target}",
        )

    info_file = target / "dataset_info.yaml"
    details: dict[str, Any] = {
        "name": name,
        "path": str(target),
        "exists": True,
    }

    if info_file.exists():
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                details["dataset_info"] = yaml.safe_load(f)
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            logger.warning("Failed reading %s: %s", info_file, exc)
            details["dataset_info_error"] = str(exc)

    classes_file = target / "classes.txt"
    if classes_file.exists():
        try:
            details["classes"] = classes_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug("Failed reading %s: %s", classes_file, exc)

    return details
