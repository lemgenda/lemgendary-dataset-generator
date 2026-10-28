import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
import yaml

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
                folder = data.get("_registry_metadata", {}).get("output_folder_name", "../LemGendaryDatasets")
                p = Path(folder)
                if not p.is_absolute():
                    p = (config_file.parent / p).resolve()
                return p
        except Exception as exc:
            logger.debug("Could not parse output_folder_name from %s: %s", config_file, exc)
    return (Path(__file__).resolve().parent.parent.parent.parent / "LemGendaryDatasets").resolve()


@router.get("", response_model=DatasetListResponse)
async def list_datasets() -> DatasetListResponse:
    """List compiled dataset manifolds found in output storage root."""
    root = _resolve_output_root()
    datasets: List[DatasetInfoResponse] = []

    if not root.exists():
        return DatasetListResponse(datasets=[], total=0)

    for entry in root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        info_file = entry / "dataset_info.yaml"
        sample_count = 0
        task = "vision"
        formats: List[str] = ["directory"]

        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                    task = meta.get("task", task)
                    sample_count = int(meta.get("total_samples", 0) or 0)
            except Exception as exc:
                logger.debug("Failed reading %s: %s", info_file, exc)

        size_bytes = 0
        has_parquet = False
        has_jsonl = False

        try:
            for item in os.scandir(entry):
                if item.is_file():
                    fname = item.name.lower()
                    if fname.endswith(".parquet"):
                        has_parquet = True
                    elif fname.endswith(".jsonl"):
                        has_jsonl = True
                    try:
                        size_bytes += item.stat().st_size
                    except OSError as stat_exc:
                        logger.debug("Failed stat on %s: %s", item.path, stat_exc)
                elif item.is_dir():
                    dname = item.name
                    if dname == "mds":
                        formats.append("mds")
                    elif dname == "litdata":
                        formats.append("litdata")
                    elif dname in ("wds", "shards"):
                        if "wds" not in formats:
                            formats.append("wds")
        except OSError as exc:
            logger.debug("Failed scanning manifold directory %s: %s", entry, exc)

        if has_parquet:
            formats.append("parquet")
        if has_jsonl:
            formats.append("jsonl")

        if sample_count == 0:
            db_file = entry / "manifold_registry.db"
            try:
                if db_file.exists() and db_file.stat().st_size < 50_000_000:
                    with sqlite3.connect(db_file, timeout=0.5) as conn:
                        row = conn.execute("SELECT MAX(rowid) FROM registry").fetchone()
                        if row and row[0]:
                            sample_count = int(row[0])
            except Exception as exc:
                logger.debug("Failed reading registry DB %s: %s", db_file, exc)

        datasets.append(
            DatasetInfoResponse(
                name=entry.name,
                path=str(entry),
                task=task,
                sample_count=sample_count,
                size_bytes=size_bytes,
                formats=formats,
                has_hardlinks=False,
                hardlink_ratio=0.0,
            )
        )

    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get("/{name}", response_model=Dict[str, Any])
async def get_dataset_details(name: str) -> Dict[str, Any]:
    """Retrieve full metadata package for a specific compiled manifold."""
    root = _resolve_output_root()
    target = root / name
    if not target.exists() or not target.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Manifold '{name}' not found at {target}",
        )

    info_file = target / "dataset_info.yaml"
    details: Dict[str, Any] = {
        "name": name,
        "path": str(target),
        "exists": True,
    }

    if info_file.exists():
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                details["dataset_info"] = yaml.safe_load(f)
        except Exception as exc:
            details["dataset_info_error"] = str(exc)

    classes_file = target / "classes.txt"
    if classes_file.exists():
        try:
            details["classes"] = classes_file.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            logger.debug("Failed reading %s: %s", classes_file, exc)

    return details
