"""Configuration inspection and validation endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
import yaml

from core.config_schema import load_unified_data

router = APIRouter(prefix="/config", tags=["Configuration"])


def _resolve_config_path() -> Path:
    candidates = [
        Path("unified_data.yaml"),
        Path(__file__).resolve().parent.parent.parent / "unified_data.yaml",
        Path.cwd() / "lemgendary-datasets" / "unified_data.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


@router.get("", response_model=Dict[str, Any])
async def get_config() -> Dict[str, Any]:
    """Retrieve the parsed contents of unified_data.yaml."""
    config_path = _resolve_config_path()
    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration file {config_path} not found",
        )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed reading configuration: {exc}",
        ) from exc


@router.post("/validate")
async def validate_config() -> Dict[str, Any]:
    """Validate unified_data.yaml against Pydantic config schema."""
    config_path = _resolve_config_path()
    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration file {config_path} not found",
        )
    try:
        data = load_unified_data(config_path)
        rm = data.registry_metadata
        return {
            "status": "valid",
            "version": rm.version,
            "prefix": rm.name_prefix,
            "suffix": rm.name_suffix,
            "output_folder": rm.output_folder_name,
            "dataset_count": len(data.datasets),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Configuration schema validation failed: {exc}",
        ) from exc
