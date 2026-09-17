"""Configuration inspection and validation endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
import yaml

from config_schema import load_unified_data

router = APIRouter(prefix="/config", tags=["Configuration"])
_CONFIG_PATH = Path("unified_data.yaml")


@router.get("", response_model=Dict[str, Any])
async def get_config() -> Dict[str, Any]:
    """Retrieve the parsed contents of unified_data.yaml."""
    if not _CONFIG_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration file {_CONFIG_PATH} not found",
        )
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
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
    if not _CONFIG_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration file {_CONFIG_PATH} not found",
        )
    try:
        data = load_unified_data(_CONFIG_PATH)
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
