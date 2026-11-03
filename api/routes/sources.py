"""Routes that list local raw datasets and their upstream references."""

import logging
from typing import Any

import yaml
from fastapi import APIRouter

from core import paths

logger = logging.getLogger("lemgendary.api.routes.sources")
router = APIRouter(prefix="/sources", tags=["Sources"])
_RAW_SETS = paths.RAW_SETS


@router.get("", response_model=dict[str, Any])
def list_sources() -> dict[str, Any]:
    """List local raw datasets and configured upstream references."""
    local_sources: list[dict[str, Any]] = []
    if _RAW_SETS.exists():
        for entry in _RAW_SETS.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                local_sources.append({
                    "name": entry.name,
                    "path": str(entry),
                    "type": "directory",
                })

    configured_datasets: dict[str, Any] = {}
    config_file = paths.UNIFIED_DATA
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                configured_datasets = data.get("datasets", {})
        except (OSError, UnicodeDecodeError, yaml.YAMLError, AttributeError) as exc:
            logger.debug("Failed reading %s in sources listing: %s", config_file, exc)

    return {
        "raw_sets_path": str(_RAW_SETS.resolve()),
        "local_sources": local_sources,
        "total_local": len(local_sources),
        "total_configured": len(configured_datasets),
    }
