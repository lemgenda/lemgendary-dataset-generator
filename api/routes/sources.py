import logging
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter
import yaml

logger = logging.getLogger("lemgendary.api.routes.sources")
router = APIRouter(prefix="/sources", tags=["Sources"])
_RAW_SETS = Path("raw-sets")


@router.get("", response_model=Dict[str, Any])
async def list_sources() -> Dict[str, Any]:
    """List local raw datasets and configured upstream references."""
    local_sources: List[Dict[str, Any]] = []
    if _RAW_SETS.exists():
        for entry in _RAW_SETS.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                local_sources.append({
                    "name": entry.name,
                    "path": str(entry),
                    "type": "directory",
                })

    configured_datasets: Dict[str, Any] = {}
    config_file = Path("unified_data.yaml")
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                configured_datasets = data.get("datasets", {})
        except Exception as exc:
            logger.debug("Failed reading %s in sources listing: %s", config_file, exc)

    return {
        "raw_sets_path": str(_RAW_SETS.resolve()),
        "local_sources": local_sources,
        "total_local": len(local_sources),
        "total_configured": len(configured_datasets),
    }
