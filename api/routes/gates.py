"""Quality and hardlink validation gate endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query, status

from audit.hardlinks import audit_hardlinks

router = APIRouter(prefix="/gates", tags=["Gates"])


@router.get("/hardlinks")
async def evaluate_hardlinks_gate(path: str = Query(..., description="Manifold directory path to audit")) -> Dict[str, Any]:
    """Audit the hardlink fraction of a manifold and return the container write verdict."""
    target = Path(path)
    if not target.exists() or not target.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory '{path}' does not exist",
        )

    res = audit_hardlinks(target)
    return {
        "path": str(target),
        "total_files": res.total_files,
        "hardlinked_files": res.hardlinked_files,
        "hardlink_pct": round(res.hardlink_pct, 2),
        "total_bytes": res.total_bytes,
        "hardlinked_bytes": res.hardlinked_bytes,
        "verdict": res.verdict,
        "summary": res.summary(),
    }
