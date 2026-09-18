"""Kaggle dataset integration and sync endpoints."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.jobs import job_manager
from api.models import JobResponse, JobType
from core.cli_args import venv_python

router = APIRouter(prefix="/kaggle", tags=["Kaggle"])


class KaggleSyncRequest(BaseModel):
    manifold: Optional[str] = Field(None, description="Specific manifold ID or folder name")
    push: bool = Field(False, description="Push local manifold to Kaggle")
    pull: bool = Field(False, description="Pull remote dataset from Kaggle")
    force: bool = Field(False, description="Force overwrite")


@router.get("/status")
async def get_kaggle_status() -> Dict[str, Any]:
    """Inspect Kaggle authentication and local credential files."""
    has_env = bool(os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))
    token_file = Path(".kaggle_token")
    has_token_file = token_file.exists() and len(token_file.read_text(encoding="utf-8").strip()) > 0
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_kaggle_json = kaggle_json.exists()

    authenticated = has_env or has_token_file or has_kaggle_json

    return {
        "authenticated": authenticated,
        "auth_methods": {
            "environment_variables": has_env,
            "dot_kaggle_token": has_token_file,
            "user_kaggle_json": has_kaggle_json,
        },
    }


@router.post("/sync", response_model=JobResponse)
async def trigger_kaggle_sync(req: KaggleSyncRequest) -> JobResponse:
    sync_script = "tools/manifold_sync.py" if (Path(__file__).resolve().parent.parent.parent / "tools" / "manifold_sync.py").exists() else "manifold_sync.py"
    cmd = [venv_python(), sync_script]
    if req.manifold:
        cmd.extend(["--manifold", req.manifold])
    if req.push:
        cmd.append("--push")
    if req.pull:
        cmd.append("--pull")
    if req.force:
        cmd.append("--force")

    job = job_manager.create_job(
        job_type=JobType.SYNC,
        command=cmd,
        parameters=req.model_dump(),
    )
    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    return job
