"""Modernization queue, batch conversion, and manifold downsampling endpoints."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import yaml

from api.jobs import job_manager
from api.models import JobResponse, JobType
from core.cli_args import venv_python
from tools.modernization_queue import load_queue_manifest

logger = logging.getLogger("lemgendary.api.routes.modernize")
router = APIRouter(prefix="/modernize", tags=["Modernization Hub"])


class QueueItemResponse(BaseModel):
    key: str
    name: str
    canonical_format: str
    modernized_folder: str
    kaggle_ref: str
    title: str
    subtitle: str
    is_containerized: bool
    has_loose_files: bool
    container_bytes: int
    container_gb: float
    status_label: str
    source_label: str


class ModernizeRunRequest(BaseModel):
    datasets: Optional[List[str]] = Field(None, description="Specific dataset key(s) to modernize. Omit for full batch.")
    workers: Optional[int] = Field(12, description="Transcoding worker threads")
    mds_shard_mb: Optional[int] = Field(512, description="MDS shard size in MB")
    purge_loose: bool = Field(True, description="Purge loose extracted images post-conversion")
    push_kaggle: bool = Field(False, description="Push modernized package to Kaggle automatically")


class ManifoldReduceRequest(BaseModel):
    dataset_key: Optional[str] = Field(None, description="Target dataset key to reduce")
    scale: Optional[int] = Field(2, description="Downsampling scale factor (e.g. 2, 4)")
    workers: Optional[int] = Field(8, description="Worker threads count")


@router.get("/queue", response_model=List[QueueItemResponse])
def get_modernization_queue() -> List[QueueItemResponse]:
    """Scan and retrieve real-time modernization status queue across all manifolds."""
    root = Path("../LemGendaryDatasets").resolve()
    unified_yaml = Path("unified_data.yaml").resolve()
    if not unified_yaml.exists():
        unified_yaml = Path(__file__).resolve().parent.parent.parent / "unified_data.yaml"

    items = load_queue_manifest(root, unified_yaml) if unified_yaml.exists() else []
    results: List[QueueItemResponse] = []

    for it in items:
        size_gb = round(it.container_bytes / (1024**3), 2)
        if it.is_containerized:
            status_str = f"CONTAINERIZED ({size_gb:.1f} GB)"
            src_str = "Shards ready"
        elif it.zip_path:
            zip_gb = round(it.zip_path.stat().st_size / (1024**3), 2)
            status_str = "PENDING STREAM"
            src_str = f"ZIP ({zip_gb:.1f} GB)"
        elif it.has_loose_files:
            status_str = "PENDING PACK"
            src_str = "Loose WebP/Images"
        elif it.legacy_dir:
            status_str = "PENDING LEGACY"
            src_str = f"{it.legacy_dir.name}/"
        else:
            status_str = "MISSING SOURCE"
            src_str = "None"

        results.append(
            QueueItemResponse(
                key=it.key,
                name=it.name,
                canonical_format=it.canonical_format,
                modernized_folder=it.modernized_folder,
                kaggle_ref=it.kaggle_ref,
                title=it.title,
                subtitle=it.subtitle,
                is_containerized=it.is_containerized,
                has_loose_files=it.has_loose_files,
                container_bytes=it.container_bytes,
                container_gb=size_gb,
                status_label=status_str,
                source_label=src_str,
            )
        )

    return results


@router.post("/run", response_model=JobResponse)
async def run_modernization(req: ModernizeRunRequest) -> JobResponse:
    """Launch asynchronous modernization queue execution."""
    cmd = [
        venv_python(),
        "tools/modernization_queue.py",
        "--workers", str(req.workers or 12),
        "--mds-shard-mb", str(req.mds_shard_mb or 512),
    ]
    if req.purge_loose:
        cmd.append("--purge-loose")
    else:
        cmd.append("--keep-zip")

    if req.push_kaggle:
        cmd.append("--push-kaggle")

    if req.datasets:
        for ds in req.datasets:
            cmd.extend(["--datasets", ds])

    job = job_manager.create_job(
        job_type=JobType.MODERNIZE,
        command=cmd,
        parameters=req.model_dump(),
    )
    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    return job


@router.post("/reduce", response_model=JobResponse)
async def run_reduce(req: ManifoldReduceRequest) -> JobResponse:
    """Launch asynchronous manifold scale reduction."""
    cmd = [
        venv_python(),
        "tools/manifold_reduce.py",
        "--reduce",
    ]
    if req.dataset_key:
        cmd.extend(["--model", req.dataset_key])
    if req.scale:
        cmd.extend(["--scale", str(req.scale)])

    job = job_manager.create_job(
        job_type=JobType.REDUCE,
        command=cmd,
        parameters=req.model_dump(),
    )
    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    return job


@router.post("/regenerate-manifolds")
def regenerate_manifolds_documentation() -> Dict[str, Any]:
    """Regenerate authoritative manifolds.md matrix from on-disk status and registries."""
    import subprocess
    cmd = [venv_python(), "tools/regenerate_manifolds_md.py"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate manifolds.md: {proc.stderr}",
        )
    return {
        "status": "success",
        "message": proc.stdout.strip(),
    }
