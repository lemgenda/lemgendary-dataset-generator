"""Job management and WebSocket log streaming endpoints."""

from __future__ import annotations

import asyncio
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status

from api.events import manager
from api.jobs import job_manager
from api.models import (
    CompileJobRequest,
    DegradeJobRequest,
    GenericJobRequest,
    JobListResponse,
    JobResponse,
    JobState,
    JobType,
)
from cli_args import venv_python

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("", response_model=JobListResponse)
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    state: Optional[JobState] = None,
    job_type: Optional[JobType] = None,
) -> JobListResponse:
    """List historical and active jobs with pagination."""
    jobs = job_manager.list_jobs(limit=limit, offset=offset, state=state, job_type=job_type)
    total = job_manager.count_total_jobs()
    return JobListResponse(jobs=jobs, total=total)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Retrieve detailed state of a single job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return job


@router.get("/{job_id}/logs")
async def get_job_logs(job_id: str, tail: Optional[int] = Query(None, ge=1)) -> dict[str, str]:
    """Retrieve raw text logs for a specific job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    content = job_manager.get_log_content(job_id, tail_lines=tail)
    return {"job_id": job_id, "logs": content}


@router.post("/compile", response_model=JobResponse)
async def submit_compile_job(req: CompileJobRequest) -> JobResponse:
    """Submit a manifold compile job."""
    cmd = [venv_python(), "manifold_compile.py"]
    if req.model:
        cmd.extend(["--model", req.model])
    if req.max_gb is not None:
        cmd.extend(["--max_gb", str(req.max_gb)])
    if req.suffix:
        cmd.extend(["--suffix", req.suffix])
    if req.workers is not None:
        cmd.extend(["--workers", str(req.workers)])
    if req.no_vetting:
        cmd.append("--no-vetting")
    if req.no_labeling:
        cmd.append("--no-labeling")
    if req.no_hash:
        cmd.append("--no-hash")
    if req.image_format:
        cmd.extend(["--image-format", req.image_format])
    if req.image_quality is not None:
        cmd.extend(["--image-quality", str(req.image_quality)])
    if req.target_quality is not None:
        cmd.extend(["--target-quality", str(req.target_quality)])
    if req.mask_format:
        cmd.extend(["--mask-format", req.mask_format])
    if req.also_format:
        cmd.extend(["--also-format", req.also_format])
    if req.force_duplicate:
        cmd.append("--force-duplicate")
    if req.accept_space_loss:
        cmd.append("--accept-space-loss")
    if req.label_strategy:
        cmd.extend(["--label-strategy", req.label_strategy])
    if req.prompt_strategy:
        cmd.extend(["--prompt-strategy", req.prompt_strategy])
    if req.mask_strategy:
        cmd.extend(["--mask-strategy", req.mask_strategy])

    job = job_manager.create_job(
        job_type=JobType.COMPILE,
        command=cmd,
        parameters=req.model_dump(),
    )
    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    return job


@router.post("/degrade", response_model=JobResponse)
async def submit_degrade_job(req: DegradeJobRequest) -> JobResponse:
    """Submit a synthetic degradation manifold derivation job."""
    cmd = [
        venv_python(), "generate_degrade.py",
        "--source", req.source,
        "--output", req.output,
        "--profile", req.profile,
        "--intensity", req.intensity,
        "--val-split", str(req.val_split),
        "--seed", str(req.seed),
        "--image-format", req.image_format,
    ]
    if req.pairs is not None:
        cmd.extend(["--pairs", str(req.pairs)])
    if req.workers is not None:
        cmd.extend(["--workers", str(req.workers)])
    if req.dry_run:
        cmd.append("--dry-run")

    job = job_manager.create_job(
        job_type=JobType.DEGRADE,
        command=cmd,
        parameters=req.model_dump(),
    )
    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    return job


@router.post("/generic", response_model=JobResponse)
async def submit_generic_job(req: GenericJobRequest) -> JobResponse:
    """Submit a generic job command."""
    job = job_manager.create_job(
        job_type=req.job_type,
        command=req.command,
        parameters=req.parameters,
    )
    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    return job


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, str]:
    """Terminate an active job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is not running or already finalized",
        )
    return {"status": "cancelled", "job_id": job_id}


ws_router = APIRouter(prefix="/ws/jobs", tags=["Jobs"])


@ws_router.websocket("/{job_id}/logs")
async def stream_job_logs(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint to replay backlog logs and stream live lines for a job."""
    job = job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    await manager.connect_job(job_id, websocket)

    # Replay backlog logs already on disk
    backlog = job_manager.get_log_content(job_id)
    if backlog:
        try:
            await websocket.send_json({"job_id": job_id, "type": "backlog", "chunk": backlog})
        except Exception:
            await manager.disconnect_job(job_id, websocket)
            return

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect_job(job_id, websocket)
    except Exception:
        await manager.disconnect_job(job_id, websocket)
