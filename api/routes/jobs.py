"""Job management and WebSocket log streaming endpoints."""

from __future__ import annotations

import logging
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status

from api.events import manager
from api.jobs import job_manager, submit_job
from api.models import (
    CompileJobRequest,
    DegradeJobRequest,
    GenericJobRequest,
    JobListResponse,
    JobResponse,
    JobState,
    JobType,
)
from core.cli_args import venv_python

logger = logging.getLogger("lemgendary.api.routes.jobs")

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


# (request field, CLI flag, kind) in argv order. "value": pass the value when truthy;
# "number": pass str(value) when not None; "switch": bare flag when truthy.
_COMPILE_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("model", "--model", "value"),
    ("max_gb", "--max_gb", "number"),
    ("suffix", "--suffix", "value"),
    ("workers", "--workers", "number"),
    ("no_vetting", "--no-vetting", "switch"),
    ("no_labeling", "--no-labeling", "switch"),
    ("no_hash", "--no-hash", "switch"),
    ("image_format", "--image-format", "value"),
    ("image_quality", "--image-quality", "number"),
    ("target_quality", "--target-quality", "number"),
    ("mask_format", "--mask-format", "value"),
    ("also_format", "--also-format", "value"),
    ("force_duplicate", "--force-duplicate", "switch"),
    ("accept_space_loss", "--accept-space-loss", "switch"),
    ("label_strategy", "--label-strategy", "value"),
    ("prompt_strategy", "--prompt-strategy", "value"),
    ("mask_strategy", "--mask-strategy", "value"),
)


def _compile_flag_args(value: Any, flag: str, kind: str) -> list[Any]:
    if kind == "number":
        return [flag, str(value)] if value is not None else []
    if not value:
        return []
    return [flag] if kind == "switch" else [flag, value]


def _compile_command(req: CompileJobRequest) -> list[Any]:
    """``core.manifold_compile`` argv for a compile job request."""
    cmd: list[Any] = [venv_python(), "-m", "core.manifold_compile"]
    for field_name, flag, kind in _COMPILE_FLAGS:
        cmd.extend(_compile_flag_args(getattr(req, field_name), flag, kind))
    return cmd


@router.post("/compile", response_model=JobResponse)
async def submit_compile_job(req: CompileJobRequest) -> JobResponse:
    """Submit a manifold compile job."""
    return submit_job(job_manager, JobType.COMPILE, _compile_command(req), req.model_dump())


@router.post("/degrade", response_model=JobResponse)
async def submit_degrade_job(req: DegradeJobRequest) -> JobResponse:
    """Start a degradation job that synthesises a paired manifold from a clean source."""
    cmd = [
        venv_python(), "-m", "tools.generate_degrade",
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

    return submit_job(job_manager, JobType.DEGRADE, cmd, req.model_dump())


@router.post("/generic", response_model=JobResponse)
async def submit_generic_job(req: GenericJobRequest) -> JobResponse:
    """Submit a generic job command."""
    return submit_job(job_manager, req.job_type, req.command, req.parameters)


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
        except (WebSocketDisconnect, RuntimeError, OSError) as exc:
            logger.debug("Failed sending log backlog for job %s; dropping client: %s", job_id, exc)
            await manager.disconnect_job(job_id, websocket)
            return

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect_job(job_id, websocket)
    except (RuntimeError, OSError) as exc:
        logger.debug("Job %s websocket client error: %s", job_id, exc)
        await manager.disconnect_job(job_id, websocket)
