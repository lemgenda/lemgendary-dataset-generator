"""Aggregated desktop GUI endpoints for LemGendary AI Studio.

The routes stay thin: dataset catalog scanning lives in :mod:`api.dataset_stats`
and compile-request translation in :mod:`api.gui_compile`.
"""

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from api.dataset_stats import build_dataset_stats_response
from api.gui_compile import (
    build_custom_entry,
    clean_source_refs,
    custom_compile_command,
    custom_compile_parameters,
    derive_manifold_names,
    persist_custom_manifold,
    quick_compile_command,
    resolve_custom_preset,
)
from api.jobs import job_manager, submit_job
from api.models import (
    ActiveJobsResponse,
    ActiveJobTelemetry,
    CompilerPresetModel,
    CustomCompileRequest,
    DatasetStatsListResponse,
    GuiStateResponse,
    JobResponse,
    JobState,
    JobType,
    PresetListResponse,
    QuickCompileRequest,
)
from api.routes.datasets import _resolve_output_root
from api.routes.health import _START_TIME, get_hardware_info
from core import presets
from core.cli_args import __version__

__all__ = [
    "custom_compile",
    "get_active_jobs_telemetry",
    "get_datasets_with_stats",
    "get_gui_state",
    "job_manager",
    "list_compiler_presets",
    "logger",
    "quick_compile",
    "router",
]

logger = logging.getLogger("lemgendary.api.routes.gui")
router = APIRouter(prefix="/gui", tags=["Desktop GUI"])


def _count_dataset_folders(root: Path) -> int:
    total_datasets = 0
    if root.exists():
        try:
            for entry in root.iterdir():
                if entry.is_dir() and not entry.name.startswith("."):
                    total_datasets += 1
        except OSError as exc:
            logger.debug("Error listing root directory %s: %s", root, exc)
    return total_datasets


@router.get("/state", response_model=GuiStateResponse)
async def get_gui_state() -> GuiStateResponse:
    """Aggregated sidecar status snapshot for LemGendary AI Studio Desktop GUI hydration."""
    total_datasets = _count_dataset_folders(_resolve_output_root())
    total_bytes = 0
    all_presets = presets.list_presets()

    return GuiStateResponse(
        service="LemGendary Dataset Compiler API",
        version=__version__,
        uptime_seconds=round(time.time() - _START_TIME, 2),
        active_jobs_count=job_manager.count_active_jobs(),
        total_datasets_count=total_datasets,
        total_storage_bytes=total_bytes,
        total_storage_gb=round(total_bytes / (1024**3), 2),
        hardware=get_hardware_info(),
        presets=sorted(all_presets.keys()),
    )


@router.get("/datasets/with-stats", response_model=DatasetStatsListResponse)
def get_datasets_with_stats() -> DatasetStatsListResponse:
    """Enriched dataset catalog with detailed format breakdown, storage metrics, and hardlinks."""
    return build_dataset_stats_response(_resolve_output_root())


def _elapsed_seconds(job: JobResponse) -> float:
    if not job.started_at:
        return 0.0
    try:
        start_dt = datetime.datetime.fromisoformat(job.started_at)
        now_dt = datetime.datetime.now(datetime.UTC)
        return max(0.1, (now_dt - start_dt).total_seconds())
    except (TypeError, ValueError) as dt_exc:
        logger.debug("Failed parsing start timestamp for job %s: %s", job.id, dt_exc)
        return 0.0


def _progress_from_log(log_content: str) -> float:
    """Progress percentage from the last log line that looks like a progress bar."""
    for line in reversed(log_content.splitlines()):
        if "%" in line and "/" in line:
            parts = line.split("%")[0].strip().split()
            if parts:
                try:
                    return float(parts[-1].strip())
                except ValueError as val_exc:
                    logger.debug(
                        "Could not parse progress percentage from line '%s': %s", line, val_exc
                    )
            break
    return 0.0


def _job_telemetry(job: JobResponse) -> ActiveJobTelemetry:
    elapsed = _elapsed_seconds(job)
    # Parse log tail to estimate progress if available
    log_content = job_manager.get_log_content(job.id, tail_lines=20)
    return ActiveJobTelemetry(
        id=job.id,
        job_type=job.job_type.value,
        state=job.state.value,
        created_at=job.created_at,
        started_at=job.started_at,
        progress_percent=_progress_from_log(log_content) if log_content else 0.0,
        current_sample=0,
        total_samples=0,
        elapsed_seconds=round(elapsed, 1),
        eta_seconds=None,
        fps=0.0,
    )


@router.get("/jobs/active", response_model=ActiveJobsResponse)
async def get_active_jobs_telemetry() -> ActiveJobsResponse:
    """Detailed live execution telemetry for running compiler background jobs."""
    running_jobs = job_manager.list_jobs(state=JobState.RUNNING)
    pending_jobs = job_manager.list_jobs(state=JobState.PENDING)
    telemetry_list: list[ActiveJobTelemetry] = [
        _job_telemetry(job) for job in running_jobs + pending_jobs
    ]
    return ActiveJobsResponse(active_jobs=telemetry_list, count=len(telemetry_list))


@router.get("/presets", response_model=PresetListResponse)
async def list_compiler_presets() -> PresetListResponse:
    """Retrieve all available compiler presets with parameters for GUI configuration."""
    raw_presets = presets.list_presets()
    converted: dict[str, CompilerPresetModel] = {}

    for name, p in raw_presets.items():
        converted[name] = CompilerPresetModel(
            name=p.name,
            title=p.title,
            description=p.description,
            image_format=p.image_format,
            image_quality=p.image_quality,
            target_quality=p.target_quality,
            mask_format=p.mask_format,
            min_resolution=p.min_resolution,
            resampling=p.resampling,
            vetting_enabled=p.vetting_enabled,
            labeling_enabled=p.labeling_enabled,
            hardlink_gate=p.hardlink_gate,
            containers=p.containers,
        )

    return PresetListResponse(presets=converted, total=len(converted))


@router.post("/quick-compile", response_model=JobResponse)
async def quick_compile(
    req: QuickCompileRequest,
) -> JobResponse:
    """Rapid dispatch for compilation jobs using a predefined compiler preset profile."""
    try:
        preset_cfg = presets.get_preset(req.preset)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    job = submit_job(
        job_manager,
        JobType.COMPILE,
        quick_compile_command(req),
        {
            "model": req.model,
            "preset": req.preset,
            "max_gb": req.max_gb,
            "workers": req.workers,
            "preset_details": preset_cfg.to_dict(),
        },
    )
    logger.info(
        "Dispatched quick-compile job %s with preset %s for model %s", job.id, req.preset, req.model
    )
    return job


@router.post("/custom-compile", response_model=JobResponse)
async def custom_compile(
    req: CustomCompileRequest,
) -> JobResponse:
    """Compile a new custom dataset manifold from a list of multi-source repositories.

    The preset is validated and the registry entry is written (atomically,
    never overwriting an existing key) before any job is created; if either
    step fails, no job is started.
    """
    name = req.custom_name.strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Custom dataset name cannot be empty.",
        )
    preset_name = resolve_custom_preset(req.preset)
    names = derive_manifold_names(name)
    cleaned_refs = clean_source_refs(req.sources)
    persist_custom_manifold(names.model_key, build_custom_entry(names, req, cleaned_refs))

    job = submit_job(
        job_manager,
        JobType.COMPILE,
        custom_compile_command(names.model_key, preset_name, req.purge_loose_images),
        custom_compile_parameters(req, names, preset_name, cleaned_refs),
    )
    logger.info(
        "Dispatched custom compile job %s for custom manifold %s", job.id, names.pascal_name
    )
    return job
