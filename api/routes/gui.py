"""Aggregated desktop GUI endpoints for LemGendary AI Studio."""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import verify_token
from api.jobs import job_manager
from api.models import (
    ActiveJobsResponse,
    ActiveJobTelemetry,
    CompilerPresetModel,
    DatasetDetailStats,
    DatasetFormatStats,
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
from cli_args import __version__, venv_python
import presets

logger = logging.getLogger("lemgendary.api.routes.gui")
router = APIRouter(prefix="/gui", tags=["Desktop GUI"])


def _calculate_dataset_file_stats(manifold_path: Path) -> tuple[DatasetFormatStats, int, int, bool, float]:
    """Scan manifold directory to compute format breakdown, sample count, size, and hardlink metrics."""
    format_counts = {
        "webp": 0,
        "jpg": 0,
        "png": 0,
        "parquet": 0,
        "other": 0,
    }
    total_size = 0
    file_count = 0
    unique_inodes: set[int] = set()
    hardlink_count = 0

    scan_dirs = [
        manifold_path / "images",
        manifold_path / "targets",
        manifold_path / "masks",
        manifold_path,
    ]

    scanned_paths: set[Path] = set()

    for directory in scan_dirs:
        if not directory.exists():
            continue
        try:
            for entry in os.scandir(directory):
                if not entry.is_file():
                    continue
                file_path = Path(entry.path)
                if file_path in scanned_paths:
                    continue
                scanned_paths.add(file_path)

                ext = file_path.suffix.lower().lstrip(".")
                if ext == "webp":
                    format_counts["webp"] += 1
                elif ext in ("jpg", "jpeg"):
                    format_counts["jpg"] += 1
                elif ext == "png":
                    format_counts["png"] += 1
                elif ext == "parquet":
                    format_counts["parquet"] += 1
                elif ext in ("txt", "json", "yaml", "md"):
                    continue
                else:
                    format_counts["other"] += 1

                file_count += 1
                try:
                    stat_res = entry.stat()
                    total_size += stat_res.st_size
                    if hasattr(stat_res, "st_ino") and stat_res.st_ino != 0:
                        if stat_res.st_ino in unique_inodes:
                            hardlink_count += 1
                        else:
                            unique_inodes.add(stat_res.st_ino)
                except OSError as stat_exc:
                    logger.debug("Failed stat for %s: %s", entry.path, stat_exc)
        except OSError as scan_exc:
            logger.debug("Failed scanning %s: %s", directory, scan_exc)

    formats = DatasetFormatStats(
        webp=format_counts["webp"],
        jpg=format_counts["jpg"],
        png=format_counts["png"],
        parquet=format_counts["parquet"],
        other=format_counts["other"],
    )

    has_hardlinks = hardlink_count > 0
    ratio = round(hardlink_count / max(1, file_count), 4) if has_hardlinks else 0.0

    return formats, file_count, total_size, has_hardlinks, ratio


@router.get("/state", response_model=GuiStateResponse)
async def get_gui_state() -> GuiStateResponse:
    """Aggregated sidecar status snapshot for LemGendary AI Studio Desktop GUI hydration."""
    root = _resolve_output_root()
    total_datasets = 0
    total_bytes = 0

    if root.exists():
        try:
            for entry in root.iterdir():
                if entry.is_dir() and not entry.name.startswith("."):
                    total_datasets += 1
        except OSError as exc:
            logger.debug("Error listing root directory %s: %s", root, exc)

    all_presets = presets.list_presets()
    hw = get_hardware_info()
    uptime = round(time.time() - _START_TIME, 2)

    return GuiStateResponse(
        service="LemGendary Dataset Compiler API",
        version=__version__,
        uptime_seconds=uptime,
        active_jobs_count=job_manager.count_active_jobs(),
        total_datasets_count=total_datasets,
        total_storage_bytes=total_bytes,
        total_storage_gb=round(total_bytes / (1024**3), 2),
        hardware=hw,
        presets=sorted(list(all_presets.keys())),
    )


@router.get("/datasets/with-stats", response_model=DatasetStatsListResponse)
async def get_datasets_with_stats() -> DatasetStatsListResponse:
    """Enriched dataset catalog with detailed format breakdown, storage metrics, and hardlinks."""
    root = _resolve_output_root()
    dataset_stats: List[DatasetDetailStats] = []
    total_storage_bytes = 0

    if not root.exists():
        return DatasetStatsListResponse(
            datasets=[],
            total=0,
            total_size_bytes=0,
            total_size_gb=0.0,
        )

    for entry in root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        info_file = entry / "dataset_info.yaml"
        sample_count = 0
        task = "vision"

        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                    task = meta.get("task", task)
                    sample_count = meta.get("total_samples", 0)
            except Exception as exc:
                logger.debug("Failed reading %s: %s", info_file, exc)

        formats, counted_files, size_bytes, has_hardlinks, ratio = _calculate_dataset_file_stats(entry)
        if sample_count == 0:
            sample_count = formats.webp + formats.jpg + formats.png + formats.parquet

        total_storage_bytes += size_bytes
        dataset_stats.append(
            DatasetDetailStats(
                name=entry.name,
                path=str(entry),
                task=task,
                sample_count=sample_count,
                size_bytes=size_bytes,
                size_gb=round(size_bytes / (1024**3), 3),
                formats=formats,
                has_hardlinks=has_hardlinks,
                hardlink_ratio=ratio,
            )
        )

    return DatasetStatsListResponse(
        datasets=dataset_stats,
        total=len(dataset_stats),
        total_size_bytes=total_storage_bytes,
        total_size_gb=round(total_storage_bytes / (1024**3), 3),
    )


@router.get("/jobs/active", response_model=ActiveJobsResponse)
async def get_active_jobs_telemetry() -> ActiveJobsResponse:
    """Detailed live execution telemetry for running compiler background jobs."""
    running_jobs = job_manager.list_jobs(state=JobState.RUNNING)
    pending_jobs = job_manager.list_jobs(state=JobState.PENDING)
    all_active = running_jobs + pending_jobs

    telemetry_list: List[ActiveJobTelemetry] = []

    for job in all_active:
        started = job.started_at
        elapsed = 0.0
        fps = 0.0
        current_sample = 0
        total_samples = 0
        progress_pct = 0.0
        eta = None

        if started:
            try:
                start_dt = datetime.datetime.fromisoformat(started)
                now_dt = datetime.datetime.now(datetime.timezone.utc)
                elapsed = max(0.1, (now_dt - start_dt).total_seconds())
            except Exception as dt_exc:
                logger.debug("Failed parsing start timestamp for job %s: %s", job.id, dt_exc)

        # Parse log tail to estimate progress if available
        log_content = job_manager.get_log_content(job.id, tail_lines=20)
        if log_content:
            for line in reversed(log_content.splitlines()):
                if "%" in line and "/" in line:
                    parts = line.split("%")[0].strip().split()
                    if parts:
                        try:
                            progress_pct = float(parts[-1].strip())
                        except ValueError as val_exc:
                            logger.debug("Could not parse progress percentage from line '%s': %s", line, val_exc)
                    break

        telemetry_list.append(
            ActiveJobTelemetry(
                id=job.id,
                job_type=job.job_type.value,
                state=job.state.value,
                created_at=job.created_at,
                started_at=job.started_at,
                progress_percent=progress_pct,
                current_sample=current_sample,
                total_samples=total_samples,
                elapsed_seconds=round(elapsed, 1),
                eta_seconds=eta,
                fps=round(fps, 1),
            )
        )

    return ActiveJobsResponse(active_jobs=telemetry_list, count=len(telemetry_list))


@router.get("/presets", response_model=PresetListResponse)
async def list_compiler_presets() -> PresetListResponse:
    """Retrieve all available compiler presets with parameters for GUI configuration."""
    raw_presets = presets.list_presets()
    converted: Dict[str, CompilerPresetModel] = {}

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
    authenticated: str = Depends(verify_token),
) -> JobResponse:
    """Rapid dispatch for compilation jobs using a predefined compiler preset profile."""
    del authenticated
    try:
        preset_cfg = presets.get_preset(req.preset)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    cmd = [
        venv_python(),
        "manifold_compile.py",
        "--model", req.model,
        "--preset", req.preset,
    ]

    if req.max_gb is not None:
        cmd.extend(["--max_gb", str(req.max_gb)])
    if req.workers is not None:
        cmd.extend(["--workers", str(req.workers)])

    job = job_manager.create_job(
        job_type=JobType.COMPILE,
        command=cmd,
        parameters={
            "model": req.model,
            "preset": req.preset,
            "max_gb": req.max_gb,
            "workers": req.workers,
            "preset_details": preset_cfg.to_dict(),
        },
    )

    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    logger.info("Dispatched quick-compile job %s with preset %s for model %s", job.id, req.preset, req.model)
    return job
