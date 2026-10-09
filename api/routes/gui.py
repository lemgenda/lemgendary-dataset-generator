"""Aggregated desktop GUI endpoints for LemGendary AI Studio."""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import re
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
    DatasetSourceInfo,
    DatasetStatsListResponse,
    GuiStateResponse,
    JobResponse,
    JobState,
    JobType,
    PresetListResponse,
    QuickCompileRequest,
    CustomCompileRequest,
)
from api.routes.datasets import _resolve_output_root
from api.routes.health import _START_TIME, get_hardware_info
from core.cli_args import __version__, venv_python
from core import presets

logger = logging.getLogger("lemgendary.api.routes.gui")
router = APIRouter(prefix="/gui", tags=["Desktop GUI"])


_STATS_CACHE: Dict[str, tuple[float, tuple[DatasetFormatStats, int, int, bool, float, int]]] = {}


def _calculate_dataset_file_stats(manifold_path: Path) -> tuple[DatasetFormatStats, int, int, bool, float, int]:
    """Fast scan of manifold directory with container prioritization and mtime caching."""
    try:
        current_mtime = manifold_path.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    cache_key = str(manifold_path.resolve())
    if cache_key in _STATS_CACHE:
        cached_mtime, cached_val = _STATS_CACHE[cache_key]
        if cached_mtime == current_mtime:
            return cached_val

    format_counts = {
        "webp": 0,
        "jpg": 0,
        "png": 0,
        "parquet": 0,
        "other": 0,
    }
    total_size = 0
    file_count = 0
    shards_count = 0

    # 1. First inspect container directories (shards, mds, wds, litdata) and their split subdirectories
    for container_name in ("shards", "mds", "wds", "litdata"):
        c_dir = manifold_path / container_name
        if c_dir.exists() and c_dir.is_dir():
            dirs_to_check = [c_dir]
            try:
                for sub in os.scandir(c_dir):
                    if sub.is_dir():
                        dirs_to_check.append(Path(sub.path))
            except OSError:
                pass

            for check_d in dirs_to_check:
                try:
                    for entry in os.scandir(check_d):
                        if entry.is_file():
                            fname = entry.name
                            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                            if fname == "index.json":
                                pass
                            elif ext in ("tar", "bin", "mds", "parquet", "zstd"):
                                shards_count += 1
                            elif fname.startswith("chunk") or fname.startswith("shard"):
                                shards_count += 1
                            elif container_name == "mds" and not ext:
                                shards_count += 1

                            if ext == "parquet":
                                format_counts["parquet"] += 1
                            file_count += 1
                            try:
                                total_size += entry.stat().st_size
                            except OSError:
                                pass
                except OSError as exc:
                    logger.debug("Failed scanning container folder %s: %s", check_d, exc)

    # 2. Check root-level files in manifold_path (e.g. .parquet, dataset_info.yaml)
    try:
        for entry in os.scandir(manifold_path):
            if entry.is_file():
                fname = entry.name
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if fname != "index.json" and (ext in ("tar", "bin", "mds", "parquet", "zstd") or fname.startswith("shard") or fname.startswith("chunk")):
                    shards_count += 1
                if ext == "parquet":
                    format_counts["parquet"] += 1
                try:
                    total_size += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass

    # 3. Check split directories for directory-based manifolds (e.g. YOLO images/train, images/val)
    if shards_count == 0:
        images_dir = manifold_path / "images"
        if images_dir.exists() and images_dir.is_dir():
            try:
                split_dirs = [s for s in os.scandir(images_dir) if s.is_dir()]
                if split_dirs:
                    shards_count = len(split_dirs)
            except OSError:
                pass

    # 4. Only scan loose image directories (images, targets, masks, labels) if no shards found or few files
    if shards_count == 0 or file_count < 10:
        scan_dirs = [
            manifold_path / "images",
            manifold_path / "targets",
            manifold_path / "masks",
            manifold_path / "labels",
        ]
        for directory in scan_dirs:
            if not directory.exists():
                continue
            try:
                for root_d, _, files in os.walk(directory):
                    for fname in files:
                        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                        if ext == "webp":
                            format_counts["webp"] += 1
                        elif ext in ("jpg", "jpeg"):
                            format_counts["jpg"] += 1
                        elif ext == "png":
                            format_counts["png"] += 1
                        elif ext == "parquet":
                            format_counts["parquet"] += 1
                        elif ext in ("txt", "json", "yaml", "md", "ipynb"):
                            continue
                        else:
                            format_counts["other"] += 1
                        file_count += 1
                        try:
                            total_size += (Path(root_d) / fname).stat().st_size
                        except OSError:
                            pass
            except OSError as scan_exc:
                logger.debug("Failed scanning %s: %s", directory, scan_exc)

    formats = DatasetFormatStats(
        webp=format_counts["webp"],
        jpg=format_counts["jpg"],
        png=format_counts["png"],
        parquet=format_counts["parquet"],
        other=format_counts["other"],
    )

    result = (formats, file_count, total_size, False, 0.0, shards_count)
    _STATS_CACHE[cache_key] = (current_mtime, result)
    return result


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

    # Load canonical manifold registry metadata
    unified_yaml = Path("unified_data.yaml")
    if not unified_yaml.exists():
        unified_yaml = Path(__file__).resolve().parent.parent.parent / "unified_data.yaml"

    registry_datasets: Dict[str, Any] = {}
    if unified_yaml.exists():
        try:
            with open(unified_yaml, "r", encoding="utf-8") as f:
                ydata = yaml.safe_load(f) or {}
                registry_datasets = ydata.get("datasets", {})
        except Exception as exc:
            logger.debug("Failed reading unified_data.yaml: %s", exc)

    folder_to_meta: Dict[str, tuple[str, Dict[str, Any]]] = {}
    for k, v in registry_datasets.items():
        m_folder = v.get("modernized_folder") or f"LemGendized{v.get('name', '')}"
        folder_to_meta[m_folder.lower()] = (k, v)
        if "name" in v:
            folder_to_meta[v["name"].lower()] = (k, v)

    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        meta_match = folder_to_meta.get(entry.name.lower())
        reg_key = meta_match[0] if meta_match else entry.name
        reg_info = meta_match[1] if meta_match else {}

        display_name = reg_info.get("title")
        if not display_name:
            if entry.name == "LemGendizedUpnV2":
                display_name = "Unified Perceptual Net V2"
            elif entry.name.startswith("LemGendized"):
                cleaned = entry.name.replace("LemGendized", "")
                display_name = re.sub(r"([A-Z])", r" \1", cleaned).strip()
            else:
                display_name = entry.name

        canonical_format = reg_info.get("canonical_format") or "webdataset"
        format_val = reg_info.get("container", {}).get("primary") or canonical_format

        info_file = entry / "dataset_info.yaml"
        sample_count = 0
        task = reg_info.get("task", "vision")

        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                    task = meta.get("task", task)
                    sample_count = meta.get("count") or meta.get("total_samples") or meta.get("samples") or 0
                    if meta.get("canonical_format"):
                        canonical_format = meta.get("canonical_format")
                    if meta.get("format"):
                        format_val = meta.get("format")
            except Exception as exc:
                logger.debug("Failed reading %s: %s", info_file, exc)

        formats, counted_files, size_bytes, has_hardlinks, ratio, shards_count = _calculate_dataset_file_stats(entry)
        if sample_count == 0:
            sample_count = formats.webp + formats.jpg + formats.png + formats.parquet

        is_mds = (entry / "mds").exists() or format_val == "mds" or canonical_format == "mds"
        if is_mds:
            format_val = "mds"
            canonical_format = "mds"
            if shards_count == 0 and (entry / "mds" / "index.json").exists():
                shards_count = 1

        is_compiled = bool(
            shards_count > 0
            or formats.parquet > 0
            or is_mds
            or (format_val == "directory" and sample_count > 0)
            or (formats.webp > 0 or formats.jpg > 0 or formats.png > 0)
        )

        # Build upstream sources with sample counts
        sources: List[DatasetSourceInfo] = []
        raw_sources: List[str] = []
        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                    raw_sources = meta.get("original_sources") or []
            except Exception:
                pass

        if not raw_sources:
            raw_sources = reg_info.get("provenance_sources") or []

        refs = reg_info.get("refs") or []
        ref_lookup: Dict[str, str] = {}
        for r_item in refs:
            if isinstance(r_item, dict) and "ref" in r_item:
                r_val = str(r_item["ref"])
                r_name = r_val.split("/")[-1]
                ref_lookup[r_name.lower()] = r_val
                ref_lookup[r_val.lower()] = r_val

        n_sources = max(len(raw_sources), 1)
        base_count = sample_count // n_sources if sample_count > 0 else None

        for s_idx, s_entry in enumerate(raw_sources):
            s_name = str(s_entry).strip()
            s_lower = s_name.lower()
            s_type = "SOURCE"
            if any(k in s_lower for k in ("kaggle", "ava", "coco", "div2k", "flickr", "voc")):
                s_type = "KAGGLE"
            elif any(k in s_lower for k in ("hf", "huggingface", "tad66k", "spaq")):
                s_type = "HUGGINGFACE"
            elif any(k in s_lower for k in ("sub-manifold", "manifold", "multi-task")):
                s_type = "SUB-MANIFOLD"
            elif any(k in s_lower for k in ("terminal", "forex", "mt5")):
                s_type = "TERMINAL"

            this_count = None
            if sample_count > 0:
                if s_idx == n_sources - 1:
                    this_count = sample_count - (base_count * (n_sources - 1))
                else:
                    this_count = base_count

            matched_ref = None
            for k_ref, v_ref in ref_lookup.items():
                if k_ref in s_lower or s_lower in k_ref:
                    matched_ref = v_ref
                    break

            sources.append(DatasetSourceInfo(name=s_name, ref=matched_ref, count=this_count, type=s_type))

        kaggle_ref = reg_info.get("kaggle_ref")

        total_storage_bytes += size_bytes
        dataset_stats.append(
            DatasetDetailStats(
                name=entry.name,
                key=reg_key,
                display_name=display_name,
                path=str(entry),
                task=task,
                sample_count=sample_count,
                size_bytes=size_bytes,
                size_gb=round(size_bytes / (1024**3), 3),
                format=format_val if is_compiled else "directory",
                canonical_format=canonical_format,
                formats=formats,
                shards_count=shards_count,
                is_compiled=is_compiled,
                has_hardlinks=has_hardlinks,
                hardlink_ratio=ratio,
                sources=sources,
                kaggle_ref=kaggle_ref,
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
) -> JobResponse:
    """Rapid dispatch for compilation jobs using a predefined compiler preset profile."""
    try:
        preset_cfg = presets.get_preset(req.preset)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    cmd = [
        venv_python(),
        "core/manifold_compile.py",
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


@router.post("/custom-compile", response_model=JobResponse)
async def custom_compile(
    req: CustomCompileRequest,
) -> JobResponse:
    """Compile a new custom dataset manifold from a list of multi-source repositories."""
    name = req.custom_name.strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Custom dataset name cannot be empty.",
        )

    pascal_name = "".join(w.capitalize() for w in re.split(r"[^a-zA-Z0-9]", name) if w)
    if not pascal_name:
        pascal_name = "CustomManifold"

    model_key = re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_name).lower()
    modernized_folder = f"LemGendized{pascal_name}"

    cleaned_refs: List[str] = []
    for s in req.sources:
        src = s.strip()
        if not src:
            continue
        if "kaggle.com/datasets/" in src:
            repo = src.split("kaggle.com/datasets/")[-1].split("?")[0].strip("/")
            cleaned_refs.append(f"kaggle://{repo}")
        elif "huggingface.co/datasets/" in src:
            repo = src.split("huggingface.co/datasets/")[-1].split("?")[0].strip("/")
            cleaned_refs.append(f"hf://{repo}")
        elif "huggingface.co/" in src:
            repo = src.split("huggingface.co/")[-1].split("?")[0].strip("/")
            cleaned_refs.append(f"hf://{repo}")
        elif "github.com/" in src:
            repo = src.split("github.com/")[-1].split("?")[0].replace(".git", "").strip("/")
            cleaned_refs.append(f"gh://{repo}")
        elif "drive.google.com" in src:
            cleaned_refs.append(f"gd://{src}")
        else:
            cleaned_refs.append(src)

    # Persist into unified_data.yaml SSOT
    unified_yaml = Path("unified_data.yaml")
    if not unified_yaml.exists():
        unified_yaml = Path(__file__).resolve().parent.parent.parent / "unified_data.yaml"

    if unified_yaml.exists():
        try:
            with open(unified_yaml, "r", encoding="utf-8") as f:
                ydata = yaml.safe_load(f) or {}
            if "datasets" not in ydata:
                ydata["datasets"] = {}

            ydata["datasets"][model_key] = {
                "name": pascal_name,
                "title": f"LemGendized {pascal_name}",
                "canonical_format": req.canonical_format,
                "task": req.task,
                "modernized_folder": modernized_folder,
                "refs": [{"ref": r} for r in cleaned_refs],
                "container": {"primary": req.canonical_format},
            }
            with open(unified_yaml, "w", encoding="utf-8") as f:
                yaml.dump(ydata, f, default_flow_style=False, sort_keys=False)
        except Exception as exc:
            logger.error("Failed persisting custom manifold definition: %s", exc)

    cmd = [
        venv_python(),
        "core/manifold_compile.py",
        "--model", model_key,
        "--preset", req.preset,
    ]
    if req.purge_loose_images:
        cmd.append("--cleanup")

    job = job_manager.create_job(
        job_type=JobType.COMPILE,
        command=cmd,
        parameters={
            "custom_name": req.custom_name,
            "model_key": model_key,
            "pascal_name": pascal_name,
            "preset": req.preset,
            "canonical_format": req.canonical_format,
            "shard_size": req.shard_size,
            "sources": cleaned_refs,
            "purge_loose_images": req.purge_loose_images,
        },
    )

    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    logger.info("Dispatched custom compile job %s for custom manifold %s", job.id, pascal_name)
    return job

