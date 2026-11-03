"""Modernization queue, batch conversion, and manifold downsampling endpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from api.jobs import job_manager, submit_job
from api.models import JobResponse, JobType
from core.cli_args import venv_python
from services.doc_service import DocService
from tools.modernization_queue import build_queue_args, load_queue_manifest
from tools.modernize.queue_items import (
    API_LABEL_STYLE,
    ManifoldQueueItem,
    ManifoldRegistryEntry,
    queue_status_labels,
)

logger = logging.getLogger("lemgendary.api.routes.modernize")
router = APIRouter(prefix="/modernize", tags=["Modernization Hub"])


@dataclass
class QueueItemResponse(ManifoldRegistryEntry):
    """One dataset in the modernization queue."""

    is_containerized: bool
    has_loose_files: bool
    container_bytes: int
    container_gb: float
    status_label: str
    source_label: str


class ModernizeRunRequest(BaseModel):
    """Request body for a modernization run."""

    datasets: list[str] | None = Field(
        None,
        description="Specific dataset key(s) to modernize. Omit (with `dataset`) for full batch.",
    )
    dataset: str | None = Field(
        None, description="Single dataset key or folder to modernize (merged with `datasets`)."
    )
    workers: int | None = Field(12, description="Transcoding worker threads")
    mds_shard_mb: int | None = Field(512, description="MDS shard size in MB")
    purge_loose: bool = Field(
        False,
        description=(
            "Delete source material (zip, legacy folder, loose images) after a successful"
            " conversion. False keeps every source."
        ),
    )
    push_kaggle: bool = Field(False, description="Push modernized package to Kaggle automatically")
    dry_run: bool = Field(
        False,
        description="Report the plan only: nothing is converted, deleted, written or uploaded",
    )
    force: bool = Field(False, description="Re-convert manifolds that are already containerized")

    def target_datasets(self) -> list[str]:
        """Return ``dataset`` plus ``datasets`` as one key list, de-duplicated in order."""
        keys = [self.dataset] if self.dataset else []
        keys.extend(self.datasets or [])
        return list(dict.fromkeys(k.strip() for k in keys if k and k.strip()))


class ManifoldReduceRequest(BaseModel):
    """Request body for creating a reduced manifold variant."""

    dataset_key: str | None = Field(
        None, description="Target manifold folder or dataset key to reduce"
    )
    manifold: str | None = Field(None, description="Alias of `dataset_key` (manifold folder name)")
    scale: float | None = Field(
        2,
        gt=0,
        description=(
            "Sampling scale: a fraction in (0, 1] keeps that share of samples (0.25 = 25%); "
            "a value above 1 is a downsampling factor (2 keeps 1/2)."
        ),
    )
    workers: int | None = Field(8, description="Worker threads count (not used by the reduce tool)")
    max_gb: float | None = Field(
        None, gt=0, description="Optional size cap in GB for the reduced manifold"
    )
    suffix: str | None = Field(
        None, description="Suffix for the reduced manifold folder (default: Reduced)"
    )
    dry_run: bool = Field(False, description="Report the plan only: nothing is written")


def build_modernize_command(req: ModernizeRunRequest) -> list[str]:
    """Build the modernization queue command; shares its argument contract with the CLI."""
    return [venv_python(), "-m", "tools.modernization_queue"] + build_queue_args(
        datasets=req.target_datasets(),
        workers=req.workers or 12,
        mds_shard_mb=req.mds_shard_mb or 512,
        purge_source=req.purge_loose,
        push_kaggle=req.push_kaggle,
        dry_run=req.dry_run,
        force=req.force,
    )


def scale_to_fraction(scale: float | None) -> float | None:
    """Map the API ``scale`` to the reduce tool's sampling fraction."""
    if scale is None:
        return None
    return scale if scale <= 1 else 1.0 / scale


def build_reduce_command(req: ManifoldReduceRequest) -> list[str]:
    """Build a non-interactive reduce command using only arguments the tool supports."""
    target = req.manifold or req.dataset_key
    if not target:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A manifold (`manifold` or `dataset_key`) is required for reduction.",
        )
    cmd = [venv_python(), "-m", "tools.manifold_reduce", "--reduce", "--yes", "--dataset", target]
    fraction = scale_to_fraction(req.scale)
    if fraction is not None:
        cmd.extend(["--fraction", str(fraction)])
    if req.max_gb is not None:
        cmd.extend(["--max-gb", str(req.max_gb)])
    if req.suffix:
        cmd.extend(["--suffix", req.suffix])
    if req.dry_run:
        cmd.append("--dry-run")
    return cmd


@router.get("/queue", response_model=list[QueueItemResponse])
async def get_modernization_queue() -> list[QueueItemResponse]:
    """Scan and retrieve real-time modernization status queue across all manifolds."""
    root = Path("../LemGendaryDatasets").resolve()
    unified_yaml = Path("unified_data.yaml").resolve()
    if not unified_yaml.exists():
        unified_yaml = Path(__file__).resolve().parent.parent.parent / "unified_data.yaml"

    items = load_queue_manifest(root, unified_yaml) if unified_yaml.exists() else []
    return [_queue_item_response(it) for it in items]


def _queue_item_response(it: ManifoldQueueItem) -> QueueItemResponse:
    """Map a scanned queue item to its API representation."""
    status_str, src_str = queue_status_labels(it, API_LABEL_STYLE)
    identity = {f.name: getattr(it, f.name) for f in fields(ManifoldRegistryEntry)}
    return QueueItemResponse(
        **identity,
        is_containerized=it.is_containerized,
        has_loose_files=it.has_loose_files,
        container_bytes=it.container_bytes,
        container_gb=round(it.container_bytes / (1024**3), 2),
        status_label=status_str,
        source_label=src_str,
    )


@router.post("/run", response_model=JobResponse)
async def run_modernization(req: ModernizeRunRequest) -> JobResponse:
    """Launch asynchronous modernization queue execution."""
    return submit_job(
        job_manager, JobType.MODERNIZE, build_modernize_command(req), req.model_dump()
    )


@router.post("/reduce", response_model=JobResponse)
async def run_reduce(req: ManifoldReduceRequest) -> JobResponse:
    """Launch asynchronous, non-interactive manifold scale reduction."""
    return submit_job(job_manager, JobType.REDUCE, build_reduce_command(req), req.model_dump())


@router.post("/regenerate-manifolds")
async def regenerate_manifolds_documentation() -> dict[str, Any]:
    """Regenerate authoritative manifolds.md matrix from on-disk status and registries."""
    code = DocService.rebuild_manifolds_md(check_only=False)
    if code != 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate manifolds.md",
        )
    return {
        "status": "success",
        "message": "Regenerated manifolds.md successfully.",
    }
