"""Kaggle dataset integration and sync endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from api.dataset_stats import as_count, read_yaml_mapping
from api.jobs import job_manager, submit_job
from api.models import (
    JobResponse,
    JobType,
    KaggleDatasetRegistryItem,
    KaggleDownloadRequest,
    KaggleUploadRequest,
)
from api.routes.datasets import _resolve_output_root
from core import paths
from core.common_sync import is_kaggle_large_manifold
from core.kaggle_identity import default_owner
from core.secrets import get_secret_record
from services.invocation import ServiceInvocation
from services.sync_service import kaggle_invocation, manifold_sync_invocation

logger = logging.getLogger("lemgendary.api.routes.kaggle")
router = APIRouter(prefix="/kaggle", tags=["Kaggle"])


class KaggleSyncRequest(BaseModel):
    """Request body for a Kaggle push or pull of one manifold."""

    manifold: str | None = Field(default=None, description="Specific manifold ID or folder name")
    push: bool = Field(default=False, description="Push local manifold to Kaggle")
    pull: bool = Field(default=False, description="Pull remote dataset from Kaggle")
    force: bool = Field(default=False, description="Force overwrite")


def _clean_kaggle_ref(ref: str) -> str:
    """Normalize Kaggle URLs and URIs to owner/dataset format."""
    cleaned = ref.strip()
    if "kaggle.com/datasets/" in cleaned:
        cleaned = cleaned.split("kaggle.com/datasets/")[-1].split("?")[0].strip("/")
    elif cleaned.startswith("kaggle://"):
        cleaned = cleaned.replace("kaggle://", "").strip("/")
    return cleaned


def _unified_yaml_path() -> Path:
    """unified_data.yaml at the repository root."""
    return paths.UNIFIED_DATA


def _unified_datasets() -> dict[str, Any]:
    """The ``datasets`` mapping of unified_data.yaml ({} when the file is absent)."""
    unified_yaml = _unified_yaml_path()
    if not unified_yaml.exists():
        return {}
    with open(unified_yaml, encoding="utf-8") as f:
        ydata = yaml.safe_load(f) or {}
    return ydata.get("datasets", {})


def _resolve_manifold_dir(root: Path, manifold: str) -> Path:
    """Local manifold folder for a folder name or registry key (modernized_folder)."""
    manifold_dir = root / manifold
    if manifold_dir.exists():
        return manifold_dir
    datasets = _unified_datasets()
    if manifold in datasets:
        mod_f = datasets[manifold].get("modernized_folder")
        if mod_f and (root / mod_f).exists():
            return root / mod_f
    return manifold_dir


def _registry_kaggle_ref(manifold: str, manifold_dir: Path, match_name: bool) -> str | None:
    """Kaggle ref of the registry entry matching the key, folder (or slug if ``match_name``)."""
    for k, v in _unified_datasets().items():
        if (
            k == manifold
            or v.get("modernized_folder") == manifold_dir.name
            or (match_name and v.get("name") == manifold_dir.name)
        ):
            return v.get("kaggle_ref")
    return None


def _start_sync_job(work: ServiceInvocation, parameters: dict[str, Any]) -> JobResponse:
    """Create and start a SYNC job running ``work``."""
    return submit_job(job_manager, JobType.SYNC, work, parameters)


def _local_sample_count(local_path: Path) -> int:
    """Sample count recorded in a manifold's dataset_info.yaml (0 when unknown)."""
    meta = read_yaml_mapping(local_path / "dataset_info.yaml")
    return as_count(meta.get("count") or meta.get("total_samples"))


def _registry_item(
    key: str, entry: dict[str, Any], root: Path, naming: tuple[str, str]
) -> KaggleDatasetRegistryItem:
    """Registry listing row for one unified_data.yaml entry with a kaggle_ref."""
    prefix, suffix = naming
    kaggle_ref = entry.get("kaggle_ref", "")
    slug = entry.get("name", "")
    mod_folder = entry.get("modernized_folder") or f"{prefix}{slug}{suffix}"
    local_path = root / mod_folder
    is_present = local_path.exists() and local_path.is_dir()
    return KaggleDatasetRegistryItem(
        key=key,
        title=entry.get("title", key),
        name=slug,
        modernized_folder=mod_folder,
        kaggle_ref=kaggle_ref,
        clean_repo_id=_clean_kaggle_ref(kaggle_ref),
        is_local_present=is_present,
        canonical_format=entry.get("canonical_format", "webdataset"),
        sample_count=_local_sample_count(local_path) if is_present else 0,
        size_gb=0.0,
    )


@router.get("/status")
async def get_kaggle_status() -> dict[str, Any]:
    """Inspect Kaggle authentication and local credential files."""
    has_env = bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )
    rec = get_secret_record("kaggle")
    has_vault = bool(rec and str(rec.get("secret_value") or "").strip())
    token_file = Path(".kaggle_token")
    has_token_file = token_file.exists() and len(token_file.read_text(encoding="utf-8").strip()) > 0
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_kaggle_json = kaggle_json.exists()

    authenticated = has_env or has_vault or has_token_file or has_kaggle_json

    return {
        "authenticated": authenticated,
        "auth_methods": {
            "environment_variables": has_env,
            "secrets_vault": has_vault,
            "dot_kaggle_token": has_token_file,
            "user_kaggle_json": has_kaggle_json,
        },
    }


@router.get("/registry-datasets", response_model=list[KaggleDatasetRegistryItem])
def list_registry_kaggle_datasets() -> list[KaggleDatasetRegistryItem]:
    """List the production manifolds in unified_data.yaml that can move through Kaggle."""
    root = _resolve_output_root()
    unified_yaml = _unified_yaml_path()

    items: list[KaggleDatasetRegistryItem] = []
    if not unified_yaml.exists():
        return items

    try:
        with open(unified_yaml, encoding="utf-8") as f:
            ydata = yaml.safe_load(f) or {}
            datasets_meta = ydata.get("datasets", {})
            meta = ydata.get("_registry_metadata", {})
            naming = (meta.get("name_prefix", "LemGendized"), meta.get("name_suffix", ""))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not parse %s for Kaggle registry listing: %s", unified_yaml, exc)
        return items

    for k, v in datasets_meta.items():
        if v.get("kaggle_ref", ""):
            items.append(_registry_item(k, v, root, naming))

    return items


@router.post("/download", response_model=JobResponse)
async def download_kaggle_dataset(req: KaggleDownloadRequest) -> JobResponse:
    """Download a manifold or custom dataset directly from Kaggle."""
    clean_id = _clean_kaggle_ref(req.kaggle_ref)
    if not clean_id or "/" not in clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid Kaggle dataset reference: '{req.kaggle_ref}'. "
                "Must be 'owner/dataset' or URL."
            ),
        )

    root = _resolve_output_root()
    target_name = req.target_folder
    if not target_name:
        slug = clean_id.split("/")[-1]
        target_name = slug if slug.startswith("LemGendized") else f"LemGendized{slug}"

    target_dir = root / target_name
    return _start_sync_job(
        kaggle_invocation(action="download", repo_id=clean_id, output_dir=str(target_dir)),
        {
            "action": "download",
            "repo_id": clean_id,
            "target_dir": str(target_dir),
            "force": req.force,
        },
    )


@router.post("/upload", response_model=JobResponse)
async def upload_kaggle_dataset(req: KaggleUploadRequest) -> JobResponse:
    """Zip and push a local compiled manifold to Kaggle."""
    root = _resolve_output_root()
    manifold_dir = _resolve_manifold_dir(root, req.manifold)
    if not manifold_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Local manifold '{req.manifold}' not found under {root}",
        )

    repo_id = (
        req.kaggle_ref
        or _registry_kaggle_ref(req.manifold, manifold_dir, match_name=True)
        or f"{default_owner()}/{manifold_dir.name.lower()}"
    )
    clean_id = _clean_kaggle_ref(repo_id)

    work = kaggle_invocation(
        action="upload", repo_id=clean_id, output_dir=str(manifold_dir), no_wait=req.no_wait
    )
    return _start_sync_job(
        work,
        {
            "action": "upload",
            "repo_id": clean_id,
            "output_dir": str(manifold_dir),
            "no_wait": req.no_wait,
        },
    )


@router.post("/sync", response_model=JobResponse)
async def trigger_kaggle_sync(req: KaggleSyncRequest) -> JobResponse:
    """Start a Kaggle push (sync) or pull (get) job through SyncService.sync_manifold."""
    action = "get" if req.pull and not req.push else "sync"
    return _start_sync_job(
        manifold_sync_invocation(action=action, model=req.manifold), req.model_dump()
    )


class KaggleMetadataRequest(BaseModel):
    """Request body for a Kaggle metadata update."""

    manifold: str | None = None
    """Specific manifold folder/key to update. If omitted, all_datasets must be True."""
    kaggle_ref: str | None = None
    """Explicit Kaggle handle (owner/slug). Auto-resolved from unified_data.yaml if omitted."""
    all_datasets: bool = False
    """If True, update metadata for every dataset in unified_data.yaml."""


@router.post("/update-metadata", response_model=JobResponse)
async def update_kaggle_metadata(req: KaggleMetadataRequest) -> JobResponse:
    """Push Kaggle metadata (title, description, license, columns) without re-uploading data."""
    root = _resolve_output_root()

    if req.all_datasets:
        # repo_id is required by the request type and ignored with all_datasets
        work = kaggle_invocation(
            action="metadata",
            repo_id=f"{default_owner()}/placeholder",
            output_dir=str(root),
            all_datasets=True,
        )
        return _start_sync_job(work, {"action": "metadata", "all_datasets": True})

    if not req.manifold:
        raise HTTPException(
            status_code=422, detail="Either 'manifold' or 'all_datasets: true' must be provided."
        )

    manifold_dir = _resolve_manifold_dir(root, req.manifold)
    repo_id = (
        req.kaggle_ref
        or _registry_kaggle_ref(req.manifold, manifold_dir, match_name=False)
        or f"{default_owner()}/{manifold_dir.name.lower()}"
    )
    clean_id = _clean_kaggle_ref(repo_id)
    if is_kaggle_large_manifold(clean_id):
        raise HTTPException(
            status_code=400,
            detail=f"Refusing to update metadata for legacy *large manifold: {clean_id}",
        )

    return _start_sync_job(
        kaggle_invocation(action="metadata", repo_id=clean_id, output_dir=str(manifold_dir)),
        {"action": "metadata", "repo_id": clean_id, "output_dir": str(manifold_dir)},
    )
