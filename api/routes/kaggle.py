"""Kaggle dataset integration and sync endpoints."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.jobs import job_manager
from api.models import (
    JobResponse,
    JobType,
    KaggleDatasetRegistryItem,
    KaggleDownloadRequest,
    KaggleUploadRequest,
)
from api.routes.datasets import _resolve_output_root
from core.cli_args import venv_python
import yaml

router = APIRouter(prefix="/kaggle", tags=["Kaggle"])


class KaggleSyncRequest(BaseModel):
    manifold: Optional[str] = Field(None, description="Specific manifold ID or folder name")
    push: bool = Field(False, description="Push local manifold to Kaggle")
    pull: bool = Field(False, description="Pull remote dataset from Kaggle")
    force: bool = Field(False, description="Force overwrite")


def _clean_kaggle_ref(ref: str) -> str:
    """Normalize Kaggle URLs and URIs to owner/dataset format."""
    cleaned = ref.strip()
    if "kaggle.com/datasets/" in cleaned:
        cleaned = cleaned.split("kaggle.com/datasets/")[-1].split("?")[0].strip("/")
    elif cleaned.startswith("kaggle://"):
        cleaned = cleaned.replace("kaggle://", "").strip("/")
    return cleaned


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


@router.get("/registry-datasets", response_model=List[KaggleDatasetRegistryItem])
async def list_registry_kaggle_datasets() -> List[KaggleDatasetRegistryItem]:
    """List all production manifolds from unified_data.yaml that can be downloaded/uploaded via Kaggle."""
    root = _resolve_output_root()
    unified_yaml = Path("unified_data.yaml")
    if not unified_yaml.exists():
        unified_yaml = Path(__file__).resolve().parent.parent.parent / "unified_data.yaml"

    items: List[KaggleDatasetRegistryItem] = []
    if not unified_yaml.exists():
        return items

    try:
        with open(unified_yaml, "r", encoding="utf-8") as f:
            ydata = yaml.safe_load(f) or {}
            datasets_meta = ydata.get("datasets", {})
            meta = ydata.get("_registry_metadata", {})
            prefix = meta.get("name_prefix", "LemGendized")
            suffix = meta.get("name_suffix", "")
    except Exception:
        return items

    for k, v in datasets_meta.items():
        kaggle_ref = v.get("kaggle_ref", "")
        if not kaggle_ref:
            continue

        clean_id = _clean_kaggle_ref(kaggle_ref)
        slug = v.get("name", "")
        mod_folder = v.get("modernized_folder") or f"{prefix}{slug}{suffix}"
        local_path = root / mod_folder
        is_present = local_path.exists() and local_path.is_dir()

        sample_count = 0
        size_gb = 0.0
        if is_present:
            info_file = local_path / "dataset_info.yaml"
            if info_file.exists():
                try:
                    with open(info_file, "r", encoding="utf-8") as inf:
                        m = yaml.safe_load(inf) or {}
                        sample_count = m.get("count") or m.get("total_samples") or 0
                except Exception:
                    pass

        items.append(
            KaggleDatasetRegistryItem(
                key=k,
                title=v.get("title", k),
                name=slug,
                modernized_folder=mod_folder,
                kaggle_ref=kaggle_ref,
                clean_repo_id=clean_id,
                is_local_present=is_present,
                canonical_format=v.get("canonical_format", "webdataset"),
                sample_count=sample_count,
                size_gb=round(size_gb, 2),
            )
        )

    return items


@router.post("/download", response_model=JobResponse)
async def download_kaggle_dataset(req: KaggleDownloadRequest) -> JobResponse:
    """Download a manifold or custom dataset directly from Kaggle."""
    clean_id = _clean_kaggle_ref(req.kaggle_ref)
    if not clean_id or "/" not in clean_id:
        raise ValueError(f"Invalid Kaggle dataset reference: '{req.kaggle_ref}'. Must be 'owner/dataset' or URL.")

    root = _resolve_output_root()
    target_name = req.target_folder
    if not target_name:
        slug = clean_id.split("/")[-1]
        target_name = slug if slug.startswith("LemGendized") else f"LemGendized{slug}"

    target_dir = root / target_name
    cmd = [
        venv_python(),
        "sources/kaggle.py",
        "--action", "download",
        "--repo_id", clean_id,
        "--output_dir", str(target_dir),
    ]

    job = job_manager.create_job(
        job_type=JobType.SYNC,
        command=cmd,
        parameters={
            "action": "download",
            "repo_id": clean_id,
            "target_dir": str(target_dir),
            "force": req.force,
        },
    )
    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    return job


@router.post("/upload", response_model=JobResponse)
async def upload_kaggle_dataset(req: KaggleUploadRequest) -> JobResponse:
    """Zip and push a local compiled manifold to Kaggle."""
    root = _resolve_output_root()
    manifold_dir = root / req.manifold
    if not manifold_dir.exists():
        # Try checking if manifold is a key
        unified_yaml = Path("unified_data.yaml")
        if not unified_yaml.exists():
            unified_yaml = Path(__file__).resolve().parent.parent.parent / "unified_data.yaml"
        if unified_yaml.exists():
            with open(unified_yaml, "r", encoding="utf-8") as f:
                ydata = yaml.safe_load(f) or {}
                if req.manifold in ydata.get("datasets", {}):
                    mod_f = ydata["datasets"][req.manifold].get("modernized_folder")
                    if mod_f and (root / mod_f).exists():
                        manifold_dir = root / mod_f

    if not manifold_dir.exists():
        raise ValueError(f"Local manifold '{req.manifold}' not found under {root}")

    repo_id = req.kaggle_ref
    if not repo_id:
        # Lookup in unified_data.yaml
        unified_yaml = Path("unified_data.yaml")
        if not unified_yaml.exists():
            unified_yaml = Path(__file__).resolve().parent.parent.parent / "unified_data.yaml"
        if unified_yaml.exists():
            with open(unified_yaml, "r", encoding="utf-8") as f:
                ydata = yaml.safe_load(f) or {}
                for k, v in ydata.get("datasets", {}).items():
                    if k == req.manifold or v.get("modernized_folder") == manifold_dir.name or v.get("name") == manifold_dir.name:
                        repo_id = v.get("kaggle_ref")
                        break

    if not repo_id:
        repo_id = f"lemtreursi/{manifold_dir.name.lower()}"

    clean_id = _clean_kaggle_ref(repo_id)

    cmd = [
        venv_python(),
        "sources/kaggle.py",
        "--action", "upload",
        "--repo_id", clean_id,
        "--output_dir", str(manifold_dir),
    ]
    if req.no_wait:
        cmd.append("--no-wait")

    job = job_manager.create_job(
        job_type=JobType.SYNC,
        command=cmd,
        parameters={
            "action": "upload",
            "repo_id": clean_id,
            "output_dir": str(manifold_dir),
            "no_wait": req.no_wait,
        },
    )
    loop = asyncio.get_running_loop()
    job_manager.start_job(job.id, loop)
    return job


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
