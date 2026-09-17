"""Environment Manager passthrough routes delegating to lem-env."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status

from api.models import EnvPassthroughResponse
from cli_args import PROJECT_NAME, resolve_lem_env

router = APIRouter(prefix="/env", tags=["Environment Manager"])


@router.get("/status", response_model=EnvPassthroughResponse)
async def get_env_status() -> EnvPassthroughResponse:
    """Delegate ecosystem health audit to lem-env audit --fast."""
    try:
        lem_env = resolve_lem_env()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [lem_env, "audit", "--fast"],
            capture_output=True,
            text=True,
            check=False,
        ),
    )

    return EnvPassthroughResponse(
        status="success" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        output=result.stdout + (result.stderr or ""),
    )


@router.get("/validate", response_model=EnvPassthroughResponse)
@router.post("/validate", response_model=EnvPassthroughResponse)
async def validate_codebase() -> EnvPassthroughResponse:
    """Delegate full code validation to lem-env validate --project lemgendary-datasets."""
    try:
        lem_env = resolve_lem_env()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [lem_env, "validate", "--project", PROJECT_NAME],
            capture_output=True,
            text=True,
            check=False,
        ),
    )

    return EnvPassthroughResponse(
        status="success" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        output=result.stdout + (result.stderr or ""),
    )
