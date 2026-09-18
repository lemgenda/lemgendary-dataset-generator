"""Health and hardware diagnostic routes."""

from __future__ import annotations

import os
import time
from typing import Any
from fastapi import APIRouter

import psutil
import torch

from api.jobs import job_manager
from api.models import FullHealthResponse, HardwareInfo, HealthResponse
from core.cli_args import __version__

router = APIRouter(prefix="/health", tags=["Health"])

_START_TIME = time.time()


def get_hardware_info() -> HardwareInfo:
    """Probe local system CPU, memory, and CUDA acceleration hardware."""
    vm = psutil.virtual_memory()
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    cpu_count = os.cpu_count() or 1

    return HardwareInfo(
        cpu_count=cpu_count,
        ram_total_gb=round(vm.total / (1024**3), 2),
        ram_available_gb=round(vm.available / (1024**3), 2),
        cuda_available=cuda_avail,
        device_name=device_name,
        active_workers=job_manager.count_active_jobs(),
    )


@router.get("", response_model=HealthResponse)
async def check_health() -> HealthResponse:
    """Basic health and liveness endpoint."""
    uptime = round(time.time() - _START_TIME, 2)
    return HealthResponse(
        status="ok",
        service="LemGendary Dataset Compiler API",
        version=__version__,
        uptime_seconds=uptime,
        active_jobs=job_manager.count_active_jobs(),
    )


@router.get("/full", response_model=FullHealthResponse)
async def check_health_full() -> FullHealthResponse:
    """Detailed health probe including hardware sensors and resource utilization."""
    uptime = round(time.time() - _START_TIME, 2)
    hardware = get_hardware_info()
    return FullHealthResponse(
        status="ok",
        service="LemGendary Dataset Compiler API",
        version=__version__,
        uptime_seconds=uptime,
        active_jobs=job_manager.count_active_jobs(),
        hardware=hardware,
    )
