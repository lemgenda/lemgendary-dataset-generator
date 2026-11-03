"""LemGendary Dataset Compiler Suite — REST and WebSocket API Service.

Exposes endpoints and streaming logs for LemGendary AI Studio GUI and CLI sidecars.
"""

from __future__ import annotations

from api.models import (
    CompileJobRequest,
    DatasetInfoResponse,
    DegradeJobRequest,
    HealthResponse,
    JobResponse,
    JobState,
)
from api.server import app, create_app, run_server

__all__ = [
    "CompileJobRequest",
    "DatasetInfoResponse",
    "DegradeJobRequest",
    "HealthResponse",
    "JobResponse",
    "JobState",
    "app",
    "create_app",
    "run_server",
]
