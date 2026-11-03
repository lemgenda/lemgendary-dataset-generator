"""LemGendary Dataset Compiler Suite — REST and WebSocket API Service.

Exposes endpoints and streaming logs for LemGendary AI Studio GUI and CLI sidecars.

Importing the package (or a light submodule such as ``api.auth`` from the
CLI) does not build the app: the server, its routers and the job manager
(which opens ``.lgd_server/jobs.db``) load only from ``api.server``.
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

__all__ = [
    "CompileJobRequest",
    "DatasetInfoResponse",
    "DegradeJobRequest",
    "HealthResponse",
    "JobResponse",
    "JobState",
]
