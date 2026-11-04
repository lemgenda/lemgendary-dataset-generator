"""Job runner plumbing: child-process environment and live log / event broadcasting.

Used by :class:`api.jobs.JobManager` worker threads to forward subprocess output
to per-job WebSocket subscribers and the global dashboard stream.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from api.events import manager
from api.models import JobState

__all__ = [
    "ClaimedJob",
    "broadcast_global",
    "broadcast_start",
    "broadcast_termination",
    "creation_flags",
    "stream_line",
    "subprocess_env",
    "utc_now_iso",
]

_PROGRESS_MARKERS = (
    "ARCHIVING:",
    "Uploading:",
    "EXTRACTING",
    "UNPACKING",
    "KAGGLE EXTRACTION",
    "it/s",
    "B/s",
)
_PROGRESS_BAR_RE = re.compile(r"(\|[ █━─\-=#]+?\||\[[ █━─\-=#]+?\])")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


@dataclass(frozen=True)
class ClaimedJob:
    """A job row just switched to RUNNING, with what its runner thread needs."""

    job_id: str
    command: list[str]
    log_file: Path
    started_at: str
    job_type: str


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def subprocess_env() -> dict[str, str]:
    """Child environment forcing unbuffered UTF-8 Python output."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def creation_flags() -> int:
    """Hide the console window of job processes on Windows."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return 0


def broadcast_global(loop: asyncio.AbstractEventLoop, payload: dict[str, Any]) -> None:
    """Schedule ``payload`` on the global dashboard stream from a worker thread."""
    asyncio.run_coroutine_threadsafe(manager.broadcast_global(payload), loop)


def _is_progress_line(line: str, clean_line: str) -> bool:
    return (
        "\r" in line
        or ("%" in clean_line and ("|" in clean_line or "/" in clean_line))
        or any(marker in clean_line for marker in _PROGRESS_MARKERS)
    )


def stream_line(loop: asyncio.AbstractEventLoop, job_id: str, line: str) -> None:
    """Forward one output line to the job's WebSocket subscribers and the dashboard stream."""
    # Broadcast to active WebSocket subscribers asynchronously
    asyncio.run_coroutine_threadsafe(manager.broadcast_job_log(job_id, line), loop)
    # Broadcast to global dashboard stream
    clean_line = line.rstrip("\r\n")
    if not clean_line.strip():
        return

    is_progress = _is_progress_line(line, clean_line)
    # Strip ASCII progress bars; the GUI progress tracker renders the bar
    if is_progress:
        clean_line = _PROGRESS_BAR_RE.sub("", clean_line)
        clean_line = _MULTI_SPACE_RE.sub(" ", clean_line).strip()

    broadcast_global(
        loop,
        {
            "job_id": job_id,
            "type": "log",
            "step_name": "Compiler",
            "status": "info",
            "chunk": line,
            "message": clean_line,
            "is_progress": is_progress,
            "timestamp": utc_now_iso(),
        },
    )


def _event_payload(job_id: str, status: str, message: str, timestamp: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "type": "event",
        "step_name": "Compiler",
        "status": status,
        "message": message,
        "timestamp": timestamp,
    }


def broadcast_start(loop: asyncio.AbstractEventLoop, claimed: ClaimedJob) -> None:
    """Announce a job launch on the global dashboard stream."""
    message = (
        f"[JOB_START] Job {claimed.job_id} ({claimed.job_type}) started:"
        f" {' '.join(claimed.command)}"
    )
    broadcast_global(loop, _event_payload(claimed.job_id, "info", message, claimed.started_at))


def broadcast_termination(
    loop: asyncio.AbstractEventLoop,
    claimed: ClaimedJob,
    final_state: JobState,
    exit_code: int,
    completed_at: str,
) -> None:
    """Notify WebSocket subscribers and the dashboard stream that a job finished."""
    job_id = claimed.job_id
    terminal_event = (
        f"\n[PROCESS_TERMINATED] Job {job_id} {final_state.value.upper()} (Exit Code:"
        f" {exit_code})\n"
    )
    asyncio.run_coroutine_threadsafe(manager.broadcast_job_log(job_id, terminal_event), loop)
    status = "success" if final_state == JobState.COMPLETED else "error"
    message = (
        f"[JOB_{final_state.value.upper()}] Job {job_id} ({claimed.job_type}) finished with"
        f" exit code {exit_code}"
    )
    broadcast_global(loop, _event_payload(job_id, status, message, completed_at))
