"""Persistent SQLite-backed job queue and asynchronous process executor.

The job manager is the one place in the API that spawns processes. Routes
describe service work as a :class:`services.invocation.ServiceInvocation`;
:func:`job_command` turns it into the argv of this project's interpreter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import subprocess
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from api.job_stream import (
    ClaimedJob,
    broadcast_start,
    broadcast_termination,
    creation_flags,
    stream_line,
    subprocess_env,
    utc_now_iso,
)
from api.models import JobResponse, JobState, JobType
from core import paths
from core.cli_args import venv_python
from services.invocation import ServiceInvocation

logger = logging.getLogger("lemgendary.api.jobs")

_DB_DIR = paths.state_dir()
_DB_PATH = _DB_DIR / "jobs.db"
_LOGS_DIR = _DB_DIR / "logs"


class JobManager:
    """Persistent SQLite job registry, background subprocess execution and log streaming."""

    def __init__(self, db_path: Path = _DB_PATH, logs_dir: Path = _LOGS_DIR) -> None:
        self.db_path = db_path
        self.logs_dir = logs_dir
        self.active_processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        self._init_storage()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success, rolls back on error and is always closed."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _log_path(self, stored: str) -> Path:
        """Resolve a stored log path (relative to the state dir; old rows are absolute)."""
        path = Path(stored)
        return path if path.is_absolute() else self.db_path.parent / path

    def _stored_log_path(self, job_id: str) -> str:
        """The log path recorded for a job: relative to the state dir when possible."""
        log_file = self.logs_dir / f"{job_id}.log"
        try:
            return log_file.relative_to(self.db_path.parent).as_posix()
        except ValueError:
            return str(log_file)

    def _init_storage(self) -> None:
        """Create database tables, logs directory, and perform restart interruption recovery."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    command TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    exit_code INTEGER,
                    error_message TEXT,
                    log_file TEXT NOT NULL
                )
                """
            )
            # Restart recovery: Mark any leftover pending/running jobs as interrupted
            now_iso = utc_now_iso()
            cursor = conn.execute(
                """
                UPDATE jobs
                SET state = ?, completed_at = ?, error_message = ?
                WHERE state IN (?, ?)
                """,
                (
                    JobState.INTERRUPTED.value,
                    now_iso,
                    "Process interrupted by server restart. Resubmission required.",
                    JobState.PENDING.value,
                    JobState.RUNNING.value,
                ),
            )
            if cursor.rowcount > 0:
                logger.info(
                    "Marked %d orphaned jobs as interrupted following server reboot",
                    cursor.rowcount,
                )
            conn.commit()

    def create_job(
        self,
        job_type: JobType,
        command: list[str],
        parameters: dict[str, Any],
    ) -> JobResponse:
        """Register a new job in pending state and return the job descriptor."""
        job_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        log_file_rel = self._stored_log_path(job_id)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, job_type, state, command, parameters, created_at,
                    started_at, completed_at, exit_code, error_message, log_file
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)
                """,
                (
                    job_id,
                    job_type.value,
                    JobState.PENDING.value,
                    json.dumps(command),
                    json.dumps(parameters),
                    created_at,
                    log_file_rel,
                ),
            )
            conn.commit()

        logger.info("Created job %s (%s)", job_id, job_type.value)
        return JobResponse(
            id=job_id,
            job_type=job_type,
            state=JobState.PENDING,
            created_at=created_at,
            started_at=None,
            completed_at=None,
            exit_code=None,
            error_message=None,
            log_file=log_file_rel,
            parameters=parameters,
        )

    def start_job(self, job_id: str, loop: asyncio.AbstractEventLoop) -> None:
        """Spawn the background execution thread for a registered pending job."""
        thread = threading.Thread(
            target=self._execute_job_thread,
            args=(job_id, loop),
            name=f"job-runner-{job_id}",
            daemon=True,
        )
        thread.start()

    def _claim_job(self, job_id: str) -> ClaimedJob | None:
        """Mark a registered job RUNNING and return what is needed to execute it."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT command, parameters, log_file, job_type FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not row:
                logger.error("Job %s not found for execution", job_id)
                return None

            claimed = ClaimedJob(
                job_id=job_id,
                command=json.loads(row["command"]),
                log_file=self._log_path(row["log_file"]),
                started_at=utc_now_iso(),
                job_type=row["job_type"],
            )
            # Only a still-pending job is claimed: one cancelled meanwhile stays cancelled.
            updated = conn.execute(
                "UPDATE jobs SET state = ?, started_at = ? WHERE id = ? AND state = ?",
                (JobState.RUNNING.value, claimed.started_at, job_id, JobState.PENDING.value),
            ).rowcount
        if not updated:
            logger.info("Job %s is no longer pending (cancelled before start); not running it",
                        job_id)
            return None
        return claimed

    def _execute_job_thread(self, job_id: str, loop: asyncio.AbstractEventLoop) -> None:
        """Worker thread executing the subprocess, writing disk logs, and streaming lines."""
        claimed = self._claim_job(job_id)
        if claimed is None:
            return
        cmd = claimed.command

        logger.info("Executing job %s: %s", job_id, " ".join(cmd))
        claimed.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Broadcast job launch to global dashboard stream
        broadcast_start(loop, claimed)

        exit_code, error_message = self._run_process(job_id, claimed, loop)
        final_state, completed_at = self._finalize_job(job_id, exit_code, error_message)
        broadcast_termination(loop, claimed, final_state, exit_code, completed_at)

    def _run_process(
        self, job_id: str, claimed: ClaimedJob, loop: asyncio.AbstractEventLoop
    ) -> tuple[int, str | None]:
        """Run the job command, tee its output to the log file and the WebSocket streams."""
        try:
            with (
                open(claimed.log_file, "w", encoding="utf-8", buffering=1) as lf,
                subprocess.Popen(
                    claimed.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=subprocess_env(),
                    cwd=str(paths.PROJECT_ROOT),
                    creationflags=creation_flags(),
                ) as proc,
            ):
                with self._lock:
                    self.active_processes[job_id] = proc

                assert proc.stdout is not None
                for line in proc.stdout:
                    lf.write(line)
                    stream_line(loop, job_id, line)

                proc.wait()
                exit_code = proc.returncode

        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            # Background job boundary: log-file I/O, process spawn (OSError /
            # SubprocessError), stdout decoding (ValueError) and broadcasting on a
            # closed event loop (RuntimeError) become a FAILED job status instead
            # of killing the runner thread and leaving the job stuck in RUNNING.
            logger.exception("Unexpected error executing job %s", job_id)
            return 1, str(exc)
        else:
            error_message = (
                None if exit_code == 0 else f"Process exited with non-zero status code: {exit_code}"
            )
            return exit_code, error_message
        finally:
            with self._lock:
                if job_id in self.active_processes:
                    del self.active_processes[job_id]

    def _finalize_job(
        self, job_id: str, exit_code: int, error_message: str | None
    ) -> tuple[JobState, str]:
        """Persist the terminal state (keeping CANCELLED) and return it with its timestamp."""
        completed_at = utc_now_iso()
        final_state = JobState.COMPLETED if exit_code == 0 else JobState.FAILED

        # Check if process was actively cancelled
        with self._get_connection() as conn:
            current_state = conn.execute(
                "SELECT state FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if current_state and current_state["state"] == JobState.CANCELLED.value:
                final_state = JobState.CANCELLED

            conn.execute(
                """
                UPDATE jobs
                SET state = ?, completed_at = ?, exit_code = ?, error_message = ?
                WHERE id = ?
                """,
                (final_state.value, completed_at, exit_code, error_message, job_id),
            )
            conn.commit()

        logger.info(
            "Job %s finalized with state: %s (exit code: %s)", job_id, final_state.value, exit_code
        )
        return final_state, completed_at

    def cancel_job(self, job_id: str) -> bool:
        """Terminate a running process and update state to cancelled."""
        with self._lock:
            proc = self.active_processes.get(job_id)
            if proc:
                try:
                    proc.terminate()
                    logger.info("Terminated active process for job %s", job_id)
                except OSError as exc:
                    logger.warning("Failed to terminate process for job %s: %s", job_id, exc)

        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET state = ?, error_message = ? WHERE id = ? AND state IN (?, ?)",
                (
                    JobState.CANCELLED.value,
                    "Process cancelled by user request.",
                    job_id,
                    JobState.PENDING.value,
                    JobState.RUNNING.value,
                ),
            )
            conn.commit()
            return conn.total_changes > 0

    def get_job(self, job_id: str) -> JobResponse | None:
        """Fetch single job record by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return self._row_to_response(row)

    def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        state: JobState | None = None,
        job_type: JobType | None = None,
    ) -> list[JobResponse]:
        """Fetch paginated job records with optional filtering."""
        query = "SELECT * FROM jobs WHERE 1=1"
        params: list[Any] = []

        if state:
            query += " AND state = ?"
            params.append(state.value)
        if job_type:
            query += " AND job_type = ?"
            params.append(job_type.value)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_response(r) for r in rows]

    def count_active_jobs(self) -> int:
        """Count jobs currently in running or pending state."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM jobs WHERE state IN (?, ?)",
                (JobState.PENDING.value, JobState.RUNNING.value),
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def count_total_jobs(self) -> int:
        """Count all historical jobs recorded in SQLite."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM jobs").fetchone()
            return int(row["cnt"]) if row else 0

    def get_log_content(self, job_id: str, tail_lines: int | None = None) -> str:
        """Read log file content from disk."""
        log_file = self._log_path(self._stored_log_path(job_id))
        if not log_file.exists():
            return ""
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            if tail_lines is not None:
                lines = text.splitlines()
                return "\n".join(lines[-tail_lines:])
            return text
        except OSError as exc:
            logger.warning("Failed reading log file %s: %s", log_file, exc)
            return ""

    def _row_to_response(self, row: sqlite3.Row) -> JobResponse:
        params: dict[str, Any] = {}
        try:
            params = json.loads(row["parameters"])
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Job %s has unparseable parameters JSON; using empty parameters: %s", row["id"], exc
            )
            params = {}

        return JobResponse(
            id=row["id"],
            job_type=JobType(row["job_type"]),
            state=JobState(row["state"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            exit_code=row["exit_code"],
            error_message=row["error_message"],
            log_file=row["log_file"],
            parameters=params,
        )


def job_command(work: ServiceInvocation) -> list[str]:
    """The argv a job runs: a service entry point on the project interpreter."""
    return work.argv(venv_python())


def submit_job(
    jobs: JobManager,
    job_type: JobType,
    work: ServiceInvocation,
    parameters: dict[str, Any],
) -> JobResponse:
    """Register a job on ``jobs`` and start it on the running event loop.

    Must be called from a coroutine (route handler); returns the pending job.
    """
    job = jobs.create_job(job_type=job_type, command=job_command(work), parameters=parameters)
    jobs.start_job(job.id, asyncio.get_running_loop())
    return job


job_manager = JobManager()
