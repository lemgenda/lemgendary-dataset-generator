"""Persistent SQLite-backed job queue and asynchronous process executor."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.events import manager
from api.models import JobResponse, JobState, JobType

logger = logging.getLogger("lemgendary.api.jobs")

_DB_DIR = Path(".lgd_server")
_DB_PATH = _DB_DIR / "jobs.db"
_LOGS_DIR = _DB_DIR / "logs"


class JobManager:
    """Manages persistent SQLite job registry, background subprocess execution, and log streaming."""

    def __init__(self, db_path: Path = _DB_PATH, logs_dir: Path = _LOGS_DIR) -> None:
        self.db_path = db_path
        self.logs_dir = logs_dir
        self.active_processes: Dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        self._init_storage()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

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
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
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
                logger.info("Marked %d orphaned jobs as interrupted following server reboot", cursor.rowcount)
            conn.commit()

    def create_job(
        self,
        job_type: JobType,
        command: List[str],
        parameters: Dict[str, Any],
    ) -> JobResponse:
        """Register a new job in pending state and return the job descriptor."""
        job_id = str(uuid.uuid4())
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_file_rel = str(self.logs_dir / f"{job_id}.log")

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
        return self._build_response(
            job_id=job_id,
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

    def _execute_job_thread(self, job_id: str, loop: asyncio.AbstractEventLoop) -> None:
        """Worker thread executing the subprocess, writing disk logs, and streaming lines."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT command, parameters, log_file, job_type FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                logger.error("Job %s not found for execution", job_id)
                return

            cmd: List[str] = json.loads(row["command"])
            log_file = Path(row["log_file"])
            started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

            conn.execute("UPDATE jobs SET state = ?, started_at = ? WHERE id = ?", (JobState.RUNNING.value, started_at, job_id))
            conn.commit()

        logger.info("Executing job %s: %s", job_id, " ".join(cmd))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            with open(log_file, "w", encoding="utf-8", buffering=1) as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                with self._lock:
                    self.active_processes[job_id] = proc

                assert proc.stdout is not None
                for line in proc.stdout:
                    lf.write(line)
                    # Broadcast to active WebSocket subscribers asynchronously
                    asyncio.run_coroutine_threadsafe(
                        manager.broadcast_job_log(job_id, line),
                        loop,
                    )

                proc.wait()
                exit_code = proc.returncode

        except Exception as exc:
            logger.exception("Unexpected error executing job %s: %s", job_id, exc)
            exit_code = 1
            error_message = str(exc)
        else:
            error_message = None if exit_code == 0 else f"Process exited with non-zero status code: {exit_code}"
        finally:
            with self._lock:
                if job_id in self.active_processes:
                    del self.active_processes[job_id]

        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        final_state = JobState.COMPLETED if exit_code == 0 else JobState.FAILED

        # Check if process was actively cancelled
        with self._get_connection() as conn:
            current_state = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
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

        logger.info("Job %s finalized with state: %s (exit code: %s)", job_id, final_state.value, exit_code)
        # Notify WebSocket subscribers of completion
        terminal_event = f"\n[PROCESS_TERMINATED] Job {job_id} {final_state.value.upper()} (Exit Code: {exit_code})\n"
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_job_log(job_id, terminal_event),
            loop,
        )

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
                (JobState.CANCELLED.value, "Process cancelled by user request.", job_id, JobState.PENDING.value, JobState.RUNNING.value),
            )
            conn.commit()
            return conn.total_changes > 0

    def get_job(self, job_id: str) -> Optional[JobResponse]:
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
        state: Optional[JobState] = None,
        job_type: Optional[JobType] = None,
    ) -> List[JobResponse]:
        """Fetch paginated job records with optional filtering."""
        query = "SELECT * FROM jobs WHERE 1=1"
        params: List[Any] = []

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

    def get_log_content(self, job_id: str, tail_lines: Optional[int] = None) -> str:
        """Read log file content from disk."""
        log_file = self.logs_dir / f"{job_id}.log"
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

    @staticmethod
    def _build_response(
        job_id: str,
        job_type: JobType,
        state: JobState,
        created_at: str,
        started_at: Optional[str],
        completed_at: Optional[str],
        exit_code: Optional[int],
        error_message: Optional[str],
        log_file: str,
        parameters: Dict[str, Any],
    ) -> JobResponse:
        return JobResponse(
            id=job_id,
            job_type=job_type,
            state=state,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            exit_code=exit_code,
            error_message=error_message,
            log_file=log_file,
            parameters=parameters,
        )

    def _row_to_response(self, row: sqlite3.Row) -> JobResponse:
        params: Dict[str, Any] = {}
        try:
            params = json.loads(row["parameters"])
        except Exception:
            params = {}

        return self._build_response(
            job_id=row["id"],
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


job_manager = JobManager()
