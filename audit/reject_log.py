"""
Reject log writer.

Single responsibility: record every per-sample rejection into the manifold
registry's `reject_log` table (created by Phase 1.3's registry promotion).

Writes are buffered. The buffer is flushed when it reaches `buffer_limit`
entries or when `flush()` is called explicitly. A per-thread connection is
used so this class is safe under ThreadPoolExecutor; under ProcessPoolExecutor
each worker process gets its own connection, which SQLite WAL handles fine.

Phase 2 of the 2026 modernization roadmap.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class RejectLog:
    """Buffered reject-log writer bound to a registry database."""

    def __init__(self, db_path: str | Path, buffer_limit: int = 500) -> None:
        self.db_path = str(db_path)
        self.buffer_limit = buffer_limit
        self._buffer: list[tuple[str, str, str, str]] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    # ── Connection management ───────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=60.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # ── Public API ──────────────────────────────────────────────────────────
    def record(
        self,
        source: str,
        sample_name: str,
        reject_code: str,
        reason: str = "",
    ) -> None:
        """Append one rejection. Flushes when the buffer fills."""
        with self._lock:
            self._buffer.append((source, sample_name, reject_code, reason))
            if len(self._buffer) >= self.buffer_limit:
                self._flush_locked()

    def flush(self) -> None:
        """Force-flush any buffered entries. Safe to call at any time."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush and drop the thread-local connection."""
        self.flush()
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._local.conn = None

    # ── Internal ────────────────────────────────────────────────────────────
    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        conn = self._conn()
        conn.executemany(
            "INSERT INTO reject_log (source, sample_name, reject_code, reason) "
            "VALUES (?, ?, ?, ?)",
            self._buffer,
        )
        conn.commit()
        self._buffer.clear()