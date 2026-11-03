"""Test session setup: server state (token, PID file, jobs.db) goes to a temp dir.

Set before any test module imports ``api``, so the suite never reads or
writes the repository's ``.lgd_server`` folder (which a running server uses).
"""

from __future__ import annotations

import os
import shutil
import tempfile

_STATE_DIR = tempfile.mkdtemp(prefix="lgd_state_")
os.environ["LEMGENDARY_STATE_DIR"] = _STATE_DIR


def pytest_unconfigure() -> None:
    """Remove the temporary state dir after the session."""
    shutil.rmtree(_STATE_DIR)
