"""Project paths, anchored on the repository root (never the working directory).

Every route, command and compiler stage resolves the registry, raw sources,
compiled output and server state from here, so a CLI or server started from
another directory sees the same files (token, PID file, job database).
``LEMGENDARY_STATE_DIR`` relocates the server state (tests use a temp dir).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIFIED_DATA = PROJECT_ROOT / "unified_data.yaml"
PRESETS_FILE = PROJECT_ROOT / "presets.yaml"
CONFIG_JSON = PROJECT_ROOT / "config.json"
RAW_SETS = PROJECT_ROOT / "raw-sets"

# Compiled manifolds when the registry names no ``output_folder_name``.
DEFAULT_OUTPUT_FOLDER = "../LemGendaryDatasets"


def state_dir() -> Path:
    """Server state folder (token, PID file, jobs.db, job logs)."""
    override = os.environ.get("LEMGENDARY_STATE_DIR")
    return Path(override) if override else PROJECT_ROOT / ".lgd_server"


def token_file() -> Path:
    """The API token shared by the server and the CLI."""
    return state_dir() / "token"


def pid_file() -> Path:
    """The running server's PID file."""
    return state_dir() / "server.pid"


def from_root(path: str | Path) -> Path:
    """``path`` resolved against the project root when it is relative."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def output_root(output_folder_name: str | None) -> Path:
    """Compiled manifold folder: the registry's ``output_folder_name`` from the root."""
    return from_root(output_folder_name or DEFAULT_OUTPUT_FOLDER)
