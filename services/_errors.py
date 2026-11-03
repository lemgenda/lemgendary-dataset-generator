"""Shared exception tuple for service-facade CLI boundaries.

Service methods translate *expected* pipeline failures (I/O, bad data or
configuration, missing optional dependencies, failed subprocesses, registry DB
errors) into a non-zero exit code. Anything outside this set is a programming
error and is allowed to propagate with its traceback.
"""

from __future__ import annotations

import sqlite3
import subprocess

import yaml

SERVICE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    LookupError,
    RuntimeError,
    ImportError,
    subprocess.SubprocessError,
    sqlite3.Error,
    yaml.YAMLError,
)
