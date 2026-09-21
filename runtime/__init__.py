"""
LemGendary Dataset Compiler — Runtime environment bootstrap.

Phase 1.5.5 of the 2026 modernization roadmap. Extracted from the top of
compiler_core.py, which previously applied these side effects at import time.
"""
from __future__ import annotations

__all__ = ["bootstrap_runtime", "get_device_info"]

from .environment import bootstrap_runtime, get_device_info