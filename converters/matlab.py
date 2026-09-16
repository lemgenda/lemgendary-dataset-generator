"""MATLAB .mat file loader. Extracted from compiler_core.py in Phase 1.4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import scipy.io as sio


def parse_matlab(mat_path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a .mat file and return (data_dict, primary_key).

    `primary_key` is the first non-dunder key in the loaded structure. This
    is a heuristic — MATLAB files with multiple meaningful keys will only
    have their first one used by callers.

    ``spmatrix=False`` is passed explicitly to select the non-deprecated
    overload in scipy-stubs. scipy deprecates the ``spmatrix=True`` default
    in 1.14 and it will be removed in a future release; the compiler never
    consumes MATLAB sparse matrices, so ``False`` is the semantically
    correct value here as well.
    """
    data: dict[str, Any] = sio.loadmat(str(mat_path), spmatrix=False)
    key = [k for k in data if not k.startswith("__")][0]
    return data, key