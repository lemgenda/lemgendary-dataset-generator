"""MATLAB .mat file loader. Copied verbatim from compiler_core.py in Phase 1.4."""

from __future__ import annotations


def parse_matlab(mat_path):
    import scipy.io as sio  # type: ignore[import-untyped]
    data = sio.loadmat(mat_path)
    # Heuristic for finding the annotation key
    key = [k for k in data if not k.startswith("__")][0]
    return data, key