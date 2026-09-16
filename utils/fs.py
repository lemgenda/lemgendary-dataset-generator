"""Filesystem helpers. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

import os
from pathlib import Path


def get_dir_size(path) -> float:
    """Calculate recursive directory size in GB."""
    def _get_bytes(p):
        total = 0
        try:
            for entry in os.scandir(p):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += _get_bytes(entry.path)
        except (PermissionError, OSError):
            pass
        return total
    return _get_bytes(path) / (1024 ** 3)


def remove_empty_dirs(path) -> None:
    """Prune empty train/val/test scaffolding under a manifold folder."""
    path = Path(path)
    for sub in ["images", "labels", "targets", "masks", "shards"]:
        for split in ["train", "val", "test"]:
            p = path / sub / split
            if p.exists() and p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        p = path / sub
        if p.exists() and p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass