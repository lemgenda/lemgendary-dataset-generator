"""Filesystem helpers. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


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
        except (PermissionError, OSError) as exc:
            logger.debug("Failed scanning directory %s: %s", p, exc)
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
                except OSError as exc:
                    logger.debug("Non-empty directory or permission error pruning %s: %s", p, exc)
        p = path / sub
        if p.exists() and p.is_dir():
            try:
                p.rmdir()
            except OSError as exc:
                logger.debug("Non-empty directory or permission error pruning %s: %s", p, exc)