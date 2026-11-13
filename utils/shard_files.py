"""Which files are WebDataset shards, and where a manifold's shards are.

Single source of the shard suffixes and the shard listing, shared by the shard
audit (:mod:`utils.shard_audit`) and the shard offset index
(:mod:`utils.shard_offsets`).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["SHARD_SUFFIXES", "find_shards"]

SHARD_SUFFIXES = (".data", ".tar")


def find_shards(shards_dir: Path) -> list[Path]:
    """Every shard under ``shards_dir`` (all splits), sorted."""
    if not shards_dir.is_dir():
        return []
    return sorted(p for p in shards_dir.rglob("*") if p.is_file() and p.suffix in SHARD_SUFFIXES)
