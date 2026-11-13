"""ShardOffsetsService - write the shard offset index of an existing manifold.

Single Responsibility: plan and write ``shard_offsets.json`` in every shard
directory of a compiled manifold (:mod:`utils.shard_offsets`) without touching
the shards.

- ``plan`` only reads: per shard directory it says whether the index is
  missing, stale (with the reasons) or current. A present index is checked the
  way the training suite checks it (size, then modification time or three
  header reads), so the dry run does not rescan every header.
- ``apply`` writes the index of every shard directory (entries of unchanged
  shards are kept, the rest rescanned) and returns what it wrote and how long
  it took; a fresh ``plan`` afterwards finds nothing to do.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from utils.shard_files import SHARD_SUFFIXES
from utils.shard_offsets import (
    OFFSETS_FILENAME,
    OffsetsWrite,
    offsets_problems,
    shard_dirs,
    write_shard_offsets,
)

__all__ = ["ShardDirIndex", "ShardOffsetsPlan", "ShardOffsetsRun", "ShardOffsetsService"]

_SHOWN_PROBLEMS = 5
_MIB = 1024 * 1024


@dataclass(frozen=True)
class ShardDirIndex:
    """Index state of one shard directory."""

    shard_dir: Path
    shards: int
    present: bool
    problems: tuple[str, ...]

    @property
    def needs_write(self) -> bool:
        """True when the index is missing or stale."""
        return not self.present or bool(self.problems)

    def line(self, shards_root: Path) -> str:
        """One report line for this directory."""
        where = self.shard_dir.relative_to(shards_root.parent).as_posix()
        if not self.present:
            state = "missing, would write"
        elif self.problems:
            state = f"stale ({len(self.problems)} problems), would rewrite"
        else:
            state = "current"
        return f"  {where}: {self.shards} shards, {OFFSETS_FILENAME} {state}"


@dataclass
class ShardOffsetsPlan:
    """The index state of every shard directory of one manifold."""

    manifold_dir: Path
    dirs: list[ShardDirIndex] = field(default_factory=list)

    @property
    def shards_root(self) -> Path:
        """The manifold's ``shards/`` folder."""
        return self.manifold_dir / "shards"

    @property
    def changed(self) -> bool:
        """True when applying would write an index."""
        return any(d.needs_write for d in self.dirs)

    def report(self) -> list[str]:
        """Human-readable lines describing the plan."""
        lines = [f"[{self.manifold_dir.name}]"]
        if not self.dirs:
            lines.append("  no WebDataset shards")
        for state in self.dirs:
            lines.append(state.line(self.shards_root))
            lines.extend(f"    {problem}" for problem in state.problems[:_SHOWN_PROBLEMS])
            if len(state.problems) > _SHOWN_PROBLEMS:
                lines.append(f"    ... and {len(state.problems) - _SHOWN_PROBLEMS} more")
        return lines


@dataclass(frozen=True)
class ShardOffsetsRun:
    """What ``apply`` wrote, how long it took, and the plan afterwards."""

    written: list[OffsetsWrite]
    seconds: float
    remaining: ShardOffsetsPlan

    @property
    def bytes_written(self) -> int:
        """Total size of the index files written."""
        return sum(w.bytes_written for w in self.written)

    def report(self) -> list[str]:
        """One line per directory and a total line."""
        lines = []
        for w in self.written:
            size = f"{w.bytes_written / _MIB:.2f} MiB" if w.path is not None else "no file"
            lines.append(f"  {w.shard_dir.name}: {w.shards} shards, {w.scanned} rescanned,"
                         f" {w.members} members, {size}")
        lines.append(f"  total: {self.bytes_written / _MIB:.2f} MiB in {self.seconds:.1f} s")
        return lines


def _dir_state(shard_dir: Path) -> ShardDirIndex:
    shards = sum(1 for p in shard_dir.iterdir() if p.is_file() and p.suffix in SHARD_SUFFIXES)
    present = (shard_dir / OFFSETS_FILENAME).is_file()
    problems = tuple(offsets_problems(shard_dir)) if present else ()
    return ShardDirIndex(shard_dir, shards, present, problems)


class ShardOffsetsService:
    """Plan and write the shard offset index of one manifold."""

    def plan(self, manifold_dir: Path) -> ShardOffsetsPlan:
        """Read-only: the index state of every shard directory of ``manifold_dir``."""
        plan = ShardOffsetsPlan(manifold_dir=manifold_dir)
        plan.dirs = [_dir_state(d) for d in shard_dirs(plan.shards_root)]
        return plan

    def apply(self, plan: ShardOffsetsPlan) -> ShardOffsetsRun:
        """Write every directory's index; the shards are only read."""
        started = time.perf_counter()
        written = write_shard_offsets(plan.shards_root)
        seconds = time.perf_counter() - started
        return ShardOffsetsRun(written, seconds, self.plan(plan.manifold_dir))
