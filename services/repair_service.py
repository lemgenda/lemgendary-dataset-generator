"""RepairService - plan and apply on-disk repairs of compiled manifolds.

Single Responsibility: decide what a manifold needs (shard rewrites, then a
metadata resync against the repaired shards) and carry it out safely.

- ``plan`` only reads; it is what the CLI prints in its default dry run.
- ``apply`` rewrites damaged shards, then re-reads the shards and resyncs
  dataset_info.yaml and index.json. Every new file is written to a
  temporary file, verified (shards are re-audited), and swapped in with an
  atomic ``os.replace``; the original is deleted by that swap and no copy
  is kept, so a crash never leaves a half-written file in place.
- Running it again on a repaired manifold finds nothing to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tools.repair.metadata_fix import (
    MetadataFix,
    PhysicalSamples,
    apply_metadata_fix,
    plan_metadata_fix,
)
from tools.repair.shard_fix import ShardFix, apply_shard_fix, plan_shard_fix
from utils.shard_audit import ShardSetAudit, audit_shard_set, find_shards

__all__ = ["ManifoldRepairPlan", "RepairService", "physical_samples"]

_SPLITS = ("train", "val", "test")


@dataclass
class ManifoldRepairPlan:
    """Everything a repair of one manifold would change."""

    manifold_dir: Path
    shard_fixes: list[ShardFix] = field(default_factory=list)
    metadata: MetadataFix | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when applying would modify a file."""
        return bool(self.shard_fixes) or bool(self.metadata and self.metadata.changed)

    def report(self) -> list[str]:
        """Human-readable lines describing the plan."""
        lines = [f"[{self.manifold_dir.name}]"]
        for fix in self.shard_fixes:
            what = []
            if fix.drop_keys:
                what.append(f"drop {len(fix.drop_keys)} samples with zero-byte members"
                            f" ({', '.join(sorted(fix.drop_keys)[:5])})")
            if fix.regroup:
                what.append("regroup scattered members contiguously")
            lines.append(f"  shard {fix.shard.parent.name}/{fix.shard.name}: {'; '.join(what)}")
        if self.metadata is not None:
            lines.extend(f"  {line}" for line in self.metadata.summary())
        lines.extend(f"  note: {note}" for note in self.notes)
        if not self.changed:
            lines.append("  nothing to repair")
        return lines


def _shard_physical(audit: ShardSetAudit, fixes: list[ShardFix]) -> PhysicalSamples:
    """Shard keys per split, minus the samples the planned fixes drop."""
    dropped = {key for fix in fixes for key in fix.drop_keys}
    keys = {split: ks - dropped for split, ks in audit.keys_by_split.items()}
    return PhysicalSamples({s: len(k) for s, k in keys.items()}, keys, len(audit.shards))


def _directory_physical(images_dir: Path) -> PhysicalSamples:
    keys = {
        split.name: {p.stem for p in split.iterdir() if p.is_file()}
        for split in sorted(images_dir.iterdir()) if split.is_dir()
    }
    return PhysicalSamples({s: len(k) for s, k in keys.items()}, keys)


def _jsonl_split(stem: str) -> str | None:
    last = stem.rpartition("_")[2]
    return last if last in _SPLITS else None


def _jsonl_physical(data_dir: Path) -> PhysicalSamples | None:
    counts: dict[str, int] = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        split = _jsonl_split(path.stem)
        if split is None:
            continue
        with path.open(encoding="utf-8") as fh:
            counts[split] = counts.get(split, 0) + sum(1 for line in fh if line.strip())
    return PhysicalSamples(counts) if counts else None


def physical_samples(manifold_dir: Path, fixes: list[ShardFix] | None = None,
                     audit: ShardSetAudit | None = None) -> PhysicalSamples | None:
    """What is on disk, by format; None for formats without a resync (parquet, MDS)."""
    shards_dir = manifold_dir / "shards"
    if find_shards(shards_dir):
        return _shard_physical(audit or audit_shard_set(shards_dir), fixes or [])
    if (manifold_dir / "images").is_dir():
        return _directory_physical(manifold_dir / "images")
    if (manifold_dir / "data").is_dir():
        return _jsonl_physical(manifold_dir / "data")
    return None


class RepairService:
    """Plan and apply manifold repairs; see the module docstring for the safety rules."""

    def plan(self, manifold_dir: Path) -> ManifoldRepairPlan:
        """Read-only: the shard fixes and metadata resync ``manifold_dir`` needs."""
        plan = ManifoldRepairPlan(manifold_dir=manifold_dir)
        audit = audit_shard_set(manifold_dir / "shards")
        for shard in audit.shards:
            if shard.error:
                plan.notes.append(f"{shard.path.name} is unreadable ({shard.error});"
                                  " it needs to be re-downloaded or recompiled")
            fix = plan_shard_fix(shard)
            if fix is not None:
                plan.shard_fixes.append(fix)
        physical = physical_samples(manifold_dir, plan.shard_fixes, audit)
        if physical is None:
            plan.notes.append("no shards, images/ or data/*.jsonl: metadata resync skipped")
        else:
            plan.metadata = plan_metadata_fix(manifold_dir, physical)
        return plan

    def apply(self, plan: ManifoldRepairPlan) -> ManifoldRepairPlan:
        """Carry ``plan`` out and return a fresh plan of what is left (normally nothing)."""
        manifold_dir = plan.manifold_dir
        for fix in plan.shard_fixes:
            apply_shard_fix(fix)
        physical = physical_samples(manifold_dir)
        if physical is not None:
            metadata = plan_metadata_fix(manifold_dir, physical)
            if metadata.changed:
                apply_metadata_fix(metadata)
        return self.plan(manifold_dir)
