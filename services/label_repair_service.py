"""LabelRepairService - restore the labels of a compiled manifold from a loose labels/ folder.

Single Responsibility: match a manifold's shard samples to the label files
of its loose Kaggle source by key, plan the JSON sidecar updates
(:mod:`tools.repair.label_restore`) and the dataset_info.yaml label
metadata, and carry them out safely.

- ``plan`` only reads; it is what ``repair labels`` prints in its default
  dry run.
- ``apply`` refuses to touch anything when a label contradicts a box the
  shard already records (the label belongs to another image); otherwise
  every shard is rewritten to a temporary file, verified and swapped in
  atomically, then dataset_info.yaml records the strategy (``label_json``)
  and, when the source ships a ``classes.txt``, the ``classes``.
- Running it again on a repaired manifold finds nothing to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.docs.generators.dataset_info_yaml import write_dataset_info
from tools.repair.label_restore import (
    LabelRestoreError,
    LabelShardPlan,
    apply_label_restore,
    index_label_files,
    plan_label_restore,
)
from tools.repair.metadata_fix import INFO_FILE, replace_file
from utils.shard_audit import find_shards

__all__ = ["LabelRepairPlan", "LabelRepairService", "LabelSource", "read_classes"]

_SHOWN = 5


def read_classes(path: Path) -> list[str] | None:
    """Non-empty lines of a ``classes.txt``; None when the file does not exist."""
    if not path.is_file():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@dataclass(frozen=True)
class LabelSource:
    """Where the labels come from and how they become JSON fields."""

    labels_dir: Path
    strategy: str
    classes: list[str] | None = None


@dataclass
class LabelRepairPlan:
    """Everything a labels-only repair of one manifold would change."""

    manifold_dir: Path
    source: LabelSource
    shards: list[LabelShardPlan] = field(default_factory=list)
    unused_labels: int = 0
    info_changes: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def mismatched(self) -> list[str]:
        """Sample keys whose label contradicts the boxes the shard records."""
        return self._keys("mismatched")

    @property
    def changed(self) -> bool:
        """True when applying would modify a file."""
        return any(plan.changed for plan in self.shards) or bool(self.info_changes)

    def _keys(self, gap: str) -> list[str]:
        return [key for plan in self.shards for key in getattr(plan.gaps, gap)]

    def report(self) -> list[str]:
        """Human-readable lines describing the plan."""
        samples = sum(plan.samples for plan in self.shards)
        updates = sum(len(plan.update) for plan in self.shards)
        lines = [
            (f"[{self.manifold_dir.name}] labels from {self.source.labels_dir.name}/"
             f" (strategy {self.source.strategy})"),
            (f"  samples {samples}, JSON sidecars to update {updates}"
             f" (new {sum(p.added_json for p in self.shards)}), already up to date"
             f" {sum(p.unchanged for p in self.shards)}"),
        ]
        no_label = self._keys("no_label")
        if no_label:
            lines.append(f"  samples without a label file: {len(no_label)}"
                         f" ({', '.join(no_label[:_SHOWN])})")
        empty = self._keys("empty_label")
        if empty:
            lines.append(f"  labels with nothing to write (kept as they are): {len(empty)}")
        unreadable = self._keys("unreadable")
        if unreadable:
            lines.append(f"  unreadable labels (skipped): {len(unreadable)}"
                         f" ({', '.join(unreadable[:_SHOWN])})")
        if self.mismatched:
            lines.append(f"  MISMATCH: {len(self.mismatched)} labels contradict the recorded"
                         f" boxes ({', '.join(self.mismatched[:_SHOWN])}); --apply refuses")
        if self.unused_labels:
            lines.append(f"  label files with no sample in this manifold: {self.unused_labels}")
        for key, value in self.info_changes.items():
            lines.append(f"  {INFO_FILE}: {key} -> {value}")
        lines.extend(f"  note: {note}" for note in self.notes)
        if not self.changed:
            lines.append("  nothing to repair")
        return lines


def _read_info(manifold_dir: Path) -> dict[str, Any]:
    with open(manifold_dir / INFO_FILE, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _info_changes(info: dict[str, Any], source: LabelSource) -> dict[str, Any]:
    wanted: dict[str, Any] = {"label_json": source.strategy}
    if source.classes:
        wanted["classes"] = source.classes
    return {key: value for key, value in wanted.items() if info.get(key) != value}


class LabelRepairService:
    """Plan and apply labels-only repairs; see the module docstring for the safety rules."""

    def __init__(self, source: LabelSource) -> None:
        self.source = source
        self._labels: dict[str, Path] | None = None

    @property
    def labels(self) -> dict[str, Path]:
        """Label files by key, indexed once per service."""
        if self._labels is None:
            self._labels = index_label_files(self.source.labels_dir)
        return self._labels

    def plan(self, manifold_dir: Path) -> LabelRepairPlan:
        """Read-only: the sidecar updates and metadata changes ``manifold_dir`` needs."""
        plan = LabelRepairPlan(manifold_dir=manifold_dir, source=self.source)
        shards = find_shards(manifold_dir / "shards")
        if not shards:
            plan.notes.append("no WebDataset shards: a labels-only repair needs shards")
            return plan
        if not self.labels:
            plan.notes.append(f"no *.txt labels under {self.source.labels_dir}")
            return plan
        matched: set[str] = set()
        for shard in shards:
            shard_plan = plan_label_restore(shard, self.labels, self.source.strategy)
            plan.shards.append(shard_plan)
            matched |= shard_plan.matched_keys
        plan.unused_labels = len(self.labels) - len(matched)
        if not (manifold_dir / INFO_FILE).is_file():
            plan.notes.append(f"no {INFO_FILE}: the label metadata is not recorded")
        elif matched:
            plan.info_changes = _info_changes(_read_info(manifold_dir), self.source)
        return plan

    def apply(self, plan: LabelRepairPlan) -> LabelRepairPlan:
        """Carry ``plan`` out and return a fresh plan of what is left (normally nothing).

        Raises :class:`LabelRestoreError` before any write when a label
        contradicts a recorded box.
        """
        if plan.mismatched:
            raise LabelRestoreError(
                f"{len(plan.mismatched)} labels contradict the recorded boxes"
                f" ({', '.join(plan.mismatched[:_SHOWN])}); nothing was changed"
            )
        for shard_plan in plan.shards:
            apply_label_restore(shard_plan, self.labels, self.source.strategy)
        if plan.info_changes:
            info = {**_read_info(plan.manifold_dir), **plan.info_changes}
            replace_file(plan.manifold_dir / INFO_FILE, lambda tmp: write_dataset_info(tmp, info))
        return self.plan(plan.manifold_dir)
