"""TxtLabelService - turn the ``.txt`` label members of a manifold's shards into JSON labels.

Single Responsibility: plan and carry out :mod:`tools.repair.txt_label_fix`
for every shard of one manifold (``repair txt-labels``).

- ``plan`` only reads; it is what the CLI prints in its default dry run.
- ``apply`` refuses before any write when a text label contradicts its JSON
  sidecar (MISMATCH) or cannot be read; otherwise each shard is rewritten to
  a temporary file, verified and swapped in atomically (no copies kept).
- Running it again on a repaired manifold finds nothing to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tools.repair.txt_label_fix import (
    TxtLabelError,
    TxtShardPlan,
    apply_txt_label_fix,
    plan_txt_label_fix,
)
from utils.shard_audit import find_shards

__all__ = ["TxtLabelPlan", "TxtLabelService"]

_SHOWN = 5


@dataclass
class TxtLabelPlan:
    """Everything converting one manifold's text labels would change."""

    manifold_dir: Path
    strategy: str
    shards: list[TxtShardPlan] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when applying would rewrite a shard."""
        return any(plan.changed for plan in self.shards)

    @property
    def mismatched(self) -> list[str]:
        """Keys whose text label contradicts their JSON sidecar."""
        return [key for plan in self.shards for key in plan.mismatched]

    @property
    def unreadable(self) -> list[str]:
        """Keys whose text label (or JSON sidecar) cannot be read."""
        return [key for plan in self.shards for key in plan.unreadable]

    def _first_conflict(self) -> str:
        for plan in self.shards:
            for key, conflicts in plan.mismatched.items():
                return f"{key}: {conflicts[0]}"
        return ""

    def report(self) -> list[str]:
        """Human-readable lines describing the plan."""
        total = sum(len(p.text_keys) for p in self.shards)
        lines = [
            f"[{self.manifold_dir.name}] text labels -> JSON (strategy {self.strategy})",
            (f"  shards {len(self.shards)}, samples {sum(p.samples for p in self.shards)},"
             f" .txt members {total} in {sum(1 for p in self.shards if p.changed)} shards"),
        ]
        if total:
            lines.append(
                f"  new JSON {sum(len(p.new_json) for p in self.shards)}, merged into the"
                f" JSON {sum(p.tally.merged for p in self.shards)}, already in the JSON"
                f" {sum(p.tally.unchanged for p in self.shards)}, empty (removed)"
                f" {sum(len(p.tally.emptied) for p in self.shards)}"
            )
        if self.mismatched:
            lines.append(f"  MISMATCH: {len(self.mismatched)} text labels contradict their JSON"
                         f" ({self._first_conflict()}); --apply refuses")
        if self.unreadable:
            lines.append(f"  UNREADABLE: {len(self.unreadable)} labels"
                         f" ({', '.join(self.unreadable[:_SHOWN])}); --apply refuses")
        lines.extend(f"  note: {note}" for note in self.notes)
        if not self.changed:
            lines.append("  nothing to repair")
        return lines


class TxtLabelService:
    """Plan and apply text-to-JSON label repairs with one label strategy."""

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def plan(self, manifold_dir: Path) -> TxtLabelPlan:
        """Read-only: the label conversion every shard of ``manifold_dir`` needs."""
        plan = TxtLabelPlan(manifold_dir=manifold_dir, strategy=self.strategy)
        shards = find_shards(manifold_dir / "shards")
        if not shards:
            plan.notes.append("no WebDataset shards")
        plan.shards = [plan_txt_label_fix(shard, self.strategy) for shard in shards]
        return plan

    def apply(self, plan: TxtLabelPlan) -> TxtLabelPlan:
        """Carry ``plan`` out and return a fresh plan of what is left (normally nothing).

        Raises :class:`TxtLabelError` before any write when a label conflicts
        or cannot be read.
        """
        if plan.mismatched or plan.unreadable:
            raise TxtLabelError(
                f"{len(plan.mismatched)} mismatched and {len(plan.unreadable)} unreadable"
                " labels; nothing was changed"
            )
        for shard_plan in plan.shards:
            apply_txt_label_fix(shard_plan)
        return self.plan(plan.manifold_dir)
