"""PseudoLabelService - pseudo-label a manifold's samples into a label table.

Single Responsibility: resolve the manifold's registry ``class_labels``,
find the shards with pseudo-rule samples, and run
:class:`tools.repair.pseudo_labels.PseudoLabeler` over the shards whose table
is not written yet. The shards are only read; ``repair class-labels`` writes
the labels afterwards.

- ``plan`` only reads tar headers: per shard, the pseudo-rule samples and
  whether its table exists (what ``repair pseudo-labels`` prints in its
  default dry run).
- ``run`` labels the remaining shards (at most ``max_shards``), reporting
  each one through ``on_shard``; the throughput gives the ETA of the rest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from core.class_labels import PSEUDO_MODELS, ClassLabelRules
from core.source_manifolds import registry_entry_for_folder
from tools.repair.image_classifier import ImageClassifier, OnnxImageClassifier
from tools.repair.pseudo_labels import (
    Decoder,
    PoolDecoder,
    PseudoLabeler,
    ShardPseudoResult,
    pseudo_keys,
    table_path,
)
from tools.repair.source_label_fix import SourceLabelError
from utils.shard_audit import find_shards

__all__ = ["PseudoLabelService", "PseudoPlan", "PseudoShard"]


@dataclass(frozen=True)
class PseudoShard:
    """A shard with pseudo-rule samples, its table and whether the table exists."""

    shard: Path
    samples: int
    table: Path
    done: bool


@dataclass
class PseudoPlan:
    """The shards a pseudo-label run covers and what is already done."""

    manifold_dir: Path
    table_root: Path
    models: tuple[str, ...]
    shards: list[PseudoShard] = field(default_factory=list)

    @property
    def remaining(self) -> list[PseudoShard]:
        """Shards whose table is not written yet."""
        return [s for s in self.shards if not s.done]

    def report(self) -> list[str]:
        """Human-readable lines describing the plan."""
        done = [s for s in self.shards if s.done]
        lines = [f"[{self.manifold_dir.name}] pseudo labels with {', '.join(self.models)}",
                 (f"  shards with pseudo-rule samples {len(self.shards)}"
                  f" ({sum(s.samples for s in self.shards)} samples); tables written"
                  f" {len(done)} ({sum(s.samples for s in done)} samples); to run"
                  f" {len(self.remaining)} ({sum(s.samples for s in self.remaining)} samples)")]
        if not self.shards:
            lines.append("  no local shard holds pseudo-rule samples")
        return lines


ShardCallback = Callable[[ShardPseudoResult], None]


def _rules(manifold_dir: Path) -> ClassLabelRules:
    entry = registry_entry_for_folder(manifold_dir.name) or {}
    spec = entry.get("class_labels")
    if not spec:
        raise SourceLabelError(f"{manifold_dir.name}: the registry entry has no class_labels")
    return ClassLabelRules.from_spec(spec)


class PseudoLabelService:
    """Plan and run the pseudo labels of one manifold."""

    def __init__(self, rules: ClassLabelRules, table_root: Path,
                 classifiers: Callable[[], dict[str, ImageClassifier]], decoder: Decoder,
                 batch_size: int = 64) -> None:
        self.rules = rules
        self.table_root = table_root
        self.classifiers = classifiers
        self.decoder = decoder
        self.batch_size = batch_size

    @classmethod
    def plan_only(cls, manifold_dir: Path, table_root: Path) -> PseudoPlan:
        """The plan of ``manifold_dir`` (no model is loaded)."""
        return plan_pseudo_labels(manifold_dir, _rules(manifold_dir), table_root)

    @staticmethod
    def onnx_classifiers(rules: ClassLabelRules,
                         threads: int = 0) -> Callable[[], dict[str, ImageClassifier]]:
        """A loader of the rules' models as ONNX classifiers."""
        def _load() -> dict[str, ImageClassifier]:
            return {name: OnnxImageClassifier(PSEUDO_MODELS[name], threads)
                    for name in rules.models}
        return _load

    @classmethod
    def run_onnx(cls, manifold_dir: Path, table_root: Path, workers: int,
                 max_shards: int | None = None,
                 on_shard: ShardCallback | None = None) -> list[ShardPseudoResult]:
        """Label the remaining shards with the ONNX models and ``workers`` decode processes."""
        rules = _rules(manifold_dir)
        plan = plan_pseudo_labels(manifold_dir, rules, table_root)
        with PoolDecoder(workers) as decoder:
            service = cls(rules, table_root, cls.onnx_classifiers(rules), decoder)
            return service.run(plan, max_shards, on_shard)

    def run(self, plan: PseudoPlan, max_shards: int | None = None,
            on_shard: ShardCallback | None = None) -> list[ShardPseudoResult]:
        """Label up to ``max_shards`` of the plan's remaining shards."""
        todo = plan.remaining if max_shards is None else plan.remaining[:max_shards]
        if not todo:
            return []
        labeler = PseudoLabeler(self.rules, self.classifiers(), self.decoder, self.batch_size)
        results: list[ShardPseudoResult] = []
        for item in todo:
            result = labeler.label_shard(item.shard, item.table)
            results.append(result)
            if on_shard is not None:
                on_shard(result)
        return results


def plan_pseudo_labels(manifold_dir: Path, rules: ClassLabelRules,
                       table_root: Path) -> PseudoPlan:
    """Read-only: the shards with pseudo-rule samples and their table state."""
    plan = PseudoPlan(manifold_dir, table_root, rules.models)
    shards_dir = manifold_dir / "shards"
    for shard in find_shards(shards_dir):
        samples = pseudo_keys(shard, rules)
        if samples:
            table = table_path(table_root, shards_dir, shard)
            plan.shards.append(PseudoShard(shard, samples, table, table.is_file()))
    return plan
