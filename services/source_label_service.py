"""SourceLabelService - write a source's per-file labels into a compiled manifold's shards.

Single Responsibility: resolve the manifold's source label scheme and key
layouts from the registry (``source_labels``, ``key_layouts``), pick the
labeler for the scheme, and run :func:`tools.repair.source_label_fix.label_shard`
over every shard (in worker processes when ``workers`` > 1).

- ``plan`` only reads; it is what ``repair masks`` and ``repair
  face-landmarks`` print in their default dry run.
- ``apply`` refuses before any write while the plan has a problem; otherwise
  each shard the plan found to change is rewritten, verified and swapped in atomically,
  dataset_info.yaml records ``label_json`` and the labeler's fields (mask
  ``classes``, also written to classes.txt), and a fresh plan is returned
  (normally nothing left to do).

New schemes plug in through :data:`LABELER_FACTORIES` (scheme type to
labeler), so the engine and this service stay closed for modification.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.docs.generators.dataset_info_yaml import write_dataset_info
from core.sample_policy import KeyLayout, parse_key_layouts
from core.source_labels import SOURCE_LABEL_SCHEMES, MaskScheme
from core.source_manifolds import registry_entry_for_folder
from tools.repair.mask_labeler import MaskLabeler
from tools.repair.metadata_fix import INFO_FILE, replace_file
from tools.repair.source_label_fix import (
    LABELLED,
    UNCHANGED,
    SampleLabeler,
    ShardLabelResult,
    SourceFiles,
    SourceLabelError,
    label_shard,
)
from utils.shard_audit import find_shards

__all__ = ["LABELER_FACTORIES", "SourceLabelPlan", "SourceLabelService"]

LabelerFactory = Callable[[Any, str], SampleLabeler]
LABELER_FACTORIES: dict[type, LabelerFactory] = {MaskScheme: MaskLabeler}

_SHOWN = 5
_CLASSES_FILE = "classes.txt"
_PERCENTILES = (5, 50, 95)


@dataclass
class SourceLabelPlan:
    """What labelling one manifold found per shard."""

    manifold_dir: Path
    title: str
    results: list[ShardLabelResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when a shard has samples to label."""
        return any(r.changed for r in self.results)

    @property
    def problems(self) -> int:
        """Samples whose label file is missing or unusable (apply refuses)."""
        return sum(r.problem_count for r in self.results)

    def statuses(self) -> Counter[str]:
        """Samples per status over all shards."""
        total: Counter[str] = Counter()
        for result in self.results:
            total.update(result.statuses)
        return total

    def metrics(self) -> dict[str, list[float]]:
        """Every metric value over all shards."""
        merged: dict[str, list[float]] = {}
        for result in self.results:
            for name, values in result.metrics.items():
                merged.setdefault(name, []).extend(values)
        return merged

    def report(self) -> list[str]:
        """Human-readable lines describing the plan."""
        statuses = self.statuses()
        lines = [f"[{self.manifold_dir.name}] {self.title}",
                 (f"  shards {len(self.results)}, samples"
                  f" {sum(r.samples for r in self.results)}, shards to rewrite"
                  f" {sum(1 for r in self.results if r.changed)}, rewritten"
                  f" {sum(1 for r in self.results if r.written)}")]
        lines.append("  " + ", ".join(f"{name} {n}" for name, n in sorted(statuses.items())))
        for name, values in sorted(self.metrics().items()):
            arr = np.asarray(values, dtype=np.float64)
            p5, p50, p95 = np.percentile(arr, _PERCENTILES)
            lines.append(f"  {name}: n {arr.size}, mean {arr.mean():.4f}, p5 {p5:.4f},"
                         f" median {p50:.4f}, p95 {p95:.4f}")
        if self.problems:
            examples = [f"{k}: {v}" for r in self.results for k, v in r.problems.items()]
            lines.append(f"  PROBLEMS: {self.problems} samples ({' | '.join(examples[:_SHOWN])});"
                         " --apply refuses")
        lines.extend(f"  note: {note}" for note in self.notes)
        if not self.changed and not self.problems:
            lines.append(f"  nothing to change ({statuses[UNCHANGED]} already labelled)")
        return lines


_WORKER: dict[str, SourceFiles] = {}


def _init_worker(root: Path, layouts: Mapping[str, KeyLayout], suffix: str) -> None:
    _WORKER["files"] = SourceFiles.build(root, layouts, suffix)


def _worker_shard(shard: Path, labeler: SampleLabeler, layouts: Mapping[str, KeyLayout],
                  write: bool) -> ShardLabelResult:
    return label_shard(shard, labeler, layouts, _WORKER["files"], write)


ShardCallback = Callable[[ShardLabelResult], None]


class SourceLabelService:
    """Plan and apply one source's per-file labels for one manifold."""

    def __init__(self, labeler: SampleLabeler, layouts: Mapping[str, KeyLayout], root: Path,
                 workers: int = 1, label_json: str | None = None) -> None:
        self.labeler = labeler
        self.label_json = label_json
        self.layouts = dict(layouts)
        self.root = root
        self.workers = max(1, workers)
        self.on_shard: ShardCallback | None = None

    @classmethod
    def for_manifold(cls, manifold_dir: Path, root: Path, scheme_type: type,
                     workers: int = 1) -> SourceLabelService:
        """The service for ``manifold_dir``'s registry ``source_labels`` of ``scheme_type``.

        Raises :class:`SourceLabelError` when the entry has no such source
        labels or no key layouts, ``OSError`` / ``yaml.YAMLError`` when the registry
        cannot be read.
        """
        entry = registry_entry_for_folder(manifold_dir.name) or {}
        spec = entry.get("source_labels") or {}
        scheme = SOURCE_LABEL_SCHEMES.get(str(spec.get("scheme", "")))
        if not isinstance(scheme, scheme_type):
            raise SourceLabelError(f"{manifold_dir.name}: the registry entry has no"
                                   f" {scheme_type.__name__} source_labels")
        layouts = parse_key_layouts(entry.get("key_layouts"))
        if not layouts:
            raise SourceLabelError(f"{manifold_dir.name}: the registry entry has no key_layouts")
        labeler = LABELER_FACTORIES[scheme_type](scheme, str(spec["source"]))
        return cls(labeler, layouts, root, workers, entry.get("label_json"))

    def _run(self, manifold_dir: Path, title: str, write: bool,
             only: list[Path] | None = None) -> SourceLabelPlan:
        plan = SourceLabelPlan(manifold_dir=manifold_dir, title=title)
        shards = find_shards(manifold_dir / "shards") if only is None else only
        if not shards:
            plan.notes.append("no WebDataset shards")
            return plan
        if self.workers == 1:
            files = SourceFiles.build(self.root, self.layouts, self.labeler.file_suffix)
            for shard in shards:
                self._collect(plan, label_shard(shard, self.labeler, self.layouts, files, write))
            return plan
        with ProcessPoolExecutor(max_workers=self.workers, initializer=_init_worker,
                                 initargs=(self.root, self.layouts,
                                           self.labeler.file_suffix)) as pool:
            futures = [pool.submit(_worker_shard, shard, self.labeler, self.layouts, write)
                       for shard in shards]
            for future in as_completed(futures):
                self._collect(plan, future.result())
        plan.results.sort(key=lambda r: r.shard)
        return plan

    def _collect(self, plan: SourceLabelPlan, result: ShardLabelResult) -> None:
        plan.results.append(result)
        if self.on_shard is not None:
            self.on_shard(result)

    def _record_metadata(self, manifold_dir: Path) -> None:
        """dataset_info.yaml ``label_json`` and the labeler's fields; classes.txt for classes."""
        fields = dict(self.labeler.info_fields())
        if self.label_json:
            fields["label_json"] = self.label_json
        info_path = manifold_dir / INFO_FILE
        if info_path.is_file():
            loaded = yaml.safe_load(info_path.read_text(encoding="utf-8"))
            info = loaded if isinstance(loaded, dict) else {}
            if any(info.get(k) != v for k, v in fields.items()):
                info.update(fields)
                replace_file(info_path, lambda tmp: write_dataset_info(tmp, info))
        classes = fields.get("classes")
        if classes:
            text = "\n".join(classes) + "\n"
            replace_file(manifold_dir / _CLASSES_FILE,
                         lambda tmp: tmp.write_text(text, encoding="utf-8", newline="\n"))

    def plan(self, manifold_dir: Path) -> SourceLabelPlan:
        """Read-only: what labelling every shard of ``manifold_dir`` would change.

        Raises :class:`SourceLabelError` when the label files do not match
        the key layouts.
        """
        return self._run(manifold_dir, f"source labels, dry run ({self.labeler.file_suffix})",
                         write=False)

    def apply(self, plan: SourceLabelPlan) -> SourceLabelPlan:
        """Carry ``plan`` out and return a fresh plan of what is left.

        Raises :class:`SourceLabelError` before any write while ``plan`` has
        a problem, or when a rewritten shard fails its verification.
        """
        if plan.problems:
            raise SourceLabelError(f"{plan.problems} samples have no usable label file;"
                                   " nothing was changed")
        written = self._run(plan.manifold_dir, "source labels, apply", write=True,
                            only=[r.shard for r in plan.results if r.changed])
        if written.problems:
            raise SourceLabelError(f"{written.problems} samples have no usable label file;"
                                   " their shards were not rewritten")
        self._record_metadata(plan.manifold_dir)
        fresh = self.plan(plan.manifold_dir)
        fresh.notes.append(f"rewritten {sum(1 for r in written.results if r.written)} shards,"
                           f" labelled {written.statuses()[LABELLED]} samples")
        return fresh
