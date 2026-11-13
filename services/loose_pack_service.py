"""LoosePackService - pack a loose images/labels manifold into WebDataset shards in place.

Single Responsibility: migrate one directory manifold (``images/<split>/``
and ``labels/<split>/``, the YOLO layout of ``LemGendizedYoloV8n``) into
``shards/<split>/shard-*.data`` with ``<key>.webp`` + ``<key>.json`` members
(``repair pack-loose``).

- ``plan`` only reads: per split the images, labels, empty labels (images
  without objects, kept as valid negatives), images without a label file and
  label files without an image, plus the class names the manifold ships
  (``classes.txt``, else the ``names`` of a YOLO ``data.yaml``).
- ``apply`` refuses while an image has no label file, a label has no image or
  shards already exist. It packs with the registry's label strategy
  (:func:`tools.modernize.loose_convert.convert_loose_directory_to_webdataset`),
  then verifies that every split's shard keys are exactly its image stems,
  that every labelled sample has its JSON label and that the labels pass the
  shard verifier; only then dataset_info.yaml records ``classes`` and
  ``label_json``, index.json and the counts are resynced
  (:class:`services.repair_service.RepairService`) and the loose folders are
  removed. On a failed verification the loose folders are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.docs.generators.dataset_info_yaml import write_dataset_info
from services.repair_service import RepairService
from tools.modernize.common import (
    LooseSplit,
    loose_split_names,
    purge_loose_dirs,
    scan_loose_split,
)
from tools.modernize.loose_convert import convert_loose_directory_to_webdataset
from tools.repair.metadata_fix import INDEX_FILE, INFO_FILE, replace_file
from utils.label_audit import audit_labels, label_audit_problems
from utils.shard_audit import audit_shard_set, find_shards

__all__ = ["LoosePackError", "LoosePackPlan", "LoosePackService", "SplitCounts",
           "dataset_class_names"]

_SHOWN = 5
_CLASSES_FILE = "classes.txt"
_YOLO_YAMLS = ("data.yaml", "dataset.yaml")


class LoosePackError(RuntimeError):
    """The loose manifold cannot be packed, or the packed shards failed verification."""


def _yaml_names(path: Path) -> list[str] | None:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = loaded.get("names") if isinstance(loaded, dict) else None
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return None


def dataset_class_names(manifold_dir: Path) -> list[str] | None:
    """Class names of the manifold: ``classes.txt``, else a YOLO yaml's ``names``."""
    classes = manifold_dir / _CLASSES_FILE
    if classes.is_file():
        return [line.strip() for line in classes.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    for name in _YOLO_YAMLS:
        path = manifold_dir / name
        if path.is_file():
            return _yaml_names(path)
    return None


@dataclass(frozen=True)
class SplitCounts:
    """Images and labels of one loose split, and the keys that do not pair up."""

    name: str
    images: int
    labels: int
    empty_labels: int
    unlabelled: list[str]
    orphan_labels: list[str]
    image_keys: frozenset[str]

    @classmethod
    def of(cls, split: LooseSplit) -> SplitCounts:
        """Count ``split``."""
        image_keys = frozenset(p.stem for p in split.images)
        return cls(
            name=split.name,
            images=len(split.images),
            labels=len(split.labels),
            empty_labels=sum(1 for p in split.labels.values() if p.stat().st_size == 0),
            unlabelled=sorted(image_keys - split.labels.keys()),
            orphan_labels=sorted(split.labels.keys() - image_keys),
            image_keys=image_keys,
        )


@dataclass
class LoosePackPlan:
    """What packing one loose manifold would do."""

    manifold_dir: Path
    strategy: str
    splits: list[SplitCounts] = field(default_factory=list)
    classes: list[str] | None = None
    blockers: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when there are loose images to pack."""
        return any(s.images for s in self.splits)

    def report(self) -> list[str]:
        """Human-readable lines describing the plan."""
        lines = [(f"[{self.manifold_dir.name}] loose images/labels -> WebDataset shards"
                 f" (strategy {self.strategy})")]
        for s in self.splits:
            lines.append(f"  {s.name}: images {s.images}, labels {s.labels} (empty, images"
                         f" without objects: {s.empty_labels})")
            if s.unlabelled:
                lines.append(f"    images without a label file: {len(s.unlabelled)}"
                             f" ({', '.join(s.unlabelled[:_SHOWN])})")
            if s.orphan_labels:
                lines.append(f"    labels without an image: {len(s.orphan_labels)}"
                             f" ({', '.join(s.orphan_labels[:_SHOWN])})")
        lines.append(f"  classes: {len(self.classes)} ({', '.join(self.classes[:_SHOWN])}"
                     f"{', ...' if len(self.classes) > _SHOWN else ''})" if self.classes
                     else "  classes: none found (classes.txt or data.yaml names)")
        lines.extend(f"  BLOCKED: {blocker}" for blocker in self.blockers)
        if not self.changed:
            lines.append("  nothing to pack")
        return lines


def _read_info(manifold_dir: Path) -> dict[str, Any]:
    path = manifold_dir / INFO_FILE
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


class LoosePackService:
    """Plan and apply the in-place loose-to-shards migration of one manifold."""

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def plan(self, manifold_dir: Path) -> LoosePackPlan:
        """Read-only: the loose splits, their pairing and what blocks the pack."""
        plan = LoosePackPlan(manifold_dir=manifold_dir, strategy=self.strategy,
                             classes=dataset_class_names(manifold_dir))
        if (manifold_dir / "images").is_dir():
            plan.splits = [SplitCounts.of(scan_loose_split(manifold_dir, name))
                           for name in loose_split_names(manifold_dir)]
        if find_shards(manifold_dir / "shards"):
            plan.blockers.append("shards/ already holds shards; move them away first")
        unlabelled = sum(len(s.unlabelled) for s in plan.splits)
        orphans = sum(len(s.orphan_labels) for s in plan.splits)
        if unlabelled:
            plan.blockers.append(f"{unlabelled} images have no label file")
        if orphans:
            plan.blockers.append(f"{orphans} label files have no image")
        return plan

    def apply(self, plan: LoosePackPlan) -> list[str]:
        """Pack, verify, record the metadata and remove the loose folders.

        Returns the verification report lines. Raises :class:`LoosePackError`
        when the plan is blocked or the shards fail verification (the loose
        folders are then kept).
        """
        if plan.blockers:
            raise LoosePackError("; ".join(plan.blockers))
        manifold_dir = plan.manifold_dir
        if not convert_loose_directory_to_webdataset(manifold_dir, manifold_dir,
                                                     label_json=self.strategy):
            raise LoosePackError("packing or its label verification failed; loose folders kept")
        report = self._verify(plan)
        self._record_metadata(plan)
        purge_loose_dirs(manifold_dir, "Removing the loose folders (every sample verified)...")
        return report

    def _verify(self, plan: LoosePackPlan) -> list[str]:
        shards_dir = plan.manifold_dir / "shards"
        audit = audit_shard_set(shards_dir)
        problems = []
        for split in plan.splits:
            keys = audit.keys_by_split.get(split.name, set())
            if keys != split.image_keys:
                problems.append(f"{split.name}: {len(split.image_keys - keys)} images missing"
                                f" from the shards, {len(keys - split.image_keys)} unexpected")
        unlabelled = [k for shard in audit.shards for k, have in shard.suffixes.items()
                      if ".json" not in have]
        labels = sum(s.labels for s in plan.splits)
        if len(unlabelled) != sum(s.images for s in plan.splits) - labels:
            problems.append(f"{len(unlabelled)} samples without a JSON label, expected"
                            f" {sum(s.images for s in plan.splits) - labels}")
        labels_audit = audit_labels(shards_dir, self.strategy)
        problems.extend(label_audit_problems(labels_audit))
        if problems:
            raise LoosePackError("shard verification failed, loose folders kept: "
                                 + "; ".join(problems))
        return [(f"  verified: {audit.total_samples} samples in {len(audit.shards)} shards"
                f" ({', '.join(f'{k} {v}' for k, v in sorted(audit.split_counts.items()))}),"
                f" {labels_audit.json_labels} JSON labels, every image with its label")]

    def _record_metadata(self, plan: LoosePackPlan) -> None:
        manifold_dir = plan.manifold_dir
        info = {"name": manifold_dir.name, **_read_info(manifold_dir),
                "label_json": self.strategy}
        if plan.classes:
            info["classes"] = plan.classes
        replace_file(manifold_dir / INFO_FILE, lambda tmp: write_dataset_info(tmp, info))
        index_path = manifold_dir / INDEX_FILE
        if not index_path.is_file():
            index_path.write_text(json.dumps([]), encoding="utf-8")
        service = RepairService()
        resync = service.plan(manifold_dir)
        if resync.changed:
            service.apply(resync)
