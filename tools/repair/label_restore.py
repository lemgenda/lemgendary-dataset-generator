"""Write the labels of a loose ``labels/`` folder into the JSON sidecars of shards.

Labels-only repair: a compiled WebDataset manifold whose sample keys are
the file stems of a loose Kaggle source (for example ``LemGendizedRetinaFace``
packed from ``lemgendizedretinafacemobilenetlarge``) gets each sample's
label from ``labels/<split>/<key>.txt`` without recompiling an image.

The text label goes through a ``label_json`` strategy
(:mod:`utils.label_json`) and the result is merged into the sample's
``<key>.json`` (the strategy's keys replace the old ones; every other key,
such as ``split``, is kept). No ``.txt`` member is added: the JSON sidecar
is the one label the training suite reads for these samples.

Determinism check: when a sidecar already records boxes
(``objects[].bbox_cxcywh_normalized``), the label's boxes must match them;
a mismatch means the label belongs to a different image, and the shard is
not rewritten (:func:`apply_label_restore` raises).

The rewritten shard goes to a temporary file, is re-audited, and replaces
the original with ``os.replace``; no copy of the original is kept.
"""

from __future__ import annotations

import json
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.repair.shard_fix import REPAIR_TMP_SUFFIX
from utils.label_json import build_label_json
from utils.sample_group import add_tar_member
from utils.shard_audit import audit_shard, split_member_name

__all__ = [
    "LabelGaps",
    "LabelRestoreError",
    "LabelShardPlan",
    "apply_label_restore",
    "index_label_files",
    "plan_label_restore",
]

_LABEL_EXT = ".txt"
_JSON_SUFFIX = ".json"
_BOX_KEY = "bbox_cxcywh_normalized"
_BOX_TOLERANCE = 1e-4


class LabelRestoreError(RuntimeError):
    """A shard cannot take the labels (mismatch or failed verification); it is untouched."""


def index_label_files(labels_dir: Path) -> dict[str, Path]:
    """Every ``*.txt`` label under ``labels_dir`` by file stem (the sample key).

    Raises ``ValueError`` when two files share a stem (for example the same
    key under ``train/`` and ``val/``), because the key would be ambiguous.
    """
    labels: dict[str, Path] = {}
    for path in labels_dir.rglob(f"*{_LABEL_EXT}"):
        if not path.is_file():
            continue
        if path.stem in labels:
            raise ValueError(f"label key {path.stem} appears twice: {labels[path.stem]} and {path}")
        labels[path.stem] = path
    return labels


@dataclass
class LabelGaps:
    """Samples a restore leaves as they are, by reason."""

    no_label: list[str] = field(default_factory=list)
    empty_label: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LabelShardPlan:
    """What restoring labels would change in one shard."""

    shard: Path
    samples: int
    update: frozenset[str] = frozenset()
    added_json: int = 0
    unchanged: int = 0
    gaps: LabelGaps = field(default_factory=LabelGaps)
    matched_keys: frozenset[str] = field(default_factory=frozenset)

    @property
    def changed(self) -> bool:
        """True when applying would rewrite the shard."""
        return bool(self.update)


def _base(key: str) -> str:
    return key.rpartition("/")[2]


def _sidecars(path: Path) -> dict[str, dict[str, Any] | None]:
    """Parsed JSON sidecar of every sample key in shard order (None: missing or unreadable)."""
    sidecars: dict[str, dict[str, Any] | None] = {}
    with tarfile.open(path, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            key, suffix = split_member_name(member.name)
            sidecars.setdefault(key, None)
            if suffix != _JSON_SUFFIX:
                continue
            handle = tf.extractfile(member)
            try:
                parsed = json.loads(handle.read().decode("utf-8")) if handle else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            sidecars[key] = parsed if isinstance(parsed, dict) else None
    return sidecars


def _boxes(meta: dict[str, Any] | None) -> list[list[float]]:
    objects = meta.get("objects") if meta else None
    if not isinstance(objects, list):
        return []
    return [list(obj[_BOX_KEY]) for obj in objects
            if isinstance(obj, dict) and isinstance(obj.get(_BOX_KEY), list)]


def _same_boxes(old: list[list[float]], new: list[list[float]]) -> bool:
    if len(old) != len(new):
        return False
    return all(len(a) == len(b) and all(abs(x - y) <= _BOX_TOLERANCE for x, y in zip(a, b))
               for a, b in zip(old, new))


def derived_sidecar(strategy: str, label: Path) -> dict[str, Any] | None:
    """The JSON fields ``strategy`` derives from the label file (None: nothing to write).

    Raises ``ValueError`` for a label the strategy cannot read and
    ``OSError`` / ``UnicodeDecodeError`` for an unreadable file.
    """
    return build_label_json(strategy, label.read_text(encoding="utf-8"))


@dataclass
class _Tally:
    update: set[str] = field(default_factory=set)
    matched: set[str] = field(default_factory=set)
    gaps: LabelGaps = field(default_factory=LabelGaps)
    added_json: int = 0
    unchanged: int = 0


def _tally_sample(tally: _Tally, key: str, meta: dict[str, Any] | None,
                  label: Path | None, strategy: str) -> None:
    if label is None:
        tally.gaps.no_label.append(key)
        return
    tally.matched.add(_base(key))
    try:
        derived = derived_sidecar(strategy, label)
    except (ValueError, OSError, UnicodeDecodeError):
        tally.gaps.unreadable.append(key)
        return
    if derived is None:
        tally.gaps.empty_label.append(key)
        return
    old_boxes = _boxes(meta)
    if old_boxes and not _same_boxes(old_boxes, _boxes(derived)):
        tally.gaps.mismatched.append(key)
        return
    if meta is not None and {**meta, **derived} == meta:
        tally.unchanged += 1
        return
    tally.update.add(key)
    if meta is None:
        tally.added_json += 1


def plan_label_restore(shard: Path, labels: dict[str, Path], strategy: str) -> LabelShardPlan:
    """Read-only: which samples of ``shard`` the labels would change, and what does not fit."""
    tally = _Tally()
    samples = 0
    for key, meta in _sidecars(shard).items():
        samples += 1
        _tally_sample(tally, key, meta, labels.get(_base(key)), strategy)
    return LabelShardPlan(
        shard=shard,
        samples=samples,
        update=frozenset(tally.update),
        added_json=tally.added_json,
        unchanged=tally.unchanged,
        gaps=tally.gaps,
        matched_keys=frozenset(tally.matched),
    )


def _merged_payload(meta: dict[str, Any] | None, derived: dict[str, Any]) -> bytes:
    return json.dumps({**(meta or {}), **derived}, separators=(",", ":")).encode("utf-8")


def _read_meta(tf: tarfile.TarFile, member: tarfile.TarInfo | None) -> dict[str, Any] | None:
    handle = tf.extractfile(member) if member is not None else None
    if handle is None:
        return None
    parsed = json.loads(handle.read().decode("utf-8"))
    return parsed if isinstance(parsed, dict) else None


def _write_group(src: tarfile.TarFile, dst: tarfile.TarFile, key: str,
                 members: list[tarfile.TarInfo], derived: dict[str, Any] | None) -> None:
    """Copy one sample's members; with ``derived`` its merged JSON sidecar goes last."""
    if derived is None:
        for member in members:
            dst.addfile(member, src.extractfile(member))
        return
    json_member = next((m for m in members if split_member_name(m.name)[1] == _JSON_SUFFIX), None)
    payload = _merged_payload(_read_meta(src, json_member), derived)
    for member in members:
        if member is not json_member:
            dst.addfile(member, src.extractfile(member))
    add_tar_member(dst, f"{key}{_JSON_SUFFIX}", payload)


def _rewrite(plan: LabelShardPlan, labels: dict[str, Path], strategy: str, out: Path) -> None:
    with tarfile.open(plan.shard, "r:") as src:
        groups: dict[str, list[tarfile.TarInfo]] = {}
        for member in src.getmembers():
            if member.isfile():
                groups.setdefault(split_member_name(member.name)[0], []).append(member)
        with tarfile.open(out, "w") as dst:
            for key, members in groups.items():
                derived = None
                if key in plan.update:
                    derived = derived_sidecar(strategy, labels[_base(key)])
                    if derived is None:
                        raise LabelRestoreError(f"{key}: the label has nothing to write")
                _write_group(src, dst, key, members, derived)


def _verify(plan: LabelShardPlan, out: Path) -> None:
    before = audit_shard(plan.shard)
    after = audit_shard(out)
    expected_members = before.members + plan.added_json
    if (after.error or after.needs_rewrite or after.samples != plan.samples
            or after.members != expected_members):
        raise LabelRestoreError(
            f"{plan.shard.name}: rewritten shard failed verification (samples"
            f" {after.samples} of {plan.samples}, members {after.members} of"
            f" {expected_members}, error {after.error})"
        )
    missing = [key for key in plan.update if _JSON_SUFFIX not in after.suffixes.get(key, [])]
    if missing:
        raise LabelRestoreError(f"{plan.shard.name}: no JSON sidecar for {missing[:5]}")


def apply_label_restore(plan: LabelShardPlan, labels: dict[str, Path], strategy: str) -> None:
    """Rewrite the shard with the merged sidecars, verify it, then swap it in atomically.

    Raises :class:`LabelRestoreError` (shard untouched) when the plan holds
    mismatched samples or the rewritten shard fails its verification.
    """
    mismatched = plan.gaps.mismatched
    if mismatched:
        raise LabelRestoreError(
            f"{plan.shard.name}: {len(mismatched)} labels do not match the recorded"
            f" boxes ({', '.join(mismatched[:5])}); not rewritten"
        )
    if not plan.changed:
        return
    tmp = plan.shard.with_name(plan.shard.name + REPAIR_TMP_SUFFIX)
    try:
        _rewrite(plan, labels, strategy, tmp)
        _verify(plan, tmp)
    except (OSError, tarfile.TarError, ValueError, LabelRestoreError):
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, plan.shard)
