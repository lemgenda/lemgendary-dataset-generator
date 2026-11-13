"""Convert the ``.txt`` label members of WebDataset shards into their JSON labels.

JSON is the only label format inside shards (:mod:`utils.label_json`). A
shard packed before that rule carries ``<key>.txt`` members, often next to a
``<key>.json`` sidecar. For every such sample the text label goes through the
manifold's label strategy and is merged into the JSON document:

- a key only the text label has is added; a key both have must agree (numbers
  within :data:`TOLERANCE`, lists of the same length element by element), and
  the JSON's value is kept; anything the JSON alone has is kept, so no
  information is lost;
- a disagreement is a MISMATCH and a label the strategy cannot read is
  UNREADABLE: either refuses the whole apply before anything is written;
- an empty text label of a strategy without an empty document (a NIMA or a
  class label) carries nothing and is just removed.

Each shard is rewritten to a temporary file (``<key>.txt`` gone, the merged
``<key>.json`` last in its group), re-audited, and swapped in with
``os.replace``; no copy of the original is kept.
"""

from __future__ import annotations

import json
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.repair.shard_fix import REPAIR_TMP_SUFFIX
from utils.label_json import label_document
from utils.sample_group import add_tar_member
from utils.shard_audit import audit_shard, split_member_name

__all__ = [
    "TOLERANCE",
    "TxtLabelError",
    "TxtShardPlan",
    "TxtTally",
    "apply_txt_label_fix",
    "merge_label",
    "plan_txt_label_fix",
]

TOLERANCE = 1e-4
_TXT = ".txt"
_JSON = ".json"
_SHOWN = 5


class TxtLabelError(RuntimeError):
    """A shard cannot be converted (conflict or failed verification); it is untouched."""


def _number(value: Any) -> float | None:
    """``value`` as a float when it is a number (not a bool), else None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _merge(old: Any, new: Any, path: str, conflicts: list[str]) -> Any:
    """``old`` completed with what only ``new`` has; disagreements go to ``conflicts``."""
    if isinstance(old, dict) and isinstance(new, dict):
        merged = dict(old)
        for key, value in new.items():
            merged[key] = (_merge(old[key], value, f"{path}.{key}", conflicts)
                           if key in old else value)
        return merged
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            conflicts.append(f"{path}: {len(old)} entries in the JSON, {len(new)} in the text")
            return old
        return [_merge(a, b, f"{path}[{i}]", conflicts) for i, (a, b) in enumerate(zip(old, new))]
    old_number, new_number = _number(old), _number(new)
    if old_number is not None and new_number is not None:
        if abs(old_number - new_number) > TOLERANCE:
            conflicts.append(f"{path}: {old} in the JSON, {new} in the text")
        return old
    if old != new:
        conflicts.append(f"{path}: {old!r} in the JSON, {new!r} in the text")
    return old


def merge_label(meta: dict[str, Any] | None, derived: dict[str, Any]) -> tuple[dict[str, Any],
                                                                              list[str]]:
    """``meta`` merged with the text label's ``derived`` document, and the conflicts."""
    conflicts: list[str] = []
    merged = _merge(meta or {}, derived, "$", conflicts)
    return merged, conflicts


@dataclass
class TxtTally:
    """How the text labels of a shard fare: merged, already there, empty, conflicting."""

    merged: int = 0
    unchanged: int = 0
    emptied: list[str] = field(default_factory=list)
    mismatched: dict[str, list[str]] = field(default_factory=dict)
    unreadable: dict[str, str] = field(default_factory=dict)


@dataclass
class TxtShardPlan:
    """What converting the text labels of one shard would change."""

    shard: Path
    samples: int = 0
    members: int = 0
    payloads: dict[str, bytes] = field(default_factory=dict)
    new_json: set[str] = field(default_factory=set)
    text_keys: set[str] = field(default_factory=set)
    tally: TxtTally = field(default_factory=TxtTally)

    @property
    def mismatched(self) -> dict[str, list[str]]:
        """Keys whose text label contradicts the JSON sidecar, with the conflicts."""
        return self.tally.mismatched

    @property
    def unreadable(self) -> dict[str, str]:
        """Keys whose text label or JSON sidecar cannot be read, with the reason."""
        return self.tally.unreadable

    @property
    def changed(self) -> bool:
        """True when the shard holds text labels to remove."""
        return bool(self.text_keys)

    @property
    def blocked(self) -> bool:
        """True when a label conflicts or cannot be read (apply refuses)."""
        return bool(self.mismatched or self.unreadable)


def _read(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = tf.extractfile(member)
    return handle.read() if handle is not None else b""


def _groups(tf: tarfile.TarFile) -> dict[str, list[tarfile.TarInfo]]:
    groups: dict[str, list[tarfile.TarInfo]] = {}
    for member in tf.getmembers():
        if member.isfile():
            groups.setdefault(split_member_name(member.name)[0], []).append(member)
    return groups


def _by_suffix(members: list[tarfile.TarInfo], suffix: str) -> tarfile.TarInfo | None:
    return next((m for m in members if split_member_name(m.name)[1] == suffix), None)


def _plan_sample(plan: TxtShardPlan, tf: tarfile.TarFile, key: str,
                 members: list[tarfile.TarInfo], strategy: str) -> None:
    txt = _by_suffix(members, _TXT)
    if txt is None:
        return
    plan.text_keys.add(key)
    json_member = _by_suffix(members, _JSON)
    try:
        derived = label_document(strategy, _read(tf, txt).decode("utf-8"))
        meta = json.loads(_read(tf, json_member).decode("utf-8")) if json_member else None
    except (UnicodeDecodeError, ValueError) as exc:
        plan.unreadable[key] = str(exc)
        return
    if meta is not None and not isinstance(meta, dict):
        plan.unreadable[key] = "the JSON sidecar is not an object"
        return
    if derived is None:
        plan.tally.emptied.append(key)
        return
    merged, conflicts = merge_label(meta, derived)
    if conflicts:
        plan.mismatched[key] = conflicts
        return
    if merged == meta:
        plan.tally.unchanged += 1
        return
    plan.payloads[key] = json.dumps(merged, separators=(",", ":")).encode("utf-8")
    if json_member is None:
        plan.new_json.add(key)
    else:
        plan.tally.merged += 1


def plan_txt_label_fix(shard: Path, strategy: str) -> TxtShardPlan:
    """Read-only: the JSON documents the text labels of ``shard`` turn into, and conflicts."""
    plan = TxtShardPlan(shard=shard)
    with tarfile.open(shard, "r:") as tf:
        groups = _groups(tf)
        plan.samples = len(groups)
        plan.members = sum(len(m) for m in groups.values())
        for key, members in groups.items():
            _plan_sample(plan, tf, key, members, strategy)
    return plan


def _rewrite(plan: TxtShardPlan, out: Path) -> None:
    with tarfile.open(plan.shard, "r:") as src, tarfile.open(out, "w") as dst:
        for key, members in _groups(src).items():
            payload = plan.payloads.get(key)
            for member in members:
                suffix = split_member_name(member.name)[1]
                if suffix == _TXT or (payload is not None and suffix == _JSON):
                    continue
                dst.addfile(member, src.extractfile(member))
            if payload is not None:
                add_tar_member(dst, f"{key}{_JSON}", payload)


def _verify(plan: TxtShardPlan, out: Path) -> None:
    after = audit_shard(out)
    expected = plan.members - len(plan.text_keys) + len(plan.new_json)
    leftover = [k for k, have in after.suffixes.items() if _TXT in have]
    missing = [k for k in plan.payloads if _JSON not in after.suffixes.get(k, [])]
    sound = (not after.error and not after.needs_rewrite and after.samples == plan.samples
             and after.members == expected)
    if not sound or leftover or missing:
        raise TxtLabelError(
            f"{plan.shard.name}: rewritten shard failed verification (samples {after.samples}"
            f" of {plan.samples}, members {after.members} of {expected}, text labels left"
            f" {leftover[:_SHOWN]}, JSON missing {missing[:_SHOWN]}, error {after.error})"
        )


def apply_txt_label_fix(plan: TxtShardPlan) -> None:
    """Rewrite the shard without text labels, verify it, then swap it in atomically.

    Raises :class:`TxtLabelError` (shard untouched) when the plan is blocked
    or the rewritten shard fails its verification.
    """
    if plan.blocked:
        raise TxtLabelError(
            f"{plan.shard.name}: {len(plan.mismatched)} mismatched and {len(plan.unreadable)}"
            " unreadable labels; not rewritten"
        )
    if not plan.changed:
        return
    tmp = plan.shard.with_name(plan.shard.name + REPAIR_TMP_SUFFIX)
    try:
        _rewrite(plan, tmp)
        _verify(plan, tmp)
    except (OSError, tarfile.TarError, TxtLabelError):
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, plan.shard)
