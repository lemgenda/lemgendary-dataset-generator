"""Apply a sample policy (:mod:`core.sample_policy`) to the shards of a compiled manifold.

The plan reads every shard's JSON sidecars once and decides, per sample key:

1. drop the key when the registry's key layout marks it a duplicate sample
   copy or an auxiliary image (a face-parsing map scanned as an image);
2. drop it when its label holds no face;
3. otherwise give it its split: a seeded share (``val_split``) of the
   landmark-bearing photos of each source goes to val (drawn by source file
   id), every other photo goes to train.

A key stored twice keeps one copy (one already in its split, if any).

Apply, in an order that a re-run can always finish:

- samples that change split are copied into new shards of their new split
  (temporary file, verified, renamed in);
- each shard that loses samples is rewritten without them to a temporary
  file, verified and swapped in with ``os.replace`` (an emptied shard is
  deleted); no copy of an original is kept;
- shard numbers are made contiguous per split again;
- index.json (dropped entries removed, ``split`` updated) and
  dataset_info.yaml ``count`` are rewritten atomically.

Planning again after an apply finds nothing to do.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.config_schema import DatasetEntry
from core.docs.generators.dataset_info_yaml import write_dataset_info
from core.sample_policy import (
    FACE_LANDMARKS,
    FACE_NONE,
    ROLE_PHOTO,
    KeyLayout,
    SamplePolicy,
    face_kind,
    key_role,
    parse_key_layouts,
    sample_policy,
    split_rank,
)
from core.source_manifolds import registry_entry_for_folder
from tools.repair.label_restore import read_sidecars
from tools.repair.metadata_fix import INDEX_FILE, INFO_FILE, replace_file
from tools.repair.shard_fix import REPAIR_TMP_SUFFIX
from utils.container_stream import SHARD_SAMPLES
from utils.shard_audit import audit_shard, find_shards, split_member_name

__all__ = [
    "PlanCounts",
    "SamplePolicyError",
    "SamplePolicyPlan",
    "SampleRule",
    "apply_sample_policy",
    "enforce_sample_policy",
    "plan_sample_policy",
    "rule_for_manifold",
]

TRAIN, VAL = "train", "val"
DROP = "drop"
_SHARD_RE = re.compile(r"^shard-(\d+)(\.[a-z]+)$")
_SHOWN = 5


class SamplePolicyError(RuntimeError):
    """A rewritten shard failed its verification; the original is untouched."""


@dataclass(frozen=True)
class SampleRule:
    """The policy, its val share and the registry's key layouts."""

    policy: SamplePolicy
    val_split: float
    layouts: dict[str, KeyLayout] = field(default_factory=dict)


def rule_for_manifold(manifold_dir: Path) -> SampleRule | None:
    """The registry's sample rule for the manifold in ``manifold_dir``; None when it has none.

    Raises ``OSError`` or ``yaml.YAMLError`` when the registry cannot be read.
    """
    entry = registry_entry_for_folder(manifold_dir.name)
    if entry is None:
        return None
    policy = sample_policy(entry.get("label_json"), entry.get("task"))
    if policy is None:
        return None
    val_split = entry.get("val_split", DatasetEntry.model_construct(name="").val_split)
    return SampleRule(policy, float(val_split), parse_key_layouts(entry.get("key_layouts")))


@dataclass(frozen=True)
class _Seen:
    """One stored sample: where it is and what its sidecar says."""

    shard: Path
    split: str
    kind: str


@dataclass
class PlanCounts:
    """Sample counts before and after, drops by reason, and the metadata updates."""

    before: Counter[str] = field(default_factory=Counter)
    reasons: Counter[str] = field(default_factory=Counter)
    after: Counter[str] = field(default_factory=Counter)
    val_by_source: dict[str, tuple[int, int]] = field(default_factory=dict)
    index_changes: int = 0
    info_count: int | None = None


@dataclass
class SamplePolicyPlan:
    """Everything applying the policy to one manifold would change."""

    manifold_dir: Path
    rule: SampleRule
    # shard -> {key: DROP | new split}; only shards that change are listed.
    actions: dict[Path, dict[str, str]] = field(default_factory=dict)
    final: dict[str, str] = field(default_factory=dict)
    dropped: set[str] = field(default_factory=set)
    counts: PlanCounts = field(default_factory=PlanCounts)
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when applying would modify a shard, index.json or dataset_info.yaml."""
        return bool(self.actions or self.counts.index_changes or self.counts.info_count is not None)

    def moves(self, target: str) -> int:
        """Samples that move into ``target``."""
        return sum(1 for acts in self.actions.values() for a in acts.values() if a == target)

    def report(self) -> list[str]:
        """Human-readable lines describing the plan."""
        b, a = self.counts.before, self.counts.after
        lines = [
            (f"[{self.manifold_dir.name}] sample policy {self.rule.policy.name}"
             f" (val_split {self.rule.val_split})"),
            (f"  now: {b['samples']} samples (train {b['train']}, val {b['val']});"
             f" with landmarks {b['landmarks']} (val {b['val_landmarks']}),"
             f" box only {b['box_only']}, no face {b['no_face']}"),
        ]
        lines.extend(f"  drop {reason}: {n}" for reason, n in sorted(self.counts.reasons.items()))
        lines.append(f"  move train -> val {self.moves(VAL)}, val -> train {self.moves(TRAIN)}")
        lines.append(
            f"  after: {a['samples']} samples (train {a['train']}, val {a['val']});"
            f" val with landmarks {a['val_landmarks']}, train with landmarks"
            f" {a['train_landmarks']}, train box only {a['train_box_only']}"
        )
        for source, (val, total) in sorted(self.counts.val_by_source.items()):
            lines.append(f"  val {source}: {val} of {total} landmark photos")
        if self.counts.index_changes:
            lines.append(f"  {INDEX_FILE}: {self.counts.index_changes} entries removed or re-split")
        if self.counts.info_count is not None:
            lines.append(f"  {INFO_FILE}: count -> {self.counts.info_count}")
        lines.extend(f"  note: {note}" for note in self.notes)
        if not self.changed:
            lines.append("  nothing to change")
        return lines


def _split_of(shards_dir: Path, shard: Path) -> str:
    parts = shard.relative_to(shards_dir).parts
    return parts[0] if len(parts) > 1 else TRAIN


def _scan(manifold_dir: Path, plan: SamplePolicyPlan) -> dict[str, list[_Seen]]:
    shards_dir = manifold_dir / "shards"
    seen: dict[str, list[_Seen]] = {}
    for shard in find_shards(shards_dir):
        split = _split_of(shards_dir, shard)
        for key, meta in read_sidecars(shard).items():
            kind = face_kind(meta)
            seen.setdefault(key, []).append(_Seen(shard, split, kind))
            plan.counts.before["samples"] += 1
            plan.counts.before[split] += 1
            plan.counts.before[kind] += 1
            if split == VAL and kind == FACE_LANDMARKS:
                plan.counts.before["val_landmarks"] += 1
    return seen


def _drop_reason(key: str, kind: str, rule: SampleRule) -> str | None:
    role = key_role(key, rule.layouts).role
    if role != ROLE_PHOTO:
        return role
    if rule.policy.drop_without_objects and kind == FACE_NONE:
        return FACE_NONE
    return None


def _val_keys(candidates: dict[str, list[tuple[str, str]]], plan: SamplePolicyPlan) -> set[str]:
    """Per source, the ``val_split`` share of candidates with the lowest seeded rank."""
    chosen: set[str] = set()
    for source, items in candidates.items():
        ranked = sorted((split_rank(plan.rule.policy, source, file_id), key)
                        for key, file_id in items)
        count = round(len(ranked) * plan.rule.val_split)
        chosen.update(key for _, key in ranked[:count])
        plan.counts.val_by_source[source] = (count, len(ranked))
    return chosen


def _desired(seen: dict[str, list[_Seen]], plan: SamplePolicyPlan) -> dict[str, str]:
    desired: dict[str, str] = {}
    candidates: dict[str, list[tuple[str, str]]] = {}
    for key, places in seen.items():
        kind = places[0].kind
        reason = _drop_reason(key, kind, plan.rule)
        if reason is not None:
            desired[key] = DROP
            plan.counts.reasons[reason] += 1
            continue
        desired[key] = TRAIN
        if kind == FACE_LANDMARKS or not plan.rule.policy.val_needs_landmarks:
            role = key_role(key, plan.rule.layouts)
            candidates.setdefault(role.source, []).append((key, role.file_id))
    for key in _val_keys(candidates, plan):
        desired[key] = VAL
    return desired


def _place(key: str, places: list[_Seen], target: str, plan: SamplePolicyPlan) -> None:
    """Record the actions that leave exactly one copy of ``key`` in ``target``."""
    keep = next((p for p in places if p.split == target), None)
    for place in places:
        if place is keep:
            continue
        if keep is None:
            keep = place
            plan.actions.setdefault(place.shard, {})[key] = target
        else:
            plan.actions.setdefault(place.shard, {})[key] = DROP
            plan.counts.reasons["duplicate copy"] += 1


def _tally_after(key: str, target: str, kind: str, plan: SamplePolicyPlan) -> None:
    plan.final[key] = target
    plan.counts.after["samples"] += 1
    plan.counts.after[target] += 1
    plan.counts.after[f"{target}_{kind}"] += 1


def _base(key: str) -> str:
    return key.rpartition("/")[2]


def _index_entries(plan: SamplePolicyPlan) -> tuple[list[Any], int] | None:
    """index.json after the plan and the number of entries that change; None without one."""
    path = plan.manifold_dir / INDEX_FILE
    if not path.is_file():
        return None
    final = {_base(k): v for k, v in plan.final.items()}
    dropped = {_base(k) for k in plan.dropped}
    kept: list[Any] = []
    changes = 0
    for entry in json.loads(path.read_text(encoding="utf-8")):
        name = str(entry.get("name", "")) if isinstance(entry, dict) else ""
        if name in dropped:
            changes += 1
            continue
        if name in final and entry.get("split") != final[name]:
            entry["split"] = final[name]
            changes += 1
        kept.append(entry)
    return kept, changes


def _read_info(plan: SamplePolicyPlan) -> dict[str, Any] | None:
    path = plan.manifold_dir / INFO_FILE
    if not path.is_file():
        return None
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _plan_metadata(plan: SamplePolicyPlan) -> None:
    index = _index_entries(plan)
    plan.counts.index_changes = index[1] if index else 0
    info = _read_info(plan)
    if info is not None and info.get("count") != len(plan.final):
        plan.counts.info_count = len(plan.final)


def plan_sample_policy(manifold_dir: Path, rule: SampleRule) -> SamplePolicyPlan:
    """Read-only: the drops and split moves the policy asks of ``manifold_dir``."""
    plan = SamplePolicyPlan(manifold_dir=manifold_dir, rule=rule)
    seen = _scan(manifold_dir, plan)
    if not seen:
        plan.notes.append("no WebDataset shards")
        return plan
    for key, target in _desired(seen, plan).items():
        places = seen[key]
        if target == DROP:
            plan.dropped.add(key)
            for place in places:
                plan.actions.setdefault(place.shard, {})[key] = DROP
            continue
        _place(key, places, target, plan)
        _tally_after(key, target, places[0].kind, plan)
    _plan_metadata(plan)
    return plan


# ─── Apply ──────────────────────────────────────────────────────────────────
def _members_by_key(tf: tarfile.TarFile) -> Iterator[tuple[str, list[tarfile.TarInfo]]]:
    """Each sample's members in shard order (groups are contiguous in a sound shard)."""
    key, group = "", []
    for member in tf:
        if not member.isfile():
            continue
        member_key = split_member_name(member.name)[0]
        if group and member_key != key:
            yield key, group
            group = []
        key = member_key
        group.append(member)
    if group:
        yield key, group


def _verify(path: Path, keys: list[str], label: str) -> None:
    audit = audit_shard(path)
    if audit.error or audit.needs_rewrite or audit.keys != keys:
        raise SamplePolicyError(
            f"{label}: rewritten shard failed verification (samples {audit.samples} of"
            f" {len(keys)}, error {audit.error})"
        )


def _shard_number(path: Path) -> int | None:
    match = _SHARD_RE.match(path.name)
    return int(match.group(1)) if match else None


def _moves_by_target(plan: SamplePolicyPlan) -> dict[str, list[tuple[Path, str]]]:
    """``(shard, key)`` of every sample that changes split, by its new split, in shard order."""
    moves: dict[str, list[tuple[Path, str]]] = {}
    for shard, acts in sorted(plan.actions.items()):
        for key, action in acts.items():
            if action != DROP:
                moves.setdefault(action, []).append((shard, key))
    return moves


def _write_new_shard(final: Path, items: list[tuple[Path, str]]) -> None:
    """Copy the samples ``items`` into the new shard ``final`` (verified, then renamed in)."""
    wanted: dict[Path, set[str]] = {}
    for shard, key in items:
        wanted.setdefault(shard, set()).add(key)
    tmp = final.with_name(final.name + REPAIR_TMP_SUFFIX)
    written: list[str] = []
    try:
        with tarfile.open(tmp, "w") as dst:
            for shard, keys in wanted.items():
                with tarfile.open(shard, "r:") as src:
                    for key, members in _members_by_key(src):
                        if key in keys:
                            for member in members:
                                dst.addfile(member, src.extractfile(member))
                            written.append(key)
        if len(written) != len(items):
            raise SamplePolicyError(f"{final.name}: {len(written)} of {len(items)} samples found")
        _verify(tmp, written, final.name)
    except (OSError, tarfile.TarError, SamplePolicyError):
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, final)


def _copy_moves(plan: SamplePolicyPlan, ext: str) -> None:
    shards_dir = plan.manifold_dir / "shards"
    for target, items in _moves_by_target(plan).items():
        split_dir = shards_dir / target
        split_dir.mkdir(parents=True, exist_ok=True)
        numbers = [n for p in split_dir.glob("shard-*") if (n := _shard_number(p)) is not None]
        number = max(numbers, default=-1) + 1
        for start in range(0, len(items), SHARD_SAMPLES):
            _write_new_shard(split_dir / f"shard-{number:05d}{ext}",
                             items[start:start + SHARD_SAMPLES])
            number += 1


def _rewrite_source(shard: Path, acts: dict[str, str]) -> None:
    tmp = shard.with_name(shard.name + REPAIR_TMP_SUFFIX)
    kept: list[str] = []
    try:
        with tarfile.open(shard, "r:") as src, tarfile.open(tmp, "w") as dst:
            for key, members in _members_by_key(src):
                if key in acts:
                    continue
                for member in members:
                    dst.addfile(member, src.extractfile(member))
                kept.append(key)
        if not kept:
            tmp.unlink()
            shard.unlink()
            return
        _verify(tmp, kept, shard.name)
    except (OSError, tarfile.TarError, SamplePolicyError):
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, shard)


def _renumber(split_dir: Path) -> None:
    """Rename the shards of one split to contiguous numbers, keeping their order."""
    numbered = sorted((n, p) for p in split_dir.glob("shard-*")
                      if (n := _shard_number(p)) is not None)
    for want, (have, path) in enumerate(numbered):
        if want != have:
            os.replace(path, path.with_name(f"shard-{want:05d}{path.suffix}"))


def _clear_stale_tmp(shards_dir: Path) -> None:
    for stale in shards_dir.rglob(f"*{REPAIR_TMP_SUFFIX}"):
        stale.unlink()


def _rewrite_index(plan: SamplePolicyPlan) -> None:
    index = _index_entries(plan)
    if index is None or not index[1]:
        return
    entries = index[0]

    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    replace_file(plan.manifold_dir / INDEX_FILE, _write)


def _rewrite_info(plan: SamplePolicyPlan) -> None:
    info = _read_info(plan)
    if info is None or info.get("count") == len(plan.final):
        return
    info["count"] = len(plan.final)
    replace_file(plan.manifold_dir / INFO_FILE, lambda tmp: write_dataset_info(tmp, info))


def apply_sample_policy(plan: SamplePolicyPlan) -> SamplePolicyPlan:
    """Carry ``plan`` out (see the module docstring) and return a fresh plan."""
    shards_dir = plan.manifold_dir / "shards"
    if plan.actions:
        _clear_stale_tmp(shards_dir)
        ext = next(iter(sorted(plan.actions))).suffix
        _copy_moves(plan, ext)
        for shard, acts in sorted(plan.actions.items()):
            _rewrite_source(shard, acts)
        for split_dir in (p for p in shards_dir.iterdir() if p.is_dir()):
            _renumber(split_dir)
    _rewrite_index(plan)
    _rewrite_info(plan)
    return plan_sample_policy(plan.manifold_dir, plan.rule)


def enforce_sample_policy(manifold_dir: Path) -> bool:
    """Compiler step after packing shards: apply the registry's sample policy, if any.

    Returns False when the manifold is still not clean afterwards.
    """
    rule = rule_for_manifold(manifold_dir)
    if rule is None:
        return True
    plan = plan_sample_policy(manifold_dir, rule)
    for line in plan.report():
        print(f"[POLICY] {line}")
    if not plan.changed:
        return True
    remaining = apply_sample_policy(plan)
    if remaining.changed:
        print("[POLICY] The manifold still needs changes after the sample policy pass.")
        return False
    return True
