"""Inspect WebDataset tar shards: members, sample keys, zero-byte members and grouping.

Single source of truth for "is this shard set sound?", shared by the
converters' post-write verification, ``verify_archive`` and the
``repair`` CLI. Members are read from the tar headers only; payloads are
never loaded.

A sample key is the member path up to the first dot of its basename
(``a.target.webp`` belongs to ``a``). A shard is well grouped when every
key's members form one unbroken run.

A zero-byte member is a defect, except a text member: an empty ``.txt``
is a legitimate label (a detection image with no objects).
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils.shard_files import SHARD_SUFFIXES, find_shards
from utils.shard_offsets import offsets_problems

__all__ = [
    "MAY_BE_EMPTY_SUFFIXES",
    "SHARD_SUFFIXES",
    "ExpectedCounts",
    "ShardAudit",
    "ShardSetAudit",
    "audit_shard",
    "audit_shard_set",
    "expected_counts",
    "find_shards",
    "is_empty_defect",
    "members_by_key",
    "shard_set_problems",
    "split_member_name",
    "verify_shard_set",
]

_ALL_SPLIT = "all"
MAY_BE_EMPTY_SUFFIXES = (".txt",)
_REPORTED_PROBLEMS = 10


def split_member_name(name: str) -> tuple[str, str]:
    """``("dir/a", ".target.webp")`` for ``"dir/a.target.webp"`` (WebDataset key rule)."""
    head, _, base = name.rpartition("/")
    stem, dot, rest = base.partition(".")
    key = f"{head}/{stem}" if head else stem
    return key, f"{dot}{rest}"


def members_by_key(tf: tarfile.TarFile) -> Iterator[tuple[str, list[tarfile.TarInfo]]]:
    """Each sample's file members in shard order (groups are contiguous in a sound shard)."""
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


def _is_target(suffix: str) -> bool:
    return suffix.startswith(".target.")


def is_empty_defect(name: str, size: int) -> bool:
    """True when a member of ``size`` bytes named ``name`` is an unusable empty file."""
    return size == 0 and not name.lower().endswith(MAY_BE_EMPTY_SUFFIXES)


@dataclass
class ShardAudit:
    """What one shard holds, from its tar headers."""

    path: Path
    members: int = 0
    suffixes: dict[str, list[str]] = field(default_factory=dict)
    zero_byte: list[str] = field(default_factory=list)
    scattered: list[str] = field(default_factory=list)
    max_key_distance: int = 0
    error: str | None = None

    @property
    def keys(self) -> list[str]:
        """Sample keys in first-seen order."""
        return list(self.suffixes)

    @property
    def samples(self) -> int:
        """Number of distinct sample keys."""
        return len(self.suffixes)

    @property
    def zero_byte_keys(self) -> set[str]:
        """Keys with at least one empty member (the whole sample is unusable)."""
        return {split_member_name(name)[0] for name in self.zero_byte}

    @property
    def needs_rewrite(self) -> bool:
        """True when a repair would change this shard."""
        return bool(self.zero_byte or self.scattered)

    def unpaired(self) -> list[str]:
        """Keys without a ``.target.*`` member."""
        return [k for k, have in self.suffixes.items() if not any(map(_is_target, have))]

    def incomplete(self, required: tuple[str, ...]) -> list[str]:
        """Keys missing any of the ``required`` member suffixes."""
        return [k for k, have in self.suffixes.items() if not set(required) <= set(have)]


def audit_shard(path: Path) -> ShardAudit:
    """Read the tar headers of ``path``; a corrupt tar is reported in ``error``."""
    audit = ShardAudit(path=path)
    first_seen: dict[str, int] = {}
    last_key = ""
    try:
        with tarfile.open(path, "r:") as tf:
            for index, member in enumerate(tf):
                if not member.isfile():
                    continue
                audit.members += 1
                key, suffix = split_member_name(member.name)
                if is_empty_defect(member.name, member.size):
                    audit.zero_byte.append(member.name)
                if key not in first_seen:
                    first_seen[key] = index
                    audit.suffixes[key] = []
                elif key != last_key and key not in audit.scattered:
                    audit.scattered.append(key)
                audit.suffixes[key].append(suffix)
                audit.max_key_distance = max(audit.max_key_distance, index - first_seen[key])
                last_key = key
    except (OSError, tarfile.TarError) as exc:
        audit.error = str(exc)
    return audit


def _split_of(shards_dir: Path, shard: Path) -> str:
    parts = shard.relative_to(shards_dir).parts
    return parts[0] if len(parts) > 1 else _ALL_SPLIT


@dataclass
class ShardSetAudit:
    """All shards of one manifold, with per-split sample counts."""

    shards_dir: Path
    shards: list[ShardAudit] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        """Samples across every shard."""
        return sum(self.split_counts.values())

    @property
    def keys_by_split(self) -> dict[str, set[str]]:
        """Every sample key, by split."""
        out: dict[str, set[str]] = {}
        for shard in self.shards:
            out.setdefault(_split_of(self.shards_dir, shard.path), set()).update(shard.keys)
        return out


def audit_shard_set(shards_dir: Path) -> ShardSetAudit:
    """Audit every shard of ``shards_dir`` (not just the first and last)."""
    result = ShardSetAudit(shards_dir=shards_dir)
    for shard in find_shards(shards_dir):
        audit = audit_shard(shard)
        result.shards.append(audit)
        split = _split_of(shards_dir, shard)
        result.split_counts[split] = result.split_counts.get(split, 0) + audit.samples
    return result


@dataclass(frozen=True)
class ExpectedCounts:
    """Sample counts the manifold's metadata claims (None when not recorded)."""

    info_count: int | None
    index_splits: dict[str, int] | None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_index(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else None


def expected_counts(manifold_dir: Path) -> ExpectedCounts:
    """Counts from dataset_info.yaml (``count``) and index.json (entries per split)."""
    info = _read_yaml(manifold_dir / "dataset_info.yaml")
    count = info.get("count")
    index = _read_index(manifold_dir / "index.json")
    splits: dict[str, int] | None = None
    if index is not None:
        splits = {}
        for entry in index:
            split = str(entry.get("split") or _ALL_SPLIT)
            splits[split] = splits.get(split, 0) + 1
    return ExpectedCounts(int(count) if isinstance(count, int) else None, splits)


def shard_set_problems(
    audit: ShardSetAudit,
    required: tuple[str, ...] = (),
    expected_total: int | None = None,
) -> list[str]:
    """Human-readable defects: unreadable shards, empty members, scattered or
    incomplete groups, a total that differs from ``expected_total``, and a stale
    ``shard_offsets.json`` (:func:`utils.shard_offsets.offsets_problems`).

    A set where any sample has a ``.target.*`` member is a paired set: every
    sample must have one.
    """
    problems: list[str] = []
    if not audit.shards:
        problems.append(f"no shards under {audit.shards_dir}")
    paired = any(_is_target(suffix) for shard in audit.shards
                 for have in shard.suffixes.values() for suffix in have)
    for shard in audit.shards:
        unpaired = shard.unpaired() if paired else []
        if unpaired:
            problems.append(f"{shard.path.name}: {len(unpaired)} samples without a target"
                            f" (first: {unpaired[0]})")
        name = shard.path.name
        if shard.error:
            problems.append(f"{name}: unreadable tar ({shard.error})")
        if shard.zero_byte:
            problems.append(f"{name}: {len(shard.zero_byte)} zero-byte members")
        if shard.scattered:
            problems.append(
                f"{name}: {len(shard.scattered)} samples not contiguous"
                f" (max member distance {shard.max_key_distance})"
            )
        missing = shard.incomplete(required) if required else []
        if missing:
            problems.append(f"{name}: {len(missing)} samples missing one of {required}")
    if expected_total is not None and audit.total_samples != expected_total:
        problems.append(f"{audit.total_samples} samples in shards, expected {expected_total}")
    problems.extend(offsets_problems(audit.shards_dir,
                                     {shard.path: shard.members for shard in audit.shards}))
    return problems


def verify_shard_set(shards_dir: Path, tag: str) -> bool:
    """Audit every shard under ``shards_dir``; print the defects, False when any."""
    problems = shard_set_problems(audit_shard_set(shards_dir))
    for problem in problems[:_REPORTED_PROBLEMS]:
        print(f"[{tag}] {problem}")
    if len(problems) > _REPORTED_PROBLEMS:
        print(f"[{tag}] ... and {len(problems) - _REPORTED_PROBLEMS} more shard problems")
    return not problems
