"""Bring dataset_info.yaml and index.json back in line with the samples on disk.

Physical counts come from the shards (WebDataset), ``images/<split>/``
(directory manifolds) or the ``data/*_<split>.jsonl`` line counts (LLM
manifolds). The fix

- sets ``count`` (and ``total_samples``, ``<split>_samples`` and
  ``total_shards`` when recorded) to the physical numbers;
- replaces an absolute or staging ``path`` with the canonical relative one;
- drops index.json entries with no physical sample (including ``skipped``
  rows), corrects entries filed under the wrong split, adds a minimal entry
  for every physical sample the index lacks, and fills a missing ``source``.

Index entries of JSONL manifolds cannot be matched to lines, so only their
``source`` is filled; a count mismatch there is reported, not rewritten.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from core.docs.generators.dataset_info_yaml import write_dataset_info
from utils.naming import manifold_info_path

__all__ = [
    "IndexChanges",
    "MetadataFix",
    "PhysicalSamples",
    "apply_metadata_fix",
    "plan_metadata_fix",
]

INFO_FILE = "dataset_info.yaml"
INDEX_FILE = "index.json"


@dataclass
class PhysicalSamples:
    """What is on disk: counts per split, the keys when they can be named, shard count."""

    split_counts: dict[str, int]
    keys_by_split: dict[str, set[str]] | None = None
    shards: int | None = None

    @property
    def total(self) -> int:
        """Samples across every split."""
        return sum(self.split_counts.values())


@dataclass
class IndexChanges:
    """How many index.json entries a resync removes, adds, re-files or completes."""

    removed: int = 0
    added: int = 0
    resplit: int = 0
    source_filled: int = 0

    def __bool__(self) -> bool:
        return bool(self.removed or self.added or self.resplit or self.source_filled)

    def lines(self) -> list[str]:
        """One summary line per non-zero counter."""
        counts = (("removed", self.removed), ("added", self.added),
                  ("moved to their physical split", self.resplit),
                  ("given a source", self.source_filled))
        return [f"{INDEX_FILE}: {n} entries {what}" for what, n in counts if n]


@dataclass
class MetadataFix:
    """Planned edits of one manifold's metadata files."""

    manifold_dir: Path
    info_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    index: IndexChanges = field(default_factory=IndexChanges)
    notes: list[str] = field(default_factory=list)
    new_info: dict[str, Any] | None = None
    new_index: list[dict[str, Any]] | None = None

    @property
    def changed(self) -> bool:
        """True when applying would rewrite a file."""
        return bool(self.info_changes or self.index)

    def summary(self) -> list[str]:
        """One line per planned change, then the notes."""
        lines = [f"{INFO_FILE}: {k} {old!r} -> {new!r}"
                 for k, (old, new) in self.info_changes.items()]
        return lines + self.index.lines() + self.notes


def _is_absolute(path: str) -> bool:
    return PureWindowsPath(path).is_absolute() or PurePosixPath(path).is_absolute()


def _info_updates(info: dict[str, Any], physical: PhysicalSamples,
                  name: str) -> dict[str, tuple[Any, Any]]:
    wanted: dict[str, Any] = {"count": physical.total}
    if "total_samples" in info:
        wanted["total_samples"] = physical.total
    for split, count in physical.split_counts.items():
        if f"{split}_samples" in info:
            wanted[f"{split}_samples"] = count
    if "total_shards" in info and physical.shards is not None:
        wanted["total_shards"] = physical.shards
    path = info.get("path")
    if isinstance(path, str) and (_is_absolute(path) or ".staging" in path):
        wanted["path"] = manifold_info_path(name)
    return {k: (info.get(k), v) for k, v in wanted.items() if info.get(k) != v}


def _new_entry(entry_id: int, name: str, split: str, info: dict[str, Any],
               source: str) -> dict[str, Any]:
    return {
        "id": entry_id, "name": name, "source": source, "task": info.get("task"),
        "split": split, "hash": None, "nima_score": None, "caption": None,
        "style_tag": None, "cluster_id": -1,
    }


def _sync_index(fix: MetadataFix, index: list[dict[str, Any]], physical: PhysicalSamples,
                info: dict[str, Any], source: str) -> list[dict[str, Any]]:
    keys = physical.keys_by_split
    out: list[dict[str, Any]] = []
    split_of: dict[str, str] = {}
    if keys is not None:
        split_of = {k.rpartition("/")[2]: split for split, ks in keys.items() for k in ks}
    seen: set[str] = set()
    for entry in index:
        name = str(entry.get("name", ""))
        if keys is not None:
            if name not in split_of or name in seen:
                fix.index.removed += 1
                continue
            seen.add(name)
            if entry.get("split") != split_of[name]:
                entry = {**entry, "split": split_of[name]}
                fix.index.resplit += 1
        if not entry.get("source"):
            entry = {**entry, "source": source}
            fix.index.source_filled += 1
        out.append(entry)
    next_id = max((e.get("id", -1) for e in out if isinstance(e.get("id"), int)), default=-1) + 1
    for name in sorted(set(split_of) - seen):
        out.append(_new_entry(next_id, name, split_of[name], info, source))
        next_id += 1
        fix.index.added += 1
    return out


def _index_note(index: list[dict[str, Any]], physical: PhysicalSamples) -> str | None:
    counts: dict[str, int] = {}
    for entry in index:
        split = str(entry.get("split"))
        counts[split] = counts.get(split, 0) + 1
    if counts == physical.split_counts:
        return None
    return (f"{INDEX_FILE}: split counts {counts} differ from the samples on disk"
            f" {physical.split_counts}; entries cannot be matched to samples, left as is")


def _names_match(index: list[dict[str, Any]], physical: PhysicalSamples) -> bool:
    """True when most index names are physical sample keys (or keys are unknown)."""
    if physical.keys_by_split is None or not index:
        return True
    names = {k.rpartition("/")[2] for ks in physical.keys_by_split.values() for k in ks}
    matched = sum(1 for e in index if str(e.get("name", "")) in names)
    return matched * 2 >= len(index)


def plan_metadata_fix(manifold_dir: Path, physical: PhysicalSamples) -> MetadataFix:
    """Work out the metadata edits; nothing is written."""
    fix = MetadataFix(manifold_dir=manifold_dir)
    info_path, index_path = manifold_dir / INFO_FILE, manifold_dir / INDEX_FILE
    info: dict[str, Any] = {}
    if info_path.is_file():
        loaded = yaml.safe_load(info_path.read_text(encoding="utf-8"))
        info = loaded if isinstance(loaded, dict) else {}
    name = str(info.get("name") or manifold_dir.name)
    fix.info_changes = _info_updates(info, physical, name)
    if fix.info_changes:
        fix.new_info = {**info, **{k: new for k, (_, new) in fix.info_changes.items()}}
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(index, list):
            if not _names_match(index, physical):
                fix.notes.append(f"{INDEX_FILE}: fewer than half of its names match the samples"
                                 " on disk; entries kept, only missing sources are filled")
                physical = PhysicalSamples(physical.split_counts, None, physical.shards)
            fix.new_index = _sync_index(fix, index, physical, info, name)
            if physical.keys_by_split is None:
                note = _index_note(fix.new_index, physical)
                fix.notes.extend([note] if note else [])
    return fix


def _replace_file(path: Path, write: Any) -> None:
    """Write ``path`` anew via ``write(tmp)`` and swap it in atomically (no copy kept)."""
    tmp = path.with_name(path.name + ".repair-tmp")
    try:
        write(tmp)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def apply_metadata_fix(fix: MetadataFix) -> None:
    """Write the planned files; each replaces its original atomically."""
    if fix.new_info is not None and fix.info_changes:
        info = fix.new_info
        _replace_file(fix.manifold_dir / INFO_FILE,
                      lambda tmp: write_dataset_info(tmp, info))
    if fix.new_index is not None and fix.index:
        index = fix.new_index

        def _write_index(tmp: Path) -> None:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)

        _replace_file(fix.manifold_dir / INDEX_FILE, _write_index)
