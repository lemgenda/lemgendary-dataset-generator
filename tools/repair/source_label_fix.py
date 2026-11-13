"""Write per-file source labels (masks, landmarks) into the samples of WebDataset shards.

A source that ships one label file per image (SFHQ ``segmentations/``,
``landmarks/``) is mapped to the compiled samples by key: the registry's
key layout (:mod:`core.sample_policy`) gives each photo key its source file
id, and the id is the rank of the file name among the source's label files
(the compiler scanned the image folder in sorted name order, and the label
folders hold the same names). Copies and auxiliary images are left alone.

A :class:`SampleLabeler` turns one photo and its label file into the
members to add or replace (a ``.mask.webp``) and its new JSON label. Per
shard, :func:`label_shard` reads every sample once and either only reports
(dry run) or writes a temporary shard (other members kept byte for byte,
new members after the image, the JSON last), verifies it and swaps it in
with ``os.replace``; no copy of the original is kept. A shard with a
problem (no layout for a source, a missing or invalid label file) is never
written. A sample whose label is already in place is unchanged, so a re-run
finds nothing to do.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from core.sample_policy import ROLE_PHOTO, KeyLayout, key_role
from tools.repair.shard_fix import REPAIR_TMP_SUFFIX
from utils.sample_group import add_tar_member
from utils.shard_audit import audit_shard, members_by_key, split_member_name

__all__ = [
    "LABELLED",
    "UNCHANGED",
    "SampleInput",
    "SampleLabeler",
    "SampleUpdate",
    "ShardLabelResult",
    "SourceFiles",
    "SourceLabelError",
    "label_shard",
]

LABELLED = "labelled"
UNCHANGED = "unchanged"
NO_LAYOUT = "no key layout"
MISSING_FILE = "missing label file"
NO_IMAGE = "no image member"
_JSON = ".json"
_IMAGE_SUFFIXES = (".webp", ".jpg", ".jpeg", ".png")
_KEPT_PROBLEMS = 20


class SourceLabelError(RuntimeError):
    """The label files do not match the key layout, or a rewritten shard failed verification."""


@dataclass(frozen=True)
class SourceFiles:
    """The label files of each source, in file-id order (sorted by name)."""

    by_source: dict[str, list[Path]]

    @classmethod
    def build(cls, root: Path, layouts: Mapping[str, KeyLayout], suffix: str) -> SourceFiles:
        """Index ``<root>/<files_dir>/**/*<suffix>`` of every layout.

        Raises :class:`SourceLabelError` when a layout has no ``files_dir``,
        its folder is missing, a file name repeats or the file count differs
        from the layout's ``photos``.
        """
        by_source: dict[str, list[Path]] = {}
        for source, layout in layouts.items():
            if not layout.files_dir:
                raise SourceLabelError(f"{layout.source}: the key layout has no files_dir")
            folder = root / layout.files_dir
            if not folder.is_dir():
                raise SourceLabelError(f"{layout.source}: label folder {folder} not found")
            files = sorted(folder.rglob(f"*{suffix}"), key=lambda p: p.name)
            names = {p.name for p in files}
            if len(names) != len(files):
                raise SourceLabelError(f"{layout.source}: a label file name repeats")
            if len(files) != layout.photos:
                raise SourceLabelError(f"{layout.source}: {len(files)} {suffix} label files,"
                                       f" the key layout has {layout.photos} photos")
            by_source[source] = files
        return cls(by_source)

    def file_for(self, source: str, file_id: str) -> Path | None:
        """The label file of photo ``file_id`` of ``source``; None when there is none."""
        files = self.by_source.get(source.lower())
        if files is None or not file_id.isdigit() or int(file_id) >= len(files):
            return None
        return files[int(file_id)]


@dataclass(frozen=True)
class SampleInput:
    """One photo sample: its key, image size, JSON label, member suffixes and label file.

    ``encode`` is False in a dry run: the labeler checks the label file but
    may skip encoding new members (nothing is written).
    """

    key: str
    size: tuple[int, int]
    image: bytes
    meta: dict[str, Any] | None
    suffixes: frozenset[str]
    label_file: Path
    encode: bool = True


@dataclass
class SampleUpdate:
    """What a labeler makes of one sample.

    ``status`` is :data:`LABELLED`, :data:`UNCHANGED` or a problem name
    (``problem`` then says why). ``members`` maps a member suffix to its new
    payload; ``meta`` is the new JSON label; ``metrics`` are numbers the
    plan summarises (for example how far a new box moved).
    """

    status: str
    meta: dict[str, Any] | None = None
    members: dict[str, bytes] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    problem: str = ""


class SampleLabeler(Protocol):
    """Builds the new members and JSON label of one photo from its label file."""

    @property
    def file_suffix(self) -> str:
        """Suffix of the source label files (``.png``, ``.npz``)."""
        ...

    def label(self, sample: SampleInput) -> SampleUpdate:
        """The update of ``sample`` (a problem status when its label file is unusable)."""
        ...

    def info_fields(self) -> dict[str, Any]:
        """dataset_info.yaml fields the labels imply (for example the mask ``classes``)."""
        ...


@dataclass
class ShardLabelResult:
    """What labelling one shard found (and, when written, changed)."""

    shard: Path
    samples: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    problems: dict[str, str] = field(default_factory=dict)
    problem_count: int = 0
    metrics: dict[str, list[float]] = field(default_factory=dict)
    written: bool = False

    @property
    def changed(self) -> bool:
        """True when the shard has samples to label."""
        return self.statuses[LABELLED] > 0

    def add_problem(self, key: str, status: str, reason: str) -> None:
        """Count a problem; keep the first few for the report."""
        self.statuses[status] += 1
        self.problem_count += 1
        if len(self.problems) < _KEPT_PROBLEMS:
            self.problems[key] = f"{status}: {reason}" if reason else status


def _read(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = tf.extractfile(member)
    return handle.read() if handle is not None else b""


def _meta(payload: bytes | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    parsed = json.loads(payload.decode("utf-8"))
    return parsed if isinstance(parsed, dict) else None


def _image_size(image: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image)) as img:
        return img.size


@dataclass(frozen=True)
class _Context:
    labeler: SampleLabeler
    layouts: Mapping[str, KeyLayout]
    files: SourceFiles
    write: bool


def _update(ctx: _Context, key: str, payloads: dict[str, bytes],
            result: ShardLabelResult) -> SampleUpdate | None:
    """The sample's update; None for copies, auxiliary images and problems."""
    role = key_role(key, ctx.layouts)
    if role.role != ROLE_PHOTO:
        result.statuses[role.role] += 1
        return None
    if role.source.lower() not in ctx.layouts:
        result.add_problem(key, NO_LAYOUT, role.source)
        return None
    label_file = ctx.files.file_for(role.source, role.file_id)
    if label_file is None:
        result.add_problem(key, MISSING_FILE, f"file id {role.file_id}")
        return None
    image_suffix = next((s for s in payloads if s in _IMAGE_SUFFIXES), None)
    if image_suffix is None:
        result.add_problem(key, NO_IMAGE, "")
        return None
    image = payloads[image_suffix]
    sample = SampleInput(key=key, size=_image_size(image), image=image,
                         meta=_meta(payloads.get(_JSON)), suffixes=frozenset(payloads),
                         label_file=label_file, encode=ctx.write)
    update = ctx.labeler.label(sample)
    if update.status not in (LABELLED, UNCHANGED):
        result.add_problem(key, update.status, update.problem)
        return None
    result.statuses[update.status] += 1
    for name, value in update.metrics.items():
        result.metrics.setdefault(name, []).append(value)
    return update if update.status == LABELLED else None


def _write_group(dst: tarfile.TarFile, src: tarfile.TarFile, key: str,
                 members: list[tarfile.TarInfo], update: SampleUpdate | None) -> int:
    """Write one sample (updated when ``update`` is given); return its member count."""
    if update is None:
        for member in members:
            dst.addfile(member, src.extractfile(member))
        return len(members)
    replaced = set(update.members) | {_JSON}
    kept = [m for m in members if split_member_name(m.name)[1] not in replaced]
    for member in kept:
        dst.addfile(member, src.extractfile(member))
    for suffix, payload in update.members.items():
        add_tar_member(dst, f"{key}{suffix}", payload)
    add_tar_member(dst, f"{key}{_JSON}",
                   json.dumps(update.meta, separators=(",", ":")).encode("utf-8"))
    return len(kept) + len(update.members) + 1


def _verify(tmp: Path, keys: list[str], members: int, labelled: dict[str, set[str]]) -> None:
    audit = audit_shard(tmp)
    missing = [k for k, want in labelled.items() if not want <= set(audit.suffixes.get(k, []))]
    if audit.error or audit.needs_rewrite or audit.keys != keys or audit.members != members \
            or missing:
        raise SourceLabelError(
            f"{tmp.name}: rewritten shard failed verification (samples {audit.samples} of"
            f" {len(keys)}, members {audit.members} of {members}, labels missing"
            f" {missing[:3]}, error {audit.error})"
        )


def _scan(ctx: _Context, shard: Path, result: ShardLabelResult,
          dst: tarfile.TarFile | None) -> tuple[list[str], int, dict[str, set[str]]]:
    keys: list[str] = []
    written = 0
    labelled: dict[str, set[str]] = {}
    with tarfile.open(shard, "r:") as src:
        for key, members in members_by_key(src):
            keys.append(key)
            result.samples += 1
            payloads = {split_member_name(m.name)[1]: _read(src, m) for m in members}
            update = _update(ctx, key, payloads, result)
            if update is not None:
                labelled[key] = set(update.members) | {_JSON}
            if dst is not None:
                written += _write_group(dst, src, key, members, update)
    return keys, written, labelled


def label_shard(shard: Path, labeler: SampleLabeler, layouts: Mapping[str, KeyLayout],
                files: SourceFiles, write: bool) -> ShardLabelResult:
    """Label the photos of ``shard``; with ``write`` swap in the verified rewrite.

    The shard is written only when it has samples to label and no problem.
    Raises :class:`SourceLabelError` (original untouched) when the rewrite
    fails its verification.
    """
    ctx = _Context(labeler, layouts, files, write)
    result = ShardLabelResult(shard=shard)
    if not write:
        _scan(ctx, shard, result, None)
        return result
    tmp = shard.with_name(shard.name + REPAIR_TMP_SUFFIX)
    try:
        with tarfile.open(tmp, "w") as dst:
            keys, members, labelled = _scan(ctx, shard, result, dst)
        if not result.changed or result.problem_count:
            tmp.unlink()
            return result
        _verify(tmp, keys, members, labelled)
    except (OSError, tarfile.TarError, SourceLabelError):
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, shard)
    result.written = True
    return result
