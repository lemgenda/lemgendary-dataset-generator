"""Write per-file source labels (masks, landmarks) into the samples of WebDataset shards.

A source that ships one label file per image (SFHQ ``segmentations/``,
``landmarks/``) is mapped to the compiled samples by key: the registry's
key layout (:mod:`core.sample_policy`) gives each photo key its source file
id, and the id is the rank of the file name among the source's label files
(the compiler scanned the image folder in sorted name order, and the label
folders hold the same names). Copies and auxiliary images are left alone.

A :data:`KeyResolver` says what each sample key stands for: a
:class:`LayoutFiles` resolver maps a photo key to its label file through
the key layout (copies and auxiliary images are skipped), and
:func:`every_key` takes every key as a sample without a label file (labels
derived from the key and a label table, ``repair class-labels``). Resolvers
are built by a picklable factory in each worker process
(:class:`LayoutFilesFactory`, :func:`every_key_resolver`).

A :class:`SampleLabeler` turns one sample (and its label file, if any) into
the members to add or replace (a ``.mask.webp``) and its new JSON label. Per
shard, :func:`label_shard` reads every sample once and either only reports
(dry run) or writes a temporary shard (other members kept byte for byte,
new members after the image, the JSON last), verifies it and swaps it in
with ``os.replace``; no copy of the original is kept. A shard with a
problem (no layout for a source, a missing or invalid label file) is never
written. A labeler may drop a photo whose label file holds no usable label
(``DROPPED``, for example a failed landmark fit): it is left out of the
rewritten shard and listed in ``dropped``. A sample whose label is already
in place is unchanged, so a re-run finds nothing to do.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from core.sample_policy import ROLE_PHOTO, KeyLayout, key_role
from tools.repair.shard_fix import REPAIR_TMP_SUFFIX
from utils.sample_group import add_tar_member
from utils.shard_audit import audit_shard, members_by_key, split_member_name

__all__ = [
    "DROPPED",
    "LABELLED",
    "UNCHANGED",
    "KeyResolver",
    "KeySource",
    "LayoutFiles",
    "LayoutFilesFactory",
    "SampleInput",
    "SampleLabeler",
    "SampleUpdate",
    "ShardLabelResult",
    "SourceFiles",
    "SourceLabelError",
    "every_key",
    "every_key_resolver",
    "label_shard",
]

LABELLED = "labelled"
UNCHANGED = "unchanged"
DROPPED = "dropped"
NO_LAYOUT = "no key layout"
MISSING_FILE = "missing label file"
NO_IMAGE = "no image member"
_JSON = ".json"
_IMAGE_SUFFIXES = (".webp", ".jpg", ".jpeg", ".png")


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
class KeySource:
    """What a resolver makes of one key: ``skip`` (a role left alone), a
    ``problem`` (and its ``reason``), or a sample with its ``label_file`` (None
    when labels do not come from per-file sources)."""

    skip: str = ""
    problem: str = ""
    reason: str = ""
    label_file: Path | None = None


# What a sample key stands for (its role, a problem, or its label file).
KeyResolver = Callable[[str], KeySource]


@dataclass(frozen=True)
class LayoutFiles:
    """Photo keys of the key layouts, each with its per-file source label."""

    layouts: Mapping[str, KeyLayout]
    files: SourceFiles

    def __call__(self, key: str) -> KeySource:
        """Copies and auxiliary images are skipped; a photo gets its label file."""
        role = key_role(key, self.layouts)
        if role.role != ROLE_PHOTO:
            return KeySource(skip=role.role)
        if role.source.lower() not in self.layouts:
            return KeySource(problem=NO_LAYOUT, reason=role.source)
        label_file = self.files.file_for(role.source, role.file_id)
        if label_file is None:
            return KeySource(problem=MISSING_FILE, reason=f"file id {role.file_id}")
        return KeySource(label_file=label_file)


@dataclass(frozen=True)
class LayoutFilesFactory:
    """Builds a :class:`LayoutFiles` resolver (in each worker process)."""

    root: Path
    layouts: Mapping[str, KeyLayout]
    suffix: str

    def __call__(self) -> LayoutFiles:
        """Index the label files; :class:`SourceLabelError` when they do not match."""
        return LayoutFiles(self.layouts, SourceFiles.build(self.root, self.layouts, self.suffix))


def every_key(_key: str) -> KeySource:
    """Every key is a sample without a label file (labels come from the key)."""
    return KeySource()


def every_key_resolver() -> KeyResolver:
    """The factory of :func:`every_key` (no per-process setup)."""
    return every_key


@dataclass(frozen=True)
class SampleInput:
    """One sample: its key, image size, JSON label, member suffixes and label file.

    ``label_file`` is None when the resolver gives no per-file label.
    ``encode`` is False in a dry run: the labeler checks the label file but
    may skip encoding new members (nothing is written).
    """

    key: str
    size: tuple[int, int]
    image: bytes
    meta: dict[str, Any] | None
    suffixes: frozenset[str]
    label_file: Path | None
    encode: bool = True

    def require_label_file(self) -> Path:
        """The label file; ``FileNotFoundError`` when the sample has none."""
        if self.label_file is None:
            raise FileNotFoundError(f"{self.key}: no source label file")
        return self.label_file


@dataclass
class SampleUpdate:
    """What a labeler makes of one sample.

    ``status`` is :data:`LABELLED`, :data:`UNCHANGED`, :data:`DROPPED` or a
    problem name (``problem`` then says why). ``members`` maps a member suffix to its new
    payload; ``meta`` is the new JSON label; ``metrics`` are numbers the
    plan summarises (for example how far a new box moved); ``tally`` names
    the groups the plan counts the sample in (for example its class).
    """

    status: str
    meta: dict[str, Any] | None = None
    members: dict[str, bytes] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    problem: str = ""
    tally: tuple[str, ...] = ()


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
    statuses: Counter[str] = field(default_factory=Counter)
    problems: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    tallies: Counter[str] = field(default_factory=Counter)
    written: bool = False

    @property
    def samples(self) -> int:
        """Samples read (each has exactly one status)."""
        return sum(self.statuses.values())

    @property
    def changed(self) -> bool:
        """True when the shard has samples to label or drop."""
        return self.statuses[LABELLED] > 0 or bool(self.dropped)

    @property
    def problem_count(self) -> int:
        """Samples with a problem."""
        return len(self.problems)

    def add_problem(self, key: str, status: str, reason: str) -> None:
        """Record a problem sample and why."""
        self.statuses[status] += 1
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
    resolver: KeyResolver
    write: bool


def _sample_input(ctx: _Context, key: str, payloads: dict[str, bytes],
                  result: ShardLabelResult) -> SampleInput | None:
    """The labeler's input for a sample; None for skipped keys and problems."""
    source = ctx.resolver(key)
    if source.skip:
        result.statuses[source.skip] += 1
        return None
    if source.problem:
        result.add_problem(key, source.problem, source.reason)
        return None
    image_suffix = next((s for s in payloads if s in _IMAGE_SUFFIXES), None)
    if image_suffix is None:
        result.add_problem(key, NO_IMAGE, "")
        return None
    image = payloads[image_suffix]
    return SampleInput(key=key, size=_image_size(image), image=image,
                       meta=_meta(payloads.get(_JSON)), suffixes=frozenset(payloads),
                       label_file=source.label_file, encode=ctx.write)


def _update(ctx: _Context, key: str, payloads: dict[str, bytes],
            result: ShardLabelResult) -> SampleUpdate | None:
    """The sample's update; None when it is not labelled, dropped or changed."""
    sample = _sample_input(ctx, key, payloads, result)
    if sample is None:
        return None
    update = ctx.labeler.label(sample)
    if update.status in (LABELLED, UNCHANGED, DROPPED):
        result.tallies.update(update.tally)
    if update.status == DROPPED:
        result.statuses[DROPPED] += 1
        result.dropped[key] = update.problem
        return update
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
    if update is not None and update.status == DROPPED:
        return 0
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
            payloads = {split_member_name(m.name)[1]: _read(src, m) for m in members}
            update = _update(ctx, key, payloads, result)
            if update is None or update.status != DROPPED:
                keys.append(key)
            if update is not None and update.status == LABELLED:
                labelled[key] = set(update.members) | {_JSON}
            if dst is not None:
                written += _write_group(dst, src, key, members, update)
    return keys, written, labelled


def label_shard(shard: Path, labeler: SampleLabeler, resolver: KeyResolver,
                write: bool) -> ShardLabelResult:
    """Label the samples of ``shard``; with ``write`` swap in the verified rewrite.

    The shard is written only when it has samples to label and no problem.
    Raises :class:`SourceLabelError` (original untouched) when the rewrite
    fails its verification.
    """
    ctx = _Context(labeler, resolver, write)
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
    if not keys:
        tmp.unlink()
        shard.unlink()
        result.written = True
        return result
    os.replace(tmp, shard)
    result.written = True
    return result
