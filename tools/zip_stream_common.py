"""
LemGendary Dataset Compiler — Zip streaming: options, member layout, cleanup.

Shared pieces of the zip-to-container streamers (`tools.zip_stream_webdataset`,
`tools.zip_stream_mds`): the options object, zip member partitioning into
images/targets/masks/labels/metadata, metadata extraction, per-sample
assembly and encoding, and post-conversion cleanup.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from utils.container_stream import IMAGE_WEBP_QUALITY, TARGET_WEBP_QUALITY
from utils.sample_group import (
    LABEL_EXTS,
    RawMember,
    RawSample,
    SampleOutcome,
    encode_sample_or_reject,
)

__all__ = [
    "IMAGE_EXTS",
    "STREAM_ERRORS",
    "MemberEntry",
    "SampleEncoding",
    "SplitCompanions",
    "StreamContext",
    "ZipLayout",
    "ZipMembers",
    "ZipStreamOptions",
    "chunked",
    "encode_samples",
    "extract_metadata_files",
    "partition_zip_members",
    "post_process_zip",
    "raw_zip_samples",
    "resolve_options",
    "sample_entries",
    "scan_zip",
    "split_output_dir",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_METADATA_EXTS = (".yaml", ".yml", ".md", ".json", ".txt", ".ipynb")
_LAYOUT_DIRS = ("images", "targets", "masks", "labels")

# Errors that abort a streaming conversion (reported, then False is returned).
STREAM_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    KeyError,
    RuntimeError,
    ImportError,
    zipfile.BadZipFile,
    tarfile.TarError,
)

# (stem, extension, zip member)
MemberEntry = tuple[str, str, zipfile.ZipInfo]
ImagesBySplit = dict[str, list[MemberEntry]]
StemMap = dict[str, tuple[str, zipfile.ZipInfo]]
StemMapsBySplit = dict[str, StemMap]


@dataclass(frozen=True)
class SampleEncoding:
    """How members are encoded: WebP transcoding and the label JSON strategy."""

    transcode_webp: bool = True
    label_json: str | None = None  # utils.label_json strategy for text labels


_ENCODING_KEYS = ("transcode_webp", "label_json")


@dataclass(frozen=True)
class ZipStreamOptions:
    """Tunables shared by the WebDataset and MDS zip streamers."""

    shard_size_samples: int = 5000
    shard_ext: str = "data"
    mds_shard_size_bytes: int = 512 * 1024 * 1024
    delete_zip: bool = False
    legacy_dir: Path | None = None
    max_workers: int | None = None
    encoding: SampleEncoding = field(default_factory=SampleEncoding)

    @property
    def transcode_webp(self) -> bool:
        """True when images are transcoded to WebP."""
        return self.encoding.transcode_webp

    @property
    def label_json(self) -> str | None:
        """The JSON sidecar strategy for text labels, if any."""
        return self.encoding.label_json


def resolve_options(
    options: ZipStreamOptions | None, overrides: dict[str, Any]
) -> ZipStreamOptions:
    """Merge legacy keyword arguments over `options` (unknown keys raise TypeError).

    ``transcode_webp`` and ``label_json`` go into ``options.encoding``.
    """
    base = options if options is not None else ZipStreamOptions()
    rest = dict(overrides)
    encoding = {key: rest.pop(key) for key in _ENCODING_KEYS if key in rest}
    if encoding:
        base = replace(base, encoding=replace(base.encoding, **encoding))
    return replace(base, **rest) if rest else base


@dataclass(frozen=True)
class StreamContext:
    """Open archive + worker pool shared by every split of one conversion."""

    zf: zipfile.ZipFile
    pool: ThreadPoolExecutor
    transcode_webp: bool
    label_json: str | None = None


@dataclass(frozen=True)
class ZipMembers:
    """Zip members partitioned by role and split."""

    images_by_split: ImagesBySplit = field(default_factory=dict)
    targets_by_split: StemMapsBySplit = field(default_factory=dict)
    masks_by_split: StemMapsBySplit = field(default_factory=dict)
    labels_by_split: StemMapsBySplit = field(default_factory=dict)
    has_images_dir: bool = False
    metadata_members: list[zipfile.ZipInfo] = field(default_factory=list)


@dataclass(frozen=True)
class SplitCompanions:
    """Members that travel with each primary member of one split."""

    targets: StemMap = field(default_factory=dict)
    masks: StemMap = field(default_factory=dict)
    labels: StemMap = field(default_factory=dict)
    primary_quality: int = IMAGE_WEBP_QUALITY


def _read(zf: zipfile.ZipFile, entry: tuple[str, zipfile.ZipInfo] | None) -> RawMember | None:
    if entry is None:
        return None
    ext, zinfo = entry
    return RawMember(zf.read(zinfo), ext)


def raw_zip_samples(
    zf: zipfile.ZipFile, entries: list[MemberEntry], companions: SplitCompanions
) -> Iterator[RawSample]:
    """Read each sample's primary member and its target/mask/label companions."""
    for stem, ext, zinfo in entries:
        yield RawSample(
            key=stem,
            primary=RawMember(zf.read(zinfo), ext),
            target=_read(zf, companions.targets.get(stem)),
            mask=_read(zf, companions.masks.get(stem)),
            label=_read(zf, companions.labels.get(stem)),
            primary_quality=companions.primary_quality,
        )


def encode_samples(ctx: StreamContext, raws: Iterator[RawSample]) -> Iterator[SampleOutcome]:
    """Validate and encode every sample in the pool; corrupt ones come back rejected.

    ``ctx.label_json`` names the JSON sidecar strategy applied to text labels.
    """
    transcode, label_json = ctx.transcode_webp, ctx.label_json

    def _encode(raw: RawSample) -> SampleOutcome:
        return encode_sample_or_reject(replace(raw, label_json=label_json), transcode)

    return iter(ctx.pool.map(_encode, raws, chunksize=8))


def chunked(items: list[MemberEntry], size: int) -> Iterator[tuple[int, list[MemberEntry]]]:
    """Yield (shard_index, slice) pairs of at most `size` items."""
    for shard_idx, start in enumerate(range(0, len(items), size)):
        yield shard_idx, items[start:start + size]


def split_output_dir(base: Path, split: str) -> Path:
    """Return `base` for the pseudo-split 'all', otherwise `base / split`."""
    return base if split == "all" else base / split


def _is_root_metadata(member: zipfile.ZipInfo, parts: list[str]) -> bool:
    """True for root-level (or one-folder-deep, non-layout) metadata documents."""
    at_root = len(parts) == 1 or (
        len(parts) == 2 and not member.is_dir() and parts[0].lower() not in _LAYOUT_DIRS
    )
    return at_root and any(parts[-1].endswith(ext) for ext in _METADATA_EXTS)


def _split_of(lower_parts: list[str]) -> str:
    """Pick the split name from path components (val > train > test > all)."""
    for split in ("val", "train", "test"):
        if split in lower_parts:
            return split
    return "all"


def _stem_ext(fname: str, exts: set[str] | tuple[str, ...]) -> tuple[str, str] | None:
    """Return (stem, lower-case ext) when ``fname`` has one of ``exts``, else None."""
    dot_pos = fname.rfind(".")
    if dot_pos <= 0:
        return None
    ext = fname[dot_pos:].lower()
    if ext not in exts:
        return None
    return fname[:dot_pos], ext


def _role_of(lower_parts: list[str]) -> str | None:
    """Return the layout folder (images/targets/masks/labels) a member sits under."""
    for role in _LAYOUT_DIRS:
        if role in lower_parts[:-1]:
            return role
    return None


def partition_zip_members(zf: zipfile.ZipFile) -> ZipMembers:
    """Partition zip members into images, targets, masks, labels and metadata files."""
    members = ZipMembers()
    has_images_dir = False
    by_role: dict[str, StemMapsBySplit] = {
        "targets": members.targets_by_split,
        "masks": members.masks_by_split,
        "labels": members.labels_by_split,
    }
    for m in zf.infolist():
        parts = [p for p in m.filename.replace("\\", "/").split("/") if p]
        if _is_root_metadata(m, parts):
            members.metadata_members.append(m)
            continue
        if m.is_dir():
            continue
        lower_parts = [p.lower() for p in parts]
        role = _role_of(lower_parts)
        if role is None:
            continue
        stem_ext = _stem_ext(parts[-1], LABEL_EXTS if role == "labels" else IMAGE_EXTS)
        if stem_ext is None:
            continue
        stem, ext = stem_ext
        split = _split_of(lower_parts)
        if role == "images":
            has_images_dir = True
            members.images_by_split.setdefault(split, []).append((stem, ext, m))
        else:
            by_role[role].setdefault(split, {})[stem] = (ext, m)
    return replace(members, has_images_dir=has_images_dir)


def extract_metadata_files(
    zf: zipfile.ZipFile,
    metadata_members: list[zipfile.ZipInfo],
    target_dir: Path,
) -> int:
    """Extract root metadata files from the zip to target_dir."""
    count = 0
    for m in metadata_members:
        norm_name = m.filename.replace("\\", "/")
        base_name = norm_name.split("/")[-1]
        dest_file = target_dir / base_name
        print(f"  -> Extracting metadata: {base_name} ({m.file_size:,} bytes)")
        with zf.open(m) as src, open(dest_file, "wb") as dst:
            shutil.copyfileobj(src, dst)
        count += 1
    return count


@dataclass(frozen=True)
class ZipLayout:
    """Partitioned zip contents plus the number of metadata files extracted."""

    members: ZipMembers
    metadata_count: int


def scan_zip(zf: zipfile.ZipFile, target_dir: Path, tag: str) -> ZipLayout:
    """Print the entry count, partition the members and extract root metadata."""
    print(f"{tag} Discovered {len(zf.infolist())} entries inside archive.")
    members = partition_zip_members(zf)
    count = extract_metadata_files(zf, members.metadata_members, target_dir)
    return ZipLayout(members, count)


def sample_entries(layout: ZipLayout) -> list[tuple[str, list[MemberEntry], SplitCompanions]]:
    """Return (split, primary entries, companions) for every split of ``layout``.

    With an ``images/`` folder the images are primary and targets, masks and
    labels are companions; otherwise (e.g. UpnV2) the targets are primary.
    """
    members = layout.members
    if members.has_images_dir:
        return [
            (
                split,
                img_list,
                SplitCompanions(
                    targets=members.targets_by_split.get(split, {}),
                    masks=members.masks_by_split.get(split, {}),
                    labels=members.labels_by_split.get(split, {}),
                ),
            )
            for split, img_list in members.images_by_split.items()
        ]
    return [
        (
            split,
            [(stem, ext, zinfo) for stem, (ext, zinfo) in target_map.items()],
            SplitCompanions(
                masks=members.masks_by_split.get(split, {}),
                labels=members.labels_by_split.get(split, {}),
                primary_quality=TARGET_WEBP_QUALITY,
            ),
        )
        for split, target_map in members.targets_by_split.items()
    ]


def post_process_zip(zip_path: Path, legacy_dir: Path | None, delete_zip: bool) -> None:
    """Optionally delete source zip and legacy directory."""
    if delete_zip:
        print(f"[CLEANUP] Deleting source archive to free local storage: {zip_path.name}")
        try:
            os.remove(zip_path)
            print(f"[SUCCESS] Deleted {zip_path.name} successfully.")
        except OSError as ex:
            print(f"[WARN] Could not remove zip archive {zip_path}: {ex}")

    if legacy_dir is not None:
        legacy_resolved = legacy_dir.resolve()
        if legacy_resolved.exists():
            print(f"[CLEANUP] Deleting legacy manifold directory: {legacy_resolved.name}")
            try:
                shutil.rmtree(legacy_resolved)
                print(f"[SUCCESS] Deleted legacy directory {legacy_resolved.name} successfully.")
            except OSError as ex:
                print(f"[WARN] Could not remove legacy directory {legacy_resolved}: {ex}")
