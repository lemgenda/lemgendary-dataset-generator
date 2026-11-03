"""
LemGendary Dataset Compiler — Zip streaming: options, member layout, cleanup.

Shared pieces of the zip-to-container streamers (`tools.zip_stream_webdataset`,
`tools.zip_stream_mds`): the options object, zip member partitioning into
images/targets/metadata, metadata extraction, transcode workers, payload
iteration and post-conversion cleanup.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from utils.container_stream import (
    IMAGE_WEBP_QUALITY,
    TARGET_WEBP_QUALITY,
    transcode_to_webp,
)

__all__ = [
    "IMAGE_EXTS",
    "STREAM_ERRORS",
    "MemberEntry",
    "StreamContext",
    "Worker",
    "ZipLayout",
    "ZipStreamOptions",
    "chunked",
    "extract_metadata_files",
    "iter_payloads",
    "partition_zip_members",
    "post_process_zip",
    "resolve_options",
    "scan_zip",
    "split_output_dir",
    "worker_img",
    "worker_sample",
    "worker_tgt",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_METADATA_EXTS = (".yaml", ".yml", ".md", ".json", ".txt", ".ipynb")
_LAYOUT_DIRS = ("images", "targets", "masks")

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
Payload = tuple[str, bytes, str]
Worker = Callable[[Payload], Payload]
ImagesBySplit = dict[str, list[MemberEntry]]
TargetsBySplit = dict[str, dict[str, tuple[str, zipfile.ZipInfo]]]


@dataclass(frozen=True)
class ZipStreamOptions:
    """Tunables shared by the WebDataset and MDS zip streamers."""

    shard_size_samples: int = 5000
    shard_ext: str = "data"
    mds_shard_size_bytes: int = 512 * 1024 * 1024
    delete_zip: bool = False
    legacy_dir: Path | None = None
    transcode_webp: bool = True
    max_workers: int | None = None


def resolve_options(
    options: ZipStreamOptions | None, overrides: dict[str, Any]
) -> ZipStreamOptions:
    """Merge legacy keyword arguments over `options` (unknown keys raise TypeError)."""
    base = options if options is not None else ZipStreamOptions()
    return replace(base, **overrides) if overrides else base


@dataclass(frozen=True)
class StreamContext:
    """Open archive + worker pool shared by every split of one conversion."""

    zf: zipfile.ZipFile
    pool: ThreadPoolExecutor
    transcode_webp: bool


def worker_img(item: Payload) -> Payload:
    """Transcode an image payload to WebP at image quality."""
    s, raw, e = item
    out_bytes, out_ext = transcode_to_webp(raw, e, quality=IMAGE_WEBP_QUALITY)
    return s, out_bytes, out_ext


def worker_tgt(item: Payload) -> Payload:
    """Transcode a target payload to WebP at target quality."""
    s, raw, e = item
    out_bytes, out_ext = transcode_to_webp(raw, e, quality=TARGET_WEBP_QUALITY)
    return s, out_bytes, out_ext


def worker_sample(item: Payload) -> Payload:
    """Transcode a target-only sample payload to WebP at target quality."""
    s, raw, e = item
    out_bytes, out_ext = transcode_to_webp(raw, e, quality=TARGET_WEBP_QUALITY)
    return s, out_bytes, out_ext


def iter_payloads(
    ctx: StreamContext, entries: list[MemberEntry], worker: Worker
) -> Iterator[Payload]:
    """Yield (stem, bytes, ext) per entry, transcoded in the pool when enabled."""
    raw_items = ((stem, ctx.zf.read(zinfo), ext) for stem, ext, zinfo in entries)
    if ctx.transcode_webp:
        return iter(ctx.pool.map(worker, raw_items, chunksize=8))
    return raw_items


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


def _image_stem_ext(fname: str) -> tuple[str, str] | None:
    """Return (stem, lower-case ext) for an image file name, else None."""
    dot_pos = fname.rfind(".")
    if dot_pos <= 0:
        return None
    ext = fname[dot_pos:].lower()
    if ext not in IMAGE_EXTS:
        return None
    return fname[:dot_pos], ext


def partition_zip_members(
    zf: zipfile.ZipFile,
) -> tuple[ImagesBySplit, TargetsBySplit, bool, list[zipfile.ZipInfo]]:
    """Partition zip members into images, targets, and metadata files.

    Returns (images_by_split, targets_by_split, has_images_dir, metadata_members).
    """
    images_by_split: ImagesBySplit = {}
    targets_by_split: TargetsBySplit = {}
    metadata_members: list[zipfile.ZipInfo] = []
    has_images_dir = False

    for m in zf.infolist():
        parts = [p for p in m.filename.replace("\\", "/").split("/") if p]
        if _is_root_metadata(m, parts):
            metadata_members.append(m)
            continue
        if m.is_dir():
            continue
        stem_ext = _image_stem_ext(parts[-1])
        if stem_ext is None:
            continue
        stem, ext = stem_ext
        lower_parts = [p.lower() for p in parts]
        split = _split_of(lower_parts)
        if "images" in lower_parts:
            has_images_dir = True
            images_by_split.setdefault(split, []).append((stem, ext, m))
        elif "targets" in lower_parts:
            targets_by_split.setdefault(split, {})[stem] = (ext, m)

    return images_by_split, targets_by_split, has_images_dir, metadata_members


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

    images_by_split: ImagesBySplit
    targets_by_split: TargetsBySplit
    has_images_dir: bool
    metadata_count: int


def scan_zip(zf: zipfile.ZipFile, target_dir: Path, tag: str) -> ZipLayout:
    """Print the entry count, partition the members and extract root metadata."""
    print(f"{tag} Discovered {len(zf.infolist())} entries inside archive.")
    images_by_split, targets_by_split, has_images_dir, metadata_members = (
        partition_zip_members(zf)
    )
    count = extract_metadata_files(zf, metadata_members, target_dir)
    return ZipLayout(images_by_split, targets_by_split, has_images_dir, count)


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
                shutil.rmtree(legacy_resolved, ignore_errors=True)
                print(f"[SUCCESS] Deleted legacy directory {legacy_resolved.name} successfully.")
            except OSError as ex:
                print(f"[WARN] Could not remove legacy directory {legacy_resolved}: {ex}")
