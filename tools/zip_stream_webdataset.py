"""
LemGendary Dataset Compiler — Zip-to-WebDataset streamer.

Streams paired (images/ + targets/) or target-only legacy zip archives into
WebDataset tar shards with optional multi-threaded WebP transcoding, without
extracting loose files to disk.
"""

from __future__ import annotations

import io
import logging
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tools.zip_stream_common import (
    STREAM_ERRORS,
    MemberEntry,
    StreamContext,
    Worker,
    ZipLayout,
    ZipStreamOptions,
    chunked,
    iter_payloads,
    post_process_zip,
    resolve_options,
    scan_zip,
    split_output_dir,
    worker_img,
    worker_sample,
    worker_tgt,
)
from utils.container_stream import (
    prepare_shards_dir,
    resolve_worker_count,
    update_dataset_info_format,
)
from utils.progress import create_progress_bar

logger = logging.getLogger(__name__)

__all__ = ["stream_zip_to_webdataset"]


@dataclass(frozen=True)
class _SplitShards:
    """Where and how one split's tar shards are written."""

    ctx: StreamContext
    split_dir: Path
    ext: str
    shard_size: int

    def shard_file(self, shard_idx: int) -> Path:
        """Return the tar path of shard `shard_idx`."""
        return self.split_dir / f"shard-{shard_idx:05d}.{self.ext}"


@dataclass(frozen=True)
class _TarPass:
    """One pass over a split's shards: open mode, member suffix and worker."""

    mode: Literal["w", "a"]
    suffix: str
    worker: Worker


_IMAGE_PASS = _TarPass("w", "", worker_img)
_TARGET_PASS = _TarPass("a", ".target", worker_tgt)
_SAMPLE_PASS = _TarPass("w", "", worker_sample)


def _write_tar_shard(
    shards: _SplitShards, shard_idx: int, entries: list[MemberEntry], tar_pass: _TarPass, pbar: Any
) -> None:
    """Write (or append) `entries` into shard `shard_idx`, one tar member each."""
    with tarfile.open(shards.shard_file(shard_idx), tar_pass.mode) as tf:
        for stem, data, out_ext in iter_payloads(shards.ctx, entries, tar_pass.worker):
            ti = tarfile.TarInfo(name=f"{stem}{tar_pass.suffix}{out_ext}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
            pbar.update(1)


def _shard_total(total_samples: int, shard_size: int) -> int:
    return (total_samples + shard_size - 1) // shard_size


def _stream_paired_split(
    shards: _SplitShards,
    split: str,
    img_list: list[MemberEntry],
    target_map: dict[str, tuple[str, zipfile.ZipInfo]],
) -> None:
    """Pass 1 writes image shards; pass 2 appends matching targets to each shard."""
    total_samples = len(img_list)
    total_shards = _shard_total(total_samples, shards.shard_size)
    print(
        f"[STREAM] Split '{split}': {total_samples} images, {len(target_map)}"
        f" targets -> {total_shards} shards."
    )

    with create_progress_bar(
        total=total_samples, desc=f"IMAGES -> WEBP [{split}]", unit="img"
    ) as pbar:
        for shard_idx, chunk in chunked(img_list, shards.shard_size):
            _write_tar_shard(shards, shard_idx, chunk, _IMAGE_PASS, pbar)

    if not target_map:
        return
    with create_progress_bar(
        total=total_samples, desc=f"TARGETS -> WEBP [{split}]", unit="img"
    ) as pbar:
        for shard_idx, chunk in chunked(img_list, shards.shard_size):
            matched = [
                (stem, *target_map[stem]) for stem, _, _ in chunk if stem in target_map
            ]
            if matched:
                _write_tar_shard(shards, shard_idx, matched, _TARGET_PASS, pbar)


def _stream_target_only_split(
    shards: _SplitShards, split: str, target_map: dict[str, tuple[str, zipfile.ZipInfo]]
) -> None:
    """Write a target-only split (e.g. UpnV2 targets/train) as plain sample shards."""
    tgt_items: list[MemberEntry] = [
        (stem, ext, zinfo) for stem, (ext, zinfo) in target_map.items()
    ]
    total_samples = len(tgt_items)
    total_shards = _shard_total(total_samples, shards.shard_size)
    print(
        f"[STREAM] Split '{split}': {total_samples} target samples ->"
        f" {total_shards} shards."
    )
    with create_progress_bar(
        total=total_samples, desc=f"SAMPLES -> WEBP [{split}]", unit="img"
    ) as pbar:
        for shard_idx, chunk in chunked(tgt_items, shards.shard_size):
            _write_tar_shard(shards, shard_idx, chunk, _SAMPLE_PASS, pbar)


def _open_split_shards(
    ctx: StreamContext, shards_dir: Path, split: str, opts: ZipStreamOptions
) -> _SplitShards:
    """Create the split's shard folder and return its shard writer settings."""
    split_dir = split_output_dir(shards_dir, split)
    split_dir.mkdir(parents=True, exist_ok=True)
    return _SplitShards(ctx, split_dir, opts.shard_ext.lstrip("."), opts.shard_size_samples)


def _stream_layout(
    ctx: StreamContext, layout: ZipLayout, shards_dir: Path, opts: ZipStreamOptions
) -> None:
    """Stream every split of the partitioned archive into tar shards."""
    # Case A: Paired dataset (images/ and targets/)
    if layout.has_images_dir:
        for split, img_list in layout.images_by_split.items():
            _stream_paired_split(
                _open_split_shards(ctx, shards_dir, split, opts),
                split,
                img_list,
                layout.targets_by_split.get(split, {}),
            )
        return
    # Case B: Target-only dataset (targets/train, targets/val, e.g. UpnV2)
    for split, target_map in layout.targets_by_split.items():
        _stream_target_only_split(
            _open_split_shards(ctx, shards_dir, split, opts), split, target_map
        )


def _stream_archive(
    zip_path: Path, target_dir: Path, shards_dir: Path, opts: ZipStreamOptions, workers: int
) -> None:
    """Open the zip, extract root metadata and stream every split into shards."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        layout = scan_zip(zf, target_dir, "[STREAM]")
        print(
            f"[STREAM] Extracted {layout.metadata_count} root metadata/notebook files to"
            f" {target_dir.name}."
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            ctx = StreamContext(zf=zf, pool=pool, transcode_webp=opts.transcode_webp)
            _stream_layout(ctx, layout, shards_dir, opts)


def _verify_shards(shards_dir: Path, ext: str) -> bool:
    """Print the shard count/size and return False when nothing was written."""
    shards_created = (
        list(shards_dir.glob(f"**/*.{ext}"))
        or list(shards_dir.glob("**/*.tar"))
        or list(shards_dir.glob("**/*.data"))
    )
    total_shard_bytes = sum(s.stat().st_size for s in shards_created)
    print(
        f"[VERIFY] Created {len(shards_created)} shards ({total_shard_bytes / (1024**3):.2f} GB"
        " total)."
    )
    if not shards_created or total_shard_bytes == 0:
        print("[ERROR] Shard verification failed: 0 shards or 0 bytes created.")
        return False
    return True


def stream_zip_to_webdataset(
    zip_path: Path,
    target_dir: Path,
    options: ZipStreamOptions | None = None,
    **overrides: Any,
) -> bool:
    """Stream paired or single vision datasets directly from zip into WebDataset tar shards.

    Tunables come from `options` (a `ZipStreamOptions`); the legacy keyword
    arguments (`shard_size_samples`, `delete_zip`, `legacy_dir`, `transcode_webp`,
    `max_workers`, `shard_ext`) are still accepted and override it.
    """
    opts = resolve_options(options, overrides)
    target_dir = target_dir.resolve()
    shards_dir = prepare_shards_dir(target_dir)
    workers = resolve_worker_count(opts.max_workers)
    print(f"[STREAM] Opening source archive: {zip_path.name}")
    print(
        f"[STREAM] Parallel WebP transcoding: {'ENABLED' if opts.transcode_webp else 'DISABLED'}"
        f" ({workers} workers)."
    )

    try:
        _stream_archive(zip_path, target_dir, shards_dir, opts, workers)

        ds_info_path = target_dir / "dataset_info.yaml"
        if update_dataset_info_format(ds_info_path, "webdataset", opts.transcode_webp):
            print(
                f"[METADATA] Updated {ds_info_path.name} format to 'webdataset' (image_format:"
                " webp)."
            )

        if not _verify_shards(shards_dir, opts.shard_ext.lstrip(".")):
            return False

        post_process_zip(zip_path, opts.legacy_dir, opts.delete_zip)
        return True
    except STREAM_ERRORS as exc:
        logger.exception("Direct zip-to-WebDataset streaming failed for %s", zip_path)
        print(f"[ERROR] Direct zip-to-WebDataset streaming failed: {exc}")
        return False
