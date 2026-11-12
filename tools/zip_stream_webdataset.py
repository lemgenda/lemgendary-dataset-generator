"""
LemGendary Dataset Compiler — Zip-to-WebDataset streamer.

Streams paired (images/ + targets/, with optional masks/ and labels/) or
target-only legacy zip archives into WebDataset tar shards with optional
multi-threaded WebP transcoding, without extracting loose files to disk. Each
sample's members are written contiguously in a single pass and corrupt images
are rejected instead of packed.
"""

from __future__ import annotations

import logging
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.repair.sample_policy_fix import enforce_sample_policy
from tools.zip_stream_common import (
    STREAM_ERRORS,
    MemberEntry,
    SplitCompanions,
    StreamContext,
    ZipLayout,
    ZipStreamOptions,
    chunked,
    encode_samples,
    post_process_zip,
    raw_zip_samples,
    resolve_options,
    sample_entries,
    scan_zip,
    split_output_dir,
)
from utils.container_stream import (
    prepare_shards_dir,
    resolve_worker_count,
    update_dataset_info_format,
)
from utils.progress import create_progress_bar
from utils.reject_tally import RejectTally
from utils.sample_group import write_sample
from utils.shard_audit import verify_shard_set

logger = logging.getLogger(__name__)

__all__ = ["stream_zip_to_webdataset"]


@dataclass(frozen=True)
class _SplitShards:
    """Where and how one split's tar shards are written."""

    ctx: StreamContext
    split_dir: Path
    ext: str
    shard_size: int
    rejects: RejectTally

    def shard_file(self, shard_idx: int) -> Path:
        """Return the tar path of shard `shard_idx`."""
        return self.split_dir / f"shard-{shard_idx:05d}.{self.ext}"


def _write_tar_shard(
    shards: _SplitShards,
    shard_idx: int,
    entries: list[MemberEntry],
    companions: SplitCompanions,
    pbar: Any,
) -> None:
    """Write shard `shard_idx` in one pass, each sample's members back to back."""
    raws = raw_zip_samples(shards.ctx.zf, entries, companions)
    with tarfile.open(shards.shard_file(shard_idx), "w") as tf:
        for outcome in encode_samples(shards.ctx, raws):
            if outcome.sample is None:
                shards.rejects.record(outcome.key, outcome.reason)
            else:
                write_sample(tf, outcome.sample)
            pbar.update(1)


def _stream_split(
    shards: _SplitShards, split: str, entries: list[MemberEntry], companions: SplitCompanions
) -> None:
    """Write every sample of one split (with its target/mask/label) into tar shards."""
    total_samples = len(entries)
    total_shards = (total_samples + shards.shard_size - 1) // shards.shard_size
    print(
        f"[STREAM] Split '{split}': {total_samples} samples, {len(companions.targets)}"
        f" targets, {len(companions.masks)} masks, {len(companions.labels)} labels ->"
        f" {total_shards} shards."
    )
    with create_progress_bar(
        total=total_samples, desc=f"SAMPLES -> SHARDS [{split}]", unit="img"
    ) as pbar:
        for shard_idx, chunk in chunked(entries, shards.shard_size):
            _write_tar_shard(shards, shard_idx, chunk, companions, pbar)


def _open_split_shards(
    ctx: StreamContext, shards_dir: Path, split: str, opts: ZipStreamOptions,
    rejects: RejectTally,
) -> _SplitShards:
    """Create the split's shard folder and return its shard writer settings."""
    split_dir = split_output_dir(shards_dir, split)
    split_dir.mkdir(parents=True, exist_ok=True)
    return _SplitShards(
        ctx, split_dir, opts.shard_ext.lstrip("."), opts.shard_size_samples, rejects
    )


def _stream_layout(
    ctx: StreamContext, layout: ZipLayout, shards_dir: Path, opts: ZipStreamOptions,
    rejects: RejectTally,
) -> None:
    """Stream every split of the partitioned archive into tar shards."""
    for split, entries, companions in sample_entries(layout):
        _stream_split(
            _open_split_shards(ctx, shards_dir, split, opts, rejects), split, entries, companions
        )


def _stream_archive(
    zip_path: Path, target_dir: Path, shards_dir: Path, opts: ZipStreamOptions, workers: int
) -> None:
    """Open the zip, extract root metadata and stream every split into shards."""
    rejects = RejectTally(zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        layout = scan_zip(zf, target_dir, "[STREAM]")
        print(
            f"[STREAM] Extracted {layout.metadata_count} root metadata/notebook files to"
            f" {target_dir.name}."
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            ctx = StreamContext(zf=zf, pool=pool, transcode_webp=opts.transcode_webp,
                                label_json=opts.label_json)
            _stream_layout(ctx, layout, shards_dir, opts, rejects)
    rejects.report()


def _verify_shards(shards_dir: Path, ext: str) -> bool:
    """Print the shard count/size; False when nothing was written or a shard is unsound
    (zero-byte members, scattered sample groups, unreadable tar)."""
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
    if not verify_shard_set(shards_dir, "VERIFY"):
        print("[ERROR] Shard verification failed: the shards above are unsound.")
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
        if not enforce_sample_policy(target_dir):
            return False

        post_process_zip(zip_path, opts.legacy_dir, opts.delete_zip)
        return True
    except STREAM_ERRORS as exc:
        logger.exception("Direct zip-to-WebDataset streaming failed for %s", zip_path)
        print(f"[ERROR] Direct zip-to-WebDataset streaming failed: {exc}")
        return False
