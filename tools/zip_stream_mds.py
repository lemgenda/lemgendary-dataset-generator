"""
LemGendary Dataset Compiler — Zip-to-MDS streamer.

Streams paired (images/ + targets/, with optional masks/ and labels/) or
target-only legacy zip archives into MosaicML MDS shards with optional
multi-threaded WebP transcoding. Falls back to WebDataset streaming when
mosaicml-streaming is not installed.
"""

from __future__ import annotations

import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from core.lazy_imports import load_streaming
from tools.zip_stream_common import (
    STREAM_ERRORS,
    MemberEntry,
    SplitCompanions,
    StreamContext,
    ZipStreamOptions,
    encode_samples,
    post_process_zip,
    raw_zip_samples,
    resolve_options,
    sample_entries,
    scan_zip,
    split_output_dir,
)
from tools.zip_stream_webdataset import stream_zip_to_webdataset
from utils.container_stream import (
    open_loose_mds_writer,
    prepare_mds_split_dir,
    resolve_worker_count,
    update_dataset_info_format,
)
from utils.progress import create_progress_bar
from utils.reject_tally import RejectTally
from utils.sample_group import to_mds_record

logger = logging.getLogger(__name__)

__all__ = ["stream_zip_to_mds"]

# Shard size used when mosaicml-streaming is missing and WebDataset is used instead.
_FALLBACK_WEBDATASET_SHARD_SAMPLES = 1000


@dataclass(frozen=True)
class _MdsTarget:
    """MDS writer class, root folder and shard size limit for one conversion."""

    writer_cls: Any
    mds_dir: Path
    size_limit: int


@dataclass(frozen=True)
class _MdsRun:
    """Shared state of one zip-to-MDS conversion."""

    ctx: StreamContext
    mds: _MdsTarget
    rejects: RejectTally


def _stream_mds_split(
    run: _MdsRun, split: str, entries: list[MemberEntry], companions: SplitCompanions
) -> None:
    """Wipe the split folder and write every sample of the split into fresh MDS shards."""
    ctx, mds, rejects = run.ctx, run.mds, run.rejects
    total_samples = len(entries)
    mds_out_str = prepare_mds_split_dir(split_output_dir(mds.mds_dir, split))
    print(f"[STREAM-MDS] Split '{split}': {total_samples} samples -> MDS shards.")

    writer = open_loose_mds_writer(mds.writer_cls, mds_out_str, mds.size_limit)
    with create_progress_bar(total=total_samples, desc=f"MDS [{split}]", unit="img") as pbar:
        for outcome in encode_samples(ctx, raw_zip_samples(ctx.zf, entries, companions)):
            if outcome.sample is None:
                rejects.record(outcome.key, outcome.reason)
            else:
                writer.write(to_mds_record(split, outcome.sample))
            pbar.update(1)
    writer.finish()


def _stream_archive(
    zip_path: Path, target_dir: Path, mds: _MdsTarget, opts: ZipStreamOptions, workers: int
) -> None:
    """Open the zip, extract root metadata and stream every split into MDS."""
    rejects = RejectTally(zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        layout = scan_zip(zf, target_dir, "[STREAM-MDS]")
        print(f"[STREAM-MDS] Extracted {layout.metadata_count} metadata files.")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            ctx = StreamContext(zf=zf, pool=pool, transcode_webp=opts.transcode_webp,
                                label_json=opts.label_json)
            run = _MdsRun(ctx, mds, rejects)
            for split, entries, companions in sample_entries(layout):
                _stream_mds_split(run, split, entries, companions)
    rejects.report()


def _report_mds_output(mds_dir: Path) -> None:
    """Print the MDS output folder and the total size of its shards."""
    total_mds_bytes = sum(s.stat().st_size for s in list(mds_dir.rglob("*.mds")))
    print(f"[VERIFY-MDS] MDS output: {mds_dir} ({total_mds_bytes / (1024**3):.2f} GB).")


def stream_zip_to_mds(
    zip_path: Path,
    target_dir: Path,
    options: ZipStreamOptions | None = None,
    **overrides: Any,
) -> bool:
    """Stream paired or single vision datasets directly from zip into MDS shards.

    Requires mosaicml-streaming >= 0.9.0. Tunables come from `options` (a
    `ZipStreamOptions`); the legacy keyword arguments (`mds_shard_size_bytes`,
    `delete_zip`, `legacy_dir`, `transcode_webp`, `max_workers`) are still
    accepted and override it.
    """
    opts = resolve_options(options, overrides)
    try:
        streaming = load_streaming()
    except ImportError:
        print(
            "[NOTICE] mosaicml-streaming is not installed. Routing directly to WebDataset"
            " streaming..."
        )
        return stream_zip_to_webdataset(
            zip_path=zip_path,
            target_dir=target_dir,
            options=replace(
                opts,
                shard_size_samples=_FALLBACK_WEBDATASET_SHARD_SAMPLES,
                shard_ext=ZipStreamOptions.shard_ext,
            ),
        )

    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    mds_dir = target_dir / "mds"
    mds_dir.mkdir(parents=True, exist_ok=True)

    workers = resolve_worker_count(opts.max_workers)
    print(f"[STREAM-MDS] Opening source archive: {zip_path.name}")
    print(
        f"[STREAM-MDS] Parallel WebP transcoding: "
        f"{'ENABLED' if opts.transcode_webp else 'DISABLED'} ({workers} workers)."
    )
    print(f"[STREAM-MDS] MDS shard size: {opts.mds_shard_size_bytes / (1024**2):.0f} MB")

    try:
        mds = _MdsTarget(streaming.MDSWriter, mds_dir, opts.mds_shard_size_bytes)
        _stream_archive(zip_path, target_dir, mds, opts, workers)

        ds_info_path = target_dir / "dataset_info.yaml"
        if update_dataset_info_format(ds_info_path, "mds", opts.transcode_webp):
            print(f"[METADATA] Updated {ds_info_path.name} format to 'mds'.")

        _report_mds_output(mds_dir)
        post_process_zip(zip_path, opts.legacy_dir, opts.delete_zip)
        return True
    except STREAM_ERRORS as exc:
        logger.exception("Direct zip-to-MDS streaming failed for %s", zip_path)
        print(f"[ERROR] Direct zip-to-MDS streaming failed: {exc}")
        return False
