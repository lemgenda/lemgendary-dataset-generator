"""Pack a loose images/targets directory into WebDataset tar or MDS shards."""

from __future__ import annotations

import io
import tarfile
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from core.lazy_imports import load_streaming
from tools.modernize.common import (
    LooseSplit,
    encode_image,
    loose_split_names,
    purge_loose_dirs,
    scan_loose_split,
)
from utils.container_stream import (
    loose_mds_record,
    open_loose_mds_writer,
    prepare_mds_split_dir,
    prepare_shards_dir,
    resolve_worker_count,
    update_dataset_info_format,
)

# WebP quality of primary images and of restoration targets.
IMAGE_QUALITY = 92
TARGET_QUALITY = 95


@dataclass(frozen=True)
class LooseConvertOptions:
    """Options shared by the loose-directory converters."""

    shard_size_samples: int = 5000
    mds_shard_size_bytes: int = 512 * 1024 * 1024
    transcode_webp: bool = True
    max_workers: int | None = None
    purge_loose: bool = False
    shard_ext: str = "data"


def _resolve_options(
    options: LooseConvertOptions | None, overrides: dict[str, Any]
) -> LooseConvertOptions:
    return replace(options or LooseConvertOptions(), **overrides)


def _stamp_format(target_dir: Path, fmt: str, transcode_webp: bool) -> None:
    """Record the container format in dataset_info.yaml and report it."""
    info_path = target_dir / "dataset_info.yaml"
    if update_dataset_info_format(info_path, fmt, transcode_webp):
        print(f"[METADATA] Updated {info_path.name} format to {fmt}.")


def _add_member(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    ti = tarfile.TarInfo(name=name)
    ti.size = len(payload)
    tf.addfile(ti, io.BytesIO(payload))


def _write_tar_shard(
    shard_path: Path,
    encoded: list[tuple[str, bytes, str]],
    companions: dict[str, Path],
    transcode_webp: bool,
) -> None:
    """Write one tar shard: each encoded sample plus its ``.target`` companion, if any."""
    with tarfile.open(shard_path, "w") as tf:
        for stem, data, member_ext in encoded:
            _add_member(tf, f"{stem}{member_ext}", data)
            if stem in companions:
                _, t_bytes, t_ext = encode_image(companions[stem], TARGET_QUALITY, transcode_webp)
                _add_member(tf, f"{stem}.target{t_ext}", t_bytes)


def _write_tar_split(
    split: LooseSplit, out_dir: Path, pool: Executor, opts: LooseConvertOptions
) -> None:
    """Write one split as WebDataset tar shards (images + targets, or targets only)."""
    if split.images:
        primary, quality, companions = split.images, IMAGE_QUALITY, split.targets
        desc = f"SHARDS [{split.name}]"
    else:
        # Target-only split: the targets are the primary members.
        primary, quality, companions = list(split.targets.values()), TARGET_QUALITY, {}
        desc = f"TARGET SHARDS [{split.name}]"
    size = opts.shard_size_samples
    total_shards = (split.total_samples + size - 1) // size
    print(
        f"[CONVERT] Split '{split.name}': {split.total_samples} samples -> {total_shards} shards."
    )

    def _encode(p: Path) -> tuple[str, bytes, str]:
        return encode_image(p, quality, opts.transcode_webp)

    ext = opts.shard_ext.lstrip(".")
    with tqdm(total=split.total_samples, desc=desc, unit="img", dynamic_ncols=True) as pbar:
        for s_idx in range(total_shards):
            chunk = primary[s_idx * size:(s_idx + 1) * size]
            _write_tar_shard(
                out_dir / f"shard-{s_idx:05d}.{ext}",
                list(pool.map(_encode, chunk)),
                companions,
                opts.transcode_webp,
            )
            pbar.update(len(chunk))


def convert_loose_directory_to_webdataset(
    source_dir: Path,
    target_dir: Path,
    options: LooseConvertOptions | None = None,
    **overrides: Any,
) -> bool:
    """Convert a directory containing loose images/targets into WebDataset shards.

    ``options`` (or keyword ``overrides`` of its fields: shard_size_samples,
    transcode_webp, max_workers, purge_loose, shard_ext) configure the run.
    """
    opts = _resolve_options(options, overrides)
    shards_dir = prepare_shards_dir(target_dir)
    workers = resolve_worker_count(opts.max_workers)

    print(
        f"[CONVERT] Packaging loose directory {source_dir.name} into WebDataset shards ({workers}"
        " workers)..."
    )
    for split_name in loose_split_names(source_dir):
        split = scan_loose_split(source_dir, split_name)
        out_dir = split.out_dir(shards_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if split.total_samples == 0:
            continue
        with ThreadPoolExecutor(max_workers=workers) as pool:
            _write_tar_split(split, out_dir, pool, opts)

    _stamp_format(target_dir, "webdataset", opts.transcode_webp)

    if opts.purge_loose:
        purge_loose_dirs(source_dir, "Purging loose image directories to reclaim storage...")
    return True


def _write_mds_split(
    split: LooseSplit,
    split_mds_dir: Path,
    pool: Executor,
    opts: LooseConvertOptions,
    mds_writer_cls: Any,
) -> None:
    """Write one split of a loose directory as zstd-compressed MDS shards."""
    print(f"[CONVERT-MDS] Split '{split.name}': {split.total_samples} samples -> MDS.")
    writer = open_loose_mds_writer(
        mds_writer_cls, prepare_mds_split_dir(split_mds_dir), opts.mds_shard_size_bytes
    )

    def _encode(p: Path) -> tuple[str, bytes, str]:
        return encode_image(p, IMAGE_QUALITY, opts.transcode_webp)

    samples = split.images if split.images else list(split.targets.values())
    with tqdm(
        total=split.total_samples, desc=f"MDS [{split.name}]", unit="img", dynamic_ncols=True
    ) as pbar:
        for stem, img_data, _ in pool.map(_encode, samples):
            tgt_bytes = b""
            if stem in split.targets:
                tgt_path = split.targets[stem]
                if opts.transcode_webp:
                    _, tgt_bytes, _ = encode_image(tgt_path, TARGET_QUALITY, True)
                else:
                    tgt_bytes = tgt_path.read_bytes()
            writer.write(loose_mds_record(split.name, stem, img_data, tgt_bytes))
            pbar.update(1)
    writer.finish()


def convert_loose_directory_to_mds(
    source_dir: Path,
    target_dir: Path,
    options: LooseConvertOptions | None = None,
    **overrides: Any,
) -> bool:
    """Convert a loose images/targets directory into MDS shards.

    ``options`` (or keyword ``overrides`` of its fields: mds_shard_size_bytes,
    transcode_webp, max_workers, purge_loose) configure the run. Without
    mosaicml-streaming the directory is packed as WebDataset (1000-sample shards).
    """
    opts = _resolve_options(options, overrides)
    workers = resolve_worker_count(opts.max_workers)
    try:
        streaming = load_streaming()
    except ImportError:
        print(
            "[NOTICE] mosaicml-streaming is not installed. Routing loose directory directly to"
            " WebDataset shards..."
        )
        return convert_loose_directory_to_webdataset(
            source_dir,
            target_dir,
            replace(opts, shard_size_samples=1000, max_workers=workers, shard_ext="data"),
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    mds_dir = target_dir / "mds"
    mds_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[CONVERT-MDS] Packaging loose directory {source_dir.name} into MDS shards ({workers}"
        " workers)..."
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for split_name in loose_split_names(source_dir):
            split = scan_loose_split(source_dir, split_name)
            split_mds_dir = split.out_dir(mds_dir)
            split_mds_dir.mkdir(parents=True, exist_ok=True)
            if split.total_samples == 0:
                continue
            _write_mds_split(split, split_mds_dir, pool, opts, streaming.MDSWriter)

    _stamp_format(target_dir, "mds", opts.transcode_webp)

    if opts.purge_loose:
        purge_loose_dirs(source_dir, "Purging loose image directories...")
    return True
