"""Pack a loose images/targets/masks/labels directory into WebDataset tar or MDS shards.

Every sample is assembled from all of its loose files (image, ``.target``
image, ``.mask`` image, ``.txt``/``.json`` label) and written as one
contiguous member group; the label is packed as ``<key>.json`` only (a text
label through ``label_json``, the manifold's label strategy). Corrupt images
are rejected rather than packed, the written shards' labels are verified
against the strategy's schema, and the loose folders are only purged after
:mod:`tools.modernize.loose_verify` has found every loose file inside the
shards on disk.
"""

from __future__ import annotations

import tarfile
from collections.abc import Iterable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from core.lazy_imports import load_streaming
from tools.modernize.common import (
    COMPANION_SUBDIRS,
    LOOSE_SUBDIRS,
    LooseSplit,
    loose_split_names,
    purge_loose_dirs,
    scan_loose_split,
)
from tools.modernize.loose_verify import missing_loose_members
from tools.repair.sample_policy_fix import enforce_sample_policy
from utils.container_stream import (
    IMAGE_WEBP_QUALITY,
    SHARD_SAMPLES,
    TARGET_WEBP_QUALITY,
    open_loose_mds_writer,
    prepare_mds_split_dir,
    prepare_shards_dir,
    resolve_worker_count,
    update_dataset_info_format,
)
from utils.label_audit import verify_shard_labels
from utils.reject_tally import RejectTally
from utils.sample_group import (
    RawMember,
    RawSample,
    SampleOutcome,
    encode_sample_or_reject,
    to_mds_record,
    write_sample,
)
from utils.shard_offsets import write_shard_offsets

# Missing loose files listed before a blocked purge gives up printing.
_MISSING_PREVIEW = 10


@dataclass(frozen=True)
class LooseConvertOptions:
    """Options shared by the loose-directory converters."""

    shard_size_samples: int = SHARD_SAMPLES
    mds_shard_size_bytes: int = 512 * 1024 * 1024
    transcode_webp: bool = True
    max_workers: int | None = None
    purge_loose: bool = False
    shard_ext: str = "data"
    label_json: str | None = None


def _resolve_options(
    options: LooseConvertOptions | None, overrides: dict[str, Any]
) -> LooseConvertOptions:
    return replace(options or LooseConvertOptions(), **overrides)


def _stamp_format(target_dir: Path, fmt: str, transcode_webp: bool) -> None:
    """Record the container format in dataset_info.yaml and report it."""
    info_path = target_dir / "dataset_info.yaml"
    if update_dataset_info_format(info_path, fmt, transcode_webp):
        print(f"[METADATA] Updated {info_path.name} format to {fmt}.")


def _read_member(path: Path | None) -> RawMember | None:
    return None if path is None else RawMember(path.read_bytes(), path.suffix.lower())


def raw_loose_sample(split: LooseSplit, primary: Path, label_json: str | None = None) -> RawSample:
    """Read every loose file of the sample started by ``primary`` into a RawSample.

    ``label_json`` names the JSON sidecar strategy for a text label
    (:mod:`utils.label_json`).
    """
    key = primary.stem
    return RawSample(
        key=key,
        primary=RawMember(primary.read_bytes(), primary.suffix.lower()),
        target=None if split.target_only else _read_member(split.targets.get(key)),
        mask=_read_member(split.masks.get(key)),
        label=_read_member(split.labels.get(key)),
        primary_quality=TARGET_WEBP_QUALITY if split.target_only else IMAGE_WEBP_QUALITY,
        label_json=label_json,
    )


def _encode_all(
    split: LooseSplit, files: list[Path], pool: Executor, opts: LooseConvertOptions
) -> Iterable[SampleOutcome]:
    def _encode(path: Path) -> SampleOutcome:
        raw = raw_loose_sample(split, path, opts.label_json)
        return encode_sample_or_reject(raw, opts.transcode_webp)

    return pool.map(_encode, files)


def _write_tar_shard(
    shard_path: Path, outcomes: Iterable[SampleOutcome], rejects: RejectTally
) -> int:
    """Write each encoded sample contiguously into one tar shard; return files consumed."""
    consumed = 0
    with tarfile.open(shard_path, "w") as tf:
        for outcome in outcomes:
            consumed += 1
            if outcome.sample is None:
                rejects.record(outcome.key, outcome.reason)
                continue
            write_sample(tf, outcome.sample)
    return consumed


def _write_tar_split(
    split: LooseSplit, out_dir: Path, pool: Executor, opts: LooseConvertOptions,
    rejects: RejectTally,
) -> None:
    """Write one split as WebDataset tar shards, one member group per sample."""
    primary = split.primary_files()
    size = opts.shard_size_samples
    total_shards = (split.total_samples + size - 1) // size
    print(
        f"[CONVERT] Split '{split.name}': {split.total_samples} samples -> {total_shards} shards."
    )
    desc = f"{'TARGET ' if split.target_only else ''}SHARDS [{split.name}]"
    ext = opts.shard_ext.lstrip(".")
    with tqdm(total=split.total_samples, desc=desc, unit="img", dynamic_ncols=True) as pbar:
        for s_idx in range(total_shards):
            chunk = primary[s_idx * size:(s_idx + 1) * size]
            outcomes = _encode_all(split, chunk, pool, opts)
            pbar.update(_write_tar_shard(out_dir / f"shard-{s_idx:05d}.{ext}", outcomes, rejects))


def _purge_after_verification(
    source_dir: Path, verified: list[tuple[LooseSplit, Path]], opts: LooseConvertOptions,
    rejects: RejectTally,
) -> None:
    """Purge the loose folders only when every loose file is inside the shards."""
    missing: list[Path] = []
    for split, shard_dir in verified:
        missing.extend(missing_loose_members(split, shard_dir, opts.transcode_webp, rejects.keys))
    if missing:
        print(
            f"[ERROR] Purge blocked: {len(missing)} loose files are not inside the written"
            " shards (orphan masks/labels/targets or failed writes). Loose folders kept."
        )
        for path in missing[:_MISSING_PREVIEW]:
            print(f"  missing: {path.relative_to(source_dir)}")
        return
    purge_loose_dirs(source_dir, "Purging loose directories (all members verified in shards)...")


def convert_loose_directory_to_webdataset(
    source_dir: Path,
    target_dir: Path,
    options: LooseConvertOptions | None = None,
    **overrides: Any,
) -> bool:
    """Convert a directory containing loose images/targets/masks/labels into WebDataset shards.

    ``options`` (or keyword ``overrides`` of its fields: shard_size_samples,
    transcode_webp, max_workers, purge_loose, shard_ext) configure the run.
    """
    opts = _resolve_options(options, overrides)
    shards_dir = prepare_shards_dir(target_dir)
    workers = resolve_worker_count(opts.max_workers)
    rejects = RejectTally(source_dir.name)

    print(
        f"[CONVERT] Packaging loose directory {source_dir.name} into WebDataset shards ({workers}"
        " workers)..."
    )
    written: list[tuple[LooseSplit, Path]] = []
    for split_name in loose_split_names(source_dir):
        split = scan_loose_split(source_dir, split_name)
        out_dir = split.out_dir(shards_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written.append((split, out_dir))
        if split.total_samples == 0:
            continue
        with ThreadPoolExecutor(max_workers=workers) as pool:
            _write_tar_split(split, out_dir, pool, opts, rejects)

    _stamp_format(target_dir, "webdataset", opts.transcode_webp)
    rejects.report()
    write_shard_offsets(shards_dir)
    if not verify_shard_labels(shards_dir, "CONVERT", opts.label_json):
        print("[ERROR] Label verification failed: loose folders kept.")
        return False

    if opts.purge_loose:
        _purge_after_verification(source_dir, written, opts, rejects)
    return enforce_sample_policy(target_dir)


@dataclass(frozen=True)
class _MdsRun:
    """Shared state of one loose-to-MDS conversion."""

    pool: Executor
    opts: LooseConvertOptions
    writer_cls: Any
    rejects: RejectTally


def _write_mds_split(split: LooseSplit, split_mds_dir: Path, run: _MdsRun) -> None:
    """Write one split of a loose directory as zstd-compressed MDS shards."""
    pool, opts, rejects = run.pool, run.opts, run.rejects
    print(f"[CONVERT-MDS] Split '{split.name}': {split.total_samples} samples -> MDS.")
    writer = open_loose_mds_writer(
        run.writer_cls, prepare_mds_split_dir(split_mds_dir), opts.mds_shard_size_bytes
    )
    with tqdm(
        total=split.total_samples, desc=f"MDS [{split.name}]", unit="img", dynamic_ncols=True
    ) as pbar:
        for outcome in _encode_all(split, split.primary_files(), pool, opts):
            if outcome.sample is None:
                rejects.record(outcome.key, outcome.reason)
            else:
                writer.write(to_mds_record(split.name, outcome.sample))
            pbar.update(1)
    writer.finish()


def _has_companion_files(source_dir: Path) -> bool:
    return any(
        p.is_file() for sub in COMPANION_SUBDIRS for p in (source_dir / sub).rglob("*")
    )


def _purge_mds_sources(source_dir: Path) -> None:
    """Purge after an MDS run; masks/labels are kept because MDS is not re-read here."""
    if _has_companion_files(source_dir):
        print(
            "[WARN] masks/ or labels/ present: the purge barrier only verifies tar shards,"
            " so only images/ and targets/ are purged after an MDS conversion."
        )
        subdirs = tuple(sub for sub in LOOSE_SUBDIRS if sub not in COMPANION_SUBDIRS)
        purge_loose_dirs(source_dir, "Purging loose image directories...", subdirs)
        return
    purge_loose_dirs(source_dir, "Purging loose image directories...")


def convert_loose_directory_to_mds(
    source_dir: Path,
    target_dir: Path,
    options: LooseConvertOptions | None = None,
    **overrides: Any,
) -> bool:
    """Convert a loose images/targets/masks/labels directory into MDS shards.

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
    rejects = RejectTally(source_dir.name)
    print(
        f"[CONVERT-MDS] Packaging loose directory {source_dir.name} into MDS shards ({workers}"
        " workers)..."
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        run = _MdsRun(pool, opts, streaming.MDSWriter, rejects)
        for split_name in loose_split_names(source_dir):
            split = scan_loose_split(source_dir, split_name)
            split_mds_dir = split.out_dir(mds_dir)
            split_mds_dir.mkdir(parents=True, exist_ok=True)
            if split.total_samples == 0:
                continue
            _write_mds_split(split, split_mds_dir, run)

    _stamp_format(target_dir, "mds", opts.transcode_webp)
    rejects.report()

    if opts.purge_loose:
        _purge_mds_sources(source_dir)
    return True
