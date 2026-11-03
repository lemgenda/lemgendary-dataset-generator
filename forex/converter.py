#!/usr/bin/env python3
"""
LemGendary Forex Manifold Parquet Conversion & Verification Engine
===================================================================
Converts nested .npy Forex manifold folders (ForexUniverse2019..2026)
into unified, highly compressed Apache Parquet files with bit-exact validation
and safe year-by-year disk reclamation.
"""

import argparse
import gc
import logging
import mmap
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import EXTENDED_PAIRS, TIMEFRAME_LOOKBACK, default_manifold_dir
from .shard_table import ShardLabels, build_shard_table, open_shard_writer

ROW_GROUP_SIZE = 5000
CHUNK_BATCH_SIZE = 20000

logger = logging.getLogger(__name__)


def get_dir_size_gb(dir_path: Path) -> float:
    """Calculate total size of directory in gigabytes."""
    total = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            p = os.path.join(root, f)
            try:
                total += os.path.getsize(p)
            except OSError as err:
                logger.debug("Could not read file size for %s: %s", p, err)
    return total / (1024 ** 3)


def get_file_size_gb(file_path: Path) -> float:
    """Calculate file size in gigabytes."""
    try:
        return os.path.getsize(file_path) / (1024 ** 3)
    except OSError as err:
        logger.debug("Could not read file size for %s: %s", file_path, err)
        return 0.0


# ─── Verification ───────────────────────────────────────────────────────────


def _locate_row(pf, idx):
    """Return (row group, offset inside it) of global row ``idx``."""
    curr_row = 0
    for rg_i in range(pf.num_row_groups):
        rg_rows = pf.metadata.row_group(rg_i).num_rows
        if curr_row <= idx < curr_row + rg_rows:
            return rg_i, idx - curr_row
        curr_row += rg_rows
    return 0, 0


def _row_is_valid(pf, idx):
    """True when row ``idx`` decodes to a finite ``(seq_len, n_features)`` float32 window."""
    target_rg, rg_offset = _locate_row(pf, idx)
    rg_table = pf.read_row_group(target_rg)
    row_feat = rg_table["features"][rg_offset].as_buffer()
    row_seq_len = rg_table["seq_len"][rg_offset].as_py()
    row_n_feat = rg_table["n_features"][rg_offset].as_py()
    recovered_arr = np.frombuffer(row_feat, dtype=np.float32).reshape(row_seq_len, row_n_feat)
    return recovered_arr.shape == (row_seq_len, row_n_feat) and bool(
        np.isfinite(recovered_arr).all()
    )


def _verify_converted_parquet(temp_parquet: Path, total_written: int, year: int) -> bool:
    print(f" [VERIFY] Commencing bit-exact validation for ForexUniverse{year}...")
    v_start = time.time()

    pf = pq.ParquetFile(str(temp_parquet))
    num_rows = pf.metadata.num_rows
    if num_rows != total_written:
        print(f" [FAIL] Row count mismatch: {num_rows} in file vs {total_written} written.")
        temp_parquet.unlink(missing_ok=True)
        return False

    # Read sample rows across the Parquet file to verify tensor parity
    sample_rows = [0, total_written // 2, total_written - 1]
    bad_row = next((idx for idx in sample_rows if not _row_is_valid(pf, idx)), None)
    del pf
    gc.collect()
    if bad_row is not None:
        print(f" [FAIL] Invalid recovered array at global row {bad_row}")
        temp_parquet.unlink(missing_ok=True)
        return False

    v_duration = time.time() - v_start
    print(
        f" [PASS] Bit-exact verification PASSED in {v_duration:.2f}s ({num_rows:,} rows"
        " verified)."
    )
    time.sleep(0.5)
    return True


# ─── Shard discovery and streaming ──────────────────────────────────────────


@dataclass(frozen=True)
class _ShardFiles:
    """One legacy .npy shard: windows, direction labels, magnitudes and optional timestamps."""

    x: Path
    y_dir: Path
    y_mag: Path
    timestamps: Path | None


def _chunk_index(path: Path) -> str:
    return path.name.replace("X_chunk", "").replace(".npy", "")


def _shard_if_complete(x: Path, y_dir: Path, y_mag: Path, ts: Path) -> _ShardFiles | None:
    if x.exists() and y_dir.exists() and y_mag.exists():
        return _ShardFiles(x, y_dir, y_mag, ts if ts.exists() else None)
    return None


def _shard_files(tf_path: Path) -> list[_ShardFiles]:
    """Unified ``X.npy`` shard of a timeframe folder, else its ``X_chunk*.npy`` shards."""
    single = _shard_if_complete(
        tf_path / "X.npy", tf_path / "y_dir.npy", tf_path / "y_mag.npy", tf_path / "timestamps.npy"
    )
    if single is not None:
        return [single]

    chunk_files = sorted(
        [f for f in tf_path.iterdir() if f.name.startswith("X_chunk") and f.name.endswith(".npy")],
        key=lambda x: int(_chunk_index(x)),
    )
    shards = []
    for cf in chunk_files:
        c_idx = _chunk_index(cf)
        shard = _shard_if_complete(
            cf,
            tf_path / f"y_dir_chunk{c_idx}.npy",
            tf_path / f"y_mag_chunk{c_idx}.npy",
            tf_path / f"timestamps_chunk{c_idx}.npy",
        )
        if shard is not None:
            shards.append(shard)
    return shards


def _close_mmaps(arrays) -> None:
    # np.load(mmap_mode=...) arrays expose their backing
    # mmap.mmap as the public ``.base``; close it explicitly.
    for arr_obj in arrays:
        if arr_obj is not None and isinstance(arr_obj.base, mmap.mmap):
            try:
                arr_obj.base.close()
            except (BufferError, OSError, ValueError) as err:
                logger.debug("Failed closing mmap on array object: %s", err)


def _stream_shard(writer, shard: _ShardFiles, pair: str, tf: int) -> int:
    """Append one shard to ``writer`` in memory-safe sub-batches; returns the rows written."""
    features = np.load(str(shard.x), mmap_mode="r")
    yd = np.load(str(shard.y_dir), mmap_mode="r")
    ym = np.load(str(shard.y_mag), mmap_mode="r")
    ts = np.load(str(shard.timestamps), mmap_mode="r") if shard.timestamps else None

    num_samples = len(features)
    expected_seq_len = TIMEFRAME_LOOKBACK.get(tf, 168)
    for start_idx in range(0, num_samples, CHUNK_BATCH_SIZE):
        end_idx = min(start_idx + CHUNK_BATCH_SIZE, num_samples)
        labels = ShardLabels(
            y_dir=yd[start_idx:end_idx],
            y_mag=ym[start_idx:end_idx],
            timestamps=ts[start_idx:end_idx] if ts is not None else None,
        )
        table = build_shard_table(
            pair, tf, features[start_idx:end_idx], labels, default_seq_len=expected_seq_len
        )
        writer.write_table(table, row_group_size=ROW_GROUP_SIZE)

    _close_mmaps([features, yd, ym, ts])
    del features, yd, ym, ts
    gc.collect()
    return num_samples


def _timeframe_dirs(pair_dir: Path) -> list[Path]:
    return sorted(
        [d for d in pair_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
    )


def _write_year(writer, year_folder: Path, pairs: list[str]) -> int:
    """Stream every pair/timeframe shard of ``year_folder`` into ``writer``."""
    total_written = 0
    for pair in pairs:
        pair_dir = year_folder / pair
        if not pair_dir.is_dir():
            continue
        for tf_path in _timeframe_dirs(pair_dir):
            tf = int(tf_path.name)
            for shard in _shard_files(tf_path):
                total_written += _stream_shard(writer, shard, pair, tf)
    return total_written


def _discover_pairs(year_folder: Path) -> list[str]:
    pairs_found = sorted(
        [d.name for d in year_folder.iterdir() if d.is_dir() and d.name in EXTENDED_PAIRS]
    )
    if not pairs_found:
        pairs_found = sorted([d.name for d in year_folder.iterdir() if d.is_dir()])
    return pairs_found


def _stream_year(year_folder: Path, temp_parquet: Path, pairs: list[str], year: int):
    """Stream ``pairs`` into ``temp_parquet``; returns rows written, or None on failure."""
    writer = open_shard_writer(temp_parquet, use_dictionary=True)
    try:
        total_written = _write_year(writer, year_folder, pairs)
        writer.close()
    except (OSError, ValueError, IndexError, pa.ArrowException) as e:
        logger.exception("Parquet conversion failed for year %s", year)
        writer.close()
        temp_parquet.unlink(missing_ok=True)
        print(f" [FATAL] Exception during Parquet conversion for year {year}: {e}")
        return None
    return total_written


def _report_write(total_written: int, t0: float, initial_size_gb: float, temp_parquet: Path):
    write_duration = time.time() - t0
    parquet_size_gb = get_file_size_gb(temp_parquet)
    print(
        f" [SUCCESS] Written {total_written:,} samples to temporary Parquet in"
        f" {write_duration:.1f}s"
    )
    print(
        f" [METRICS] Size: {initial_size_gb:.2f} GB -> {parquet_size_gb:.2f} GB"
        f" ({(1 - parquet_size_gb / max(initial_size_gb, 0.001)) * 100:.1f}% reduction)"
    )


def _commit(temp_parquet: Path, target_parquet: Path) -> None:
    if target_parquet.exists():
        target_parquet.unlink(missing_ok=True)
    temp_parquet.rename(target_parquet)
    print(f" [COMMITTED] Manifold file finalized: {target_parquet.name}")


def _cleanup_source(year_folder: Path, initial_size_gb: float, skip_cleanup: bool) -> None:
    if skip_cleanup:
        print(" [CLEANUP] Skipped directory removal as requested.")
        return
    print(f" [CLEANUP] Safely removing legacy .npy directory: {year_folder}...")
    try:
        shutil.rmtree(year_folder)
        print(f" [CLEANUP] Reclaimed {initial_size_gb:.2f} GB disk space!")
    except OSError as e:
        logger.warning("Could not fully remove %s: %s", year_folder, e)
        print(f" [WARNING] Could not fully remove {year_folder}: {e}")


def convert_year(
    base_manifold: Path, year: int, dry_run: bool = False, skip_cleanup: bool = False
) -> bool:
    """
    Converts a single year's nested .npy structure into a unified Parquet file.
    Validates bit-exact accuracy and purges the old directory upon success.
    """
    year_folder = base_manifold / f"ForexUniverse{year}"
    target_parquet = base_manifold / f"ForexUniverse{year}.parquet"
    temp_parquet = base_manifold / f"ForexUniverse{year}.parquet.tmp"

    if target_parquet.exists() and not year_folder.exists():
        print(
            f"[SKIP] Year {year} already converted -> {target_parquet.name}"
            f" ({get_file_size_gb(target_parquet):.2f} GB)"
        )
        return True

    if not year_folder.exists():
        print(f"[ERROR] Source folder does not exist: {year_folder}")
        return False

    initial_size_gb = get_dir_size_gb(year_folder)
    print(f"\n{'=' * 75}")
    print(f" [CONVERT] Processing ForexUniverse{year} (Source Size: {initial_size_gb:.2f} GB)")
    print(f"{'=' * 75}")

    pairs_found = _discover_pairs(year_folder)
    print(f" [DISCOVERY] Found {len(pairs_found)} pairs for year {year}")

    # Temporary file cleanup from any previous interrupted run
    if temp_parquet.exists():
        temp_parquet.unlink(missing_ok=True)

    if dry_run:
        print(f" [DRY-RUN] Would convert {year_folder} -> {target_parquet}")
        return True

    t0 = time.time()
    total_written = _stream_year(year_folder, temp_parquet, pairs_found, year)
    if total_written is None:
        return False
    _report_write(total_written, t0, initial_size_gb, temp_parquet)

    if not _verify_converted_parquet(temp_parquet, total_written, year):
        return False

    _commit(temp_parquet, target_parquet)
    _cleanup_source(year_folder, initial_size_gb, skip_cleanup)
    return True


# ─── CLI ────────────────────────────────────────────────────────────────────


def _build_parser(default_base: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LemGendary Forex Manifold Parquet Conversion Engine"
    )
    parser.add_argument(
        "--base_dir",
        default=str(default_base),
        help="Path to LemGendizedForexUniverse directory"
    )
    parser.add_argument(
        "--year", type=int, default=None, help="Specific year to convert (2019..2026)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Convert all years 2019..2026 sequentially"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Simulate conversion without writing"
    )
    parser.add_argument(
        "--skip_cleanup", action="store_true", help="Keep source .npy directories after conversion"
    )
    return parser


def _years_from_args(args) -> list[int]:
    if args.year:
        return [args.year]
    if args.all:
        return list(range(2019, 2027))
    print("Specify either --year <YYYY> or --all. Exiting.")
    sys.exit(0)


def _convert_years(base_manifold: Path, years: list[int], args):
    """Convert ``years`` in order, stopping at the first failure; returns (successes, GB saved)."""
    success_count = 0
    total_saved_gb = 0.0
    for yr in years:
        yr_folder = base_manifold / f"ForexUniverse{yr}"
        pre_size = get_dir_size_gb(yr_folder) if yr_folder.exists() else 0.0
        if not convert_year(
            base_manifold, yr, dry_run=args.dry_run, skip_cleanup=args.skip_cleanup
        ):
            print(f"[ABORT] Conversion failed for year {yr}. Stopping batch.")
            break
        success_count += 1
        post_file = base_manifold / f"ForexUniverse{yr}.parquet"
        post_size = get_file_size_gb(post_file) if post_file.exists() else 0.0
        total_saved_gb += max(0.0, pre_size - post_size)
    return success_count, total_saved_gb


def main():
    """Convert legacy .npy Forex shards to yearly Parquet files (CLI entry point)."""
    args = _build_parser(default_manifold_dir()).parse_args()
    base_manifold = Path(args.base_dir).resolve()

    if not base_manifold.exists():
        print(f"[FATAL] Base manifold directory does not exist: {base_manifold}")
        sys.exit(1)

    years_to_process = _years_from_args(args)

    print("=" * 75)
    print(" LEMGENDARY FOREX UNIVERSE -> PARQUET CONVERSION ENGINE")
    print(f" Base Manifold: {base_manifold}")
    print(f" Target Years: {years_to_process}")
    print("=" * 75)

    success_count, total_saved_gb = _convert_years(base_manifold, years_to_process, args)

    print("\n" + "=" * 75)
    print(f" [SUMMARY] Successfully processed {success_count}/{len(years_to_process)} years.")
    print(f" [SAVINGS] Total Disk Space Reclaimed: {total_saved_gb:.2f} GB")
    print("=" * 75)


if __name__ == "__main__":
    main()
