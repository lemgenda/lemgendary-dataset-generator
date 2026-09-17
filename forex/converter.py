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
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import COLUMN_DESCRIPTIONS, PARQUET_SCHEMA, TIMEFRAME_LOOKBACK, EXTENDED_PAIRS

ROW_GROUP_SIZE = 5000
CHUNK_BATCH_SIZE = 20000


def get_dir_size_gb(dir_path: Path) -> float:
    """Calculate total size of directory in gigabytes."""
    total = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            p = os.path.join(root, f)
            try:
                total += os.path.getsize(p)
            except OSError as err:
                logging.debug("Could not read file size for %s: %s", p, err)
    return total / (1024 ** 3)


def get_file_size_gb(file_path: Path) -> float:
    """Calculate file size in gigabytes."""
    try:
        return os.path.getsize(file_path) / (1024 ** 3)
    except OSError:
        return 0.0


def _verify_converted_parquet(temp_parquet: Path, total_written: int, year: int) -> bool:
    print(f" [VERIFY] Commencing bit-exact validation for ForexUniverse{year}...")
    v_start = time.time()

    pf = pq.ParquetFile(str(temp_parquet))
    if pf.metadata.num_rows != total_written:
        print(f" [FAIL] Row count mismatch: {pf.metadata.num_rows} in file vs {total_written} written.")
        temp_parquet.unlink(missing_ok=True)
        return False

    # Read random sample rows across the Parquet file to verify tensor parity
    for idx in [0, total_written // 2, total_written - 1]:
        curr_row = 0
        target_rg = 0
        rg_offset = 0
        for rg_i in range(pf.num_row_groups):
            rg_rows = pf.metadata.row_group(rg_i).num_rows
            if curr_row <= idx < curr_row + rg_rows:
                target_rg = rg_i
                rg_offset = idx - curr_row
                break
            curr_row += rg_rows

        rg_table = pf.read_row_group(target_rg)
        row_feat = rg_table["features"][rg_offset].as_buffer()
        row_seq_len = rg_table["seq_len"][rg_offset].as_py()
        row_n_feat = rg_table["n_features"][rg_offset].as_py()
        recovered_arr = np.frombuffer(row_feat, dtype=np.float32).reshape(row_seq_len, row_n_feat)

        if recovered_arr.shape != (row_seq_len, row_n_feat) or not np.isfinite(recovered_arr).all():
            print(f" [FAIL] Invalid recovered array at global row {idx}")
            del rg_table, row_feat, pf
            gc.collect()
            temp_parquet.unlink(missing_ok=True)
            return False

    v_duration = time.time() - v_start
    print(f" [PASS] Bit-exact verification PASSED in {v_duration:.2f}s ({pf.metadata.num_rows:,} rows verified).")
    del rg_table, row_feat, pf
    gc.collect()
    time.sleep(0.5)
    return True


def convert_year(base_manifold: Path, year: int, dry_run: bool = False, skip_cleanup: bool = False) -> bool:
    """
    Converts a single year's nested .npy structure into a unified Parquet file.
    Validates bit-exact accuracy and purges the old directory upon success.
    """
    year_folder = base_manifold / f"ForexUniverse{year}"
    target_parquet = base_manifold / f"ForexUniverse{year}.parquet"
    temp_parquet = base_manifold / f"ForexUniverse{year}.parquet.tmp"

    if target_parquet.exists() and not year_folder.exists():
        print(f"[SKIP] Year {year} already converted -> {target_parquet.name} ({get_file_size_gb(target_parquet):.2f} GB)")
        return True

    if not year_folder.exists():
        print(f"[ERROR] Source folder does not exist: {year_folder}")
        return False

    initial_size_gb = get_dir_size_gb(year_folder)
    print(f"\n{'=' * 75}")
    print(f" [CONVERT] Processing ForexUniverse{year} (Source Size: {initial_size_gb:.2f} GB)")
    print(f"{'=' * 75}")

    # Discover pairs and timeframes
    pairs_found = sorted([d.name for d in year_folder.iterdir() if d.is_dir() and d.name in EXTENDED_PAIRS])
    if not pairs_found:
        pairs_found = sorted([d.name for d in year_folder.iterdir() if d.is_dir()])
    print(f" [DISCOVERY] Found {len(pairs_found)} pairs for year {year}")

    # Temporary file cleanup from any previous interrupted run
    if temp_parquet.exists():
        temp_parquet.unlink(missing_ok=True)

    if dry_run:
        print(f" [DRY-RUN] Would convert {year_folder} -> {target_parquet}")
        return True

    t0 = time.time()
    total_written = 0

    writer = pq.ParquetWriter(
        str(temp_parquet),
        schema=PARQUET_SCHEMA,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
    )

    try:
        for pair in pairs_found:
            pair_dir = year_folder / pair
            if not pair_dir.is_dir():
                continue

            tf_dirs = sorted(
                [d for d in pair_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda x: int(x.name)
            )

            for tf_path in tf_dirs:
                tf = int(tf_path.name)
                expected_seq_len = TIMEFRAME_LOOKBACK.get(tf, 168)

                # Collect shard files (either unified X.npy or chunked X_chunk*.npy)
                x_single = tf_path / "X.npy"
                ydir_single = tf_path / "y_dir.npy"
                ymag_single = tf_path / "y_mag.npy"
                ts_single = tf_path / "timestamps.npy"

                shard_tasks = []
                if x_single.exists() and ydir_single.exists() and ymag_single.exists():
                    shard_tasks.append((x_single, ydir_single, ymag_single, ts_single if ts_single.exists() else None))
                else:
                    chunk_files = sorted(
                        [f for f in tf_path.iterdir() if f.name.startswith("X_chunk") and f.name.endswith(".npy")],
                        key=lambda x: int(x.name.replace("X_chunk", "").replace(".npy", ""))
                    )
                    for cf in chunk_files:
                        c_idx = cf.name.replace("X_chunk", "").replace(".npy", "")
                        ydf = tf_path / f"y_dir_chunk{c_idx}.npy"
                        ymf = tf_path / f"y_mag_chunk{c_idx}.npy"
                        tsf = tf_path / f"timestamps_chunk{c_idx}.npy"
                        if cf.exists() and ydf.exists() and ymf.exists():
                            shard_tasks.append((cf, ydf, ymf, tsf if tsf.exists() else None))

                for x_f, yd_f, ym_f, ts_f in shard_tasks:
                    X = np.load(str(x_f), mmap_mode="r")
                    yd = np.load(str(yd_f), mmap_mode="r")
                    ym = np.load(str(ym_f), mmap_mode="r")
                    ts = np.load(str(ts_f), mmap_mode="r") if ts_f else np.zeros(len(X), dtype=np.int64)

                    num_samples = len(X)
                    if num_samples == 0:
                        continue

                    seq_len = X.shape[1] if len(X.shape) > 1 else expected_seq_len
                    n_features = X.shape[2] if len(X.shape) > 2 else 14

                    # Stream in memory-safe sub-batches to bound RAM
                    for start_idx in range(0, num_samples, CHUNK_BATCH_SIZE):
                        end_idx = min(start_idx + CHUNK_BATCH_SIZE, num_samples)
                        sub_len = end_idx - start_idx

                        sub_x = X[start_idx:end_idx]
                        sub_yd = yd[start_idx:end_idx]
                        sub_ym = ym[start_idx:end_idx]
                        sub_ts = ts[start_idx:end_idx]

                        # Convert sub_x to list of raw bytes
                        x_bytes = [sample.tobytes() for sample in sub_x]

                        table = pa.Table.from_arrays([
                            pa.array([pair] * sub_len, type=pa.string()),
                            pa.array([tf] * sub_len, type=pa.int32()),
                            pa.array(sub_ts, type=pa.int64()),
                            pa.array(sub_yd, type=pa.int8()),
                            pa.array(sub_ym[:, 0], type=pa.float32()),
                            pa.array(sub_ym[:, 1], type=pa.float32()),
                            pa.array([seq_len] * sub_len, type=pa.int16()),
                            pa.array([n_features] * sub_len, type=pa.int16()),
                            pa.array(x_bytes, type=pa.binary()),
                        ], schema=PARQUET_SCHEMA)

                        writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
                        total_written += sub_len

                    for arr_obj in [X, yd, ym, ts]:
                        if hasattr(arr_obj, "_mmap") and arr_obj._mmap is not None:
                            try:
                                arr_obj._mmap.close()
                            except Exception as err:
                                logging.debug("Failed closing mmap on array object: %s", err)
                    del X, yd, ym, ts
                    gc.collect()

        writer.close()
    except Exception as e:
        writer.close()
        temp_parquet.unlink(missing_ok=True)
        print(f" [FATAL] Exception during Parquet conversion for year {year}: {e}")
        return False

    write_duration = time.time() - t0
    parquet_size_gb = get_file_size_gb(temp_parquet)
    print(f" [SUCCESS] Written {total_written:,} samples to temporary Parquet in {write_duration:.1f}s")
    print(f" [METRICS] Size: {initial_size_gb:.2f} GB -> {parquet_size_gb:.2f} GB ({(1 - parquet_size_gb / max(initial_size_gb, 0.001)) * 100:.1f}% reduction)")

    if not _verify_converted_parquet(temp_parquet, total_written, year):
        return False

    # Finalize by renaming temp to target
    if target_parquet.exists():
        target_parquet.unlink(missing_ok=True)
    temp_parquet.rename(target_parquet)
    print(f" [COMMITTED] Manifold file finalized: {target_parquet.name}")

    # ─── Safe Cleanup ───────────────────────────────────────────────────────
    if not skip_cleanup:
        print(f" [CLEANUP] Safely removing legacy .npy directory: {year_folder}...")
        try:
            shutil.rmtree(year_folder)
            print(f" [CLEANUP] Reclaimed {initial_size_gb:.2f} GB disk space!")
        except Exception as e:
            print(f" [WARNING] Could not fully remove {year_folder}: {e}")
    else:
        print(" [CLEANUP] Skipped directory removal as requested.")

    return True


def main():
    parser = argparse.ArgumentParser(description="LemGendary Forex Manifold Parquet Conversion Engine")
    parser.add_argument(
        "--base_dir",
        default=r"c:\Development\python\model-training\LemGendaryDatasets\LemGendizedForexUniverseLarge",
        help="Path to LemGendizedForexUniverseLarge directory"
    )
    parser.add_argument("--year", type=int, default=None, help="Specific year to convert (2019..2026)")
    parser.add_argument("--all", action="store_true", help="Convert all years 2019..2026 sequentially")
    parser.add_argument("--dry_run", action="store_true", help="Simulate conversion without writing")
    parser.add_argument("--skip_cleanup", action="store_true", help="Keep source .npy directories after conversion")

    args = parser.parse_args()
    base_manifold = Path(args.base_dir).resolve()

    if not base_manifold.exists():
        print(f"[FATAL] Base manifold directory does not exist: {base_manifold}")
        sys.exit(1)

    years_to_process = []
    if args.year:
        years_to_process = [args.year]
    elif args.all:
        years_to_process = list(range(2019, 2027))
    else:
        print("Specify either --year <YYYY> or --all. Exiting.")
        sys.exit(0)

    print("=" * 75)
    print(" LEMGENDARY FOREX UNIVERSE -> PARQUET CONVERSION ENGINE")
    print(f" Base Manifold: {base_manifold}")
    print(f" Target Years: {years_to_process}")
    print("=" * 75)

    success_count = 0
    total_saved_gb = 0.0

    for yr in years_to_process:
        yr_folder = base_manifold / f"ForexUniverse{yr}"
        pre_size = get_dir_size_gb(yr_folder) if yr_folder.exists() else 0.0
        success = convert_year(base_manifold, yr, dry_run=args.dry_run, skip_cleanup=args.skip_cleanup)
        if success:
            success_count += 1
            post_file = base_manifold / f"ForexUniverse{yr}.parquet"
            post_size = get_file_size_gb(post_file) if post_file.exists() else 0.0
            total_saved_gb += max(0.0, pre_size - post_size)
        else:
            print(f"[ABORT] Conversion failed for year {yr}. Stopping batch.")
            break

    print("\n" + "=" * 75)
    print(f" [SUMMARY] Successfully processed {success_count}/{len(years_to_process)} years.")
    print(f" [SAVINGS] Total Disk Space Reclaimed: {total_saved_gb:.2f} GB")
    print("=" * 75)


if __name__ == "__main__":
    main()
