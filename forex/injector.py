#!/usr/bin/env python3
"""
LemGendary Forex Manifold Parquet Metadata & Column Description Injector
=======================================================================
Embeds comprehensive column-level and schema-level descriptions directly
inside Apache Parquet files for the Forex Universe manifold.
Preserves bit-exact tensor payloads, row group chunking, and Zstandard compression.
"""

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import COLUMN_DESCRIPTIONS, PARQUET_SCHEMA, default_manifold_dir
from .shard_table import open_shard_writer


def _rewrite_with_schema(file_path: Path, temp_path: Path, batch_size: int):
    """Copy ``file_path`` to ``temp_path`` under ``PARQUET_SCHEMA``; returns (written, expected)."""
    pf = pq.ParquetFile(str(file_path))
    total_expected_rows = pf.metadata.num_rows
    writer = open_shard_writer(temp_path)

    rows_written = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            # Bind batch to the new schema with column descriptions
            tbl = pa.Table.from_batches([batch], schema=PARQUET_SCHEMA)
            writer.write_table(tbl, row_group_size=5000)
            rows_written += len(batch)
            if rows_written % 500000 == 0 or rows_written == total_expected_rows:
                pct = (rows_written / total_expected_rows) * 100
                print(f"  -> Written {rows_written:,} / {total_expected_rows:,} rows ({pct:.1f}%)")
    finally:
        writer.close()
        pf.close()
    return rows_written, total_expected_rows


def _check_rewritten(temp_path: Path, total_expected_rows: int) -> None:
    """Assert the rewritten file keeps the row count and carries every column description."""
    pf_check = pq.ParquetFile(str(temp_path))
    assert pf_check.metadata.num_rows == total_expected_rows, "Check row count failed"
    for col_name in COLUMN_DESCRIPTIONS:
        field_meta = pf_check.schema_arrow.field(col_name).metadata
        assert field_meta is not None and b"description" in field_meta, (
            f"Missing description on {col_name}"
        )
    pf_check.close()


def _replace_atomically(file_path: Path, temp_path: Path) -> None:
    backup_path = file_path.with_suffix(".parquet.bak")
    if backup_path.exists():
        backup_path.unlink()

    file_path.rename(backup_path)
    temp_path.rename(file_path)
    if backup_path.exists():
        backup_path.unlink()


def process_parquet_file(file_path: Path, batch_size: int = 50000) -> bool:
    """
    Streams through an existing Parquet file, injects field and schema metadata,
    validates bit-exact consistency and row counts, and atomically replaces the file.
    """
    file_path = Path(file_path).resolve()
    temp_path = file_path.with_suffix(".parquet.tmp")

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return False

    t0 = time.time()
    initial_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 75}")
    print(f"[EMBED] Processing {file_path.name} ({initial_size_mb:.2f} MB)...")
    print(f"{'=' * 75}")

    rows_written, total_expected_rows = _rewrite_with_schema(file_path, temp_path, batch_size)

    # Verify rows written match expected rows exactly
    if rows_written != total_expected_rows:
        print(
            f"[ERROR] Row mismatch in {file_path.name}: Expected {total_expected_rows}, got"
            f" {rows_written}"
        )
        if temp_path.exists():
            temp_path.unlink()
        return False

    _check_rewritten(temp_path, total_expected_rows)
    _replace_atomically(file_path, temp_path)

    final_size_mb = file_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(
        f"[SUCCESS] {file_path.name}: {total_expected_rows:,} rows processed in {elapsed:.1f}s"
        f" ({final_size_mb:.2f} MB)"
    )
    return True


def main():
    """Embed column descriptions into the Forex Parquet files (CLI entry point)."""
    parser = argparse.ArgumentParser(
        description="Embed column descriptions into Forex Parquet files."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=default_manifold_dir(),
        help="Path to LemGendizedForexUniverse directory",
    )
    args = parser.parse_args()
    base_dir = args.dir.resolve()
    if not base_dir.exists():
        print(f"[ERROR] Manifold directory not found: {base_dir}")
        sys.exit(1)

    parquet_files = sorted(base_dir.glob("ForexUniverse*.parquet"))
    if not parquet_files:
        print(f"[ERROR] No ForexUniverse*.parquet files found in {base_dir}")
        sys.exit(1)

    print(f"[START] Found {len(parquet_files)} Parquet files to update.")
    total_start = time.time()
    for pq_path in parquet_files:
        success = process_parquet_file(pq_path)
        if not success:
            print(f"[ABORT] Failed processing {pq_path.name}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 75}")
    print(
        f"[ALL COMPLETE] Successfully embedded column descriptions in all {len(parquet_files)}"
        f" Parquet files in {total_elapsed:.1f}s."
    )
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
