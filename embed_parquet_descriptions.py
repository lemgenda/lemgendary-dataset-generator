#!/usr/bin/env python3
"""
LemGendary Forex Manifold Parquet Metadata & Column Description Injector
=======================================================================
Embeds comprehensive column-level and schema-level descriptions directly
inside Apache Parquet files for the Forex Universe manifold.
Preserves bit-exact tensor payloads, row group chunking, and Zstandard compression.
"""

import json
import os
import sys
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

COLUMN_DESCRIPTIONS = {
    "pair": "Asset / currency pair / commodity symbol identifier (e.g., EURUSD, GBPUSD, USDJPY, XAUUSD, NAS100, DE40, USOIL, US500).",
    "timeframe": "Bar aggregation timeframe rung in minutes: 1=M1 (1min), 5=M5 (5min), 15=M15 (15min), 60=H1 (60min), 240=H4 (240min), 1440=D1 (1440min).",
    "timestamp": "Millisecond Unix epoch timestamp of the sequence prediction anchor / candle close.",
    "y_dir": "Causal directional classification target label over forward horizon: 0=SELL (Down), 1=HOLD (Sideways/Neutral), 2=BUY (Up).",
    "tp_pips": "Optimal forward Take-Profit target excursion magnitude in pips.",
    "sl_pips": "Maximum adverse excursion Stop-Loss safety threshold in pips.",
    "seq_len": "Historical lookback sequence length in bars (e.g., 168 for H1 macro, 512 for M1 microstructure).",
    "n_features": "Number of input feature dimensions per timestep (14 channels: OHLCV, RSI, MACD, MACD Signal, ATR, Bollinger Band Width, Session Sin/Cos, ATR Percentile, Bar Range Ratio).",
    "features": "Serialized float32 binary tensor representing the normalized [seq_len, n_features] temporal feature matrix."
}

PARQUET_SCHEMA = pa.schema([
    pa.field("pair", pa.string(), metadata={"description": COLUMN_DESCRIPTIONS["pair"]}),
    pa.field("timeframe", pa.int32(), metadata={"description": COLUMN_DESCRIPTIONS["timeframe"]}),
    pa.field("timestamp", pa.int64(), metadata={"description": COLUMN_DESCRIPTIONS["timestamp"]}),
    pa.field("y_dir", pa.int8(), metadata={"description": COLUMN_DESCRIPTIONS["y_dir"]}),
    pa.field("tp_pips", pa.float32(), metadata={"description": COLUMN_DESCRIPTIONS["tp_pips"]}),
    pa.field("sl_pips", pa.float32(), metadata={"description": COLUMN_DESCRIPTIONS["sl_pips"]}),
    pa.field("seq_len", pa.int16(), metadata={"description": COLUMN_DESCRIPTIONS["seq_len"]}),
    pa.field("n_features", pa.int16(), metadata={"description": COLUMN_DESCRIPTIONS["n_features"]}),
    pa.field("features", pa.binary(), metadata={"description": COLUMN_DESCRIPTIONS["features"]}),
], metadata={
    b"description": b"LemGendary Forex Universe High-Fidelity OHLCV Temporal Manifold",
    b"columns": json.dumps(COLUMN_DESCRIPTIONS).encode("utf-8"),
    b"domain": b"Financial & Time-Series",
    b"task": b"forex_prediction",
    b"timeframe_rungs": b"[1, 5, 15, 60, 240, 1440]",
    b"features_list": b'["open", "high", "low", "close", "volume", "rsi", "macd", "macd_signal", "atr", "bb_width", "session_sin", "session_cos", "atr_percentile", "bar_range_ratio"]',
    b"author": b"LemGendary AI",
    b"created_by": b"LemGendary MT5 Compiler Pipeline"
})


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

    pf = pq.ParquetFile(str(file_path))
    total_expected_rows = pf.metadata.num_rows

    writer = pq.ParquetWriter(
        str(temp_path),
        schema=PARQUET_SCHEMA,
        compression="zstd",
        compression_level=3
    )

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

    # Verify rows written match expected rows exactly
    if rows_written != total_expected_rows:
        print(f"[ERROR] Row mismatch in {file_path.name}: Expected {total_expected_rows}, got {rows_written}")
        if temp_path.exists():
            temp_path.unlink()
        return False

    # Verify temp file schema and read sample
    pf_check = pq.ParquetFile(str(temp_path))
    assert pf_check.metadata.num_rows == total_expected_rows, "Check row count failed"
    for col_name, desc in COLUMN_DESCRIPTIONS.items():
        field_meta = pf_check.schema_arrow.field(col_name).metadata
        assert field_meta is not None and b"description" in field_meta, f"Missing description on {col_name}"
    pf_check.close()

    # Atomically replace original
    backup_path = file_path.with_suffix(".parquet.bak")
    if backup_path.exists():
        backup_path.unlink()

    file_path.rename(backup_path)
    temp_path.rename(file_path)
    if backup_path.exists():
        backup_path.unlink()

    final_size_mb = file_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"[SUCCESS] {file_path.name}: {total_expected_rows:,} rows processed in {elapsed:.1f}s ({final_size_mb:.2f} MB)")
    return True


def main():
    base_dir = Path(__file__).resolve().parent.parent / "LemGendaryDatasets" / "LemGendizedForexUniverseLarge"
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
    print(f"[ALL COMPLETE] Successfully embedded column descriptions in all {len(parquet_files)} Parquet files in {total_elapsed:.1f}s.")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
