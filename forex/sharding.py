"""
Windowing and resumable Parquet sharding for the MT5 download pipeline.

Cuts a labelled bar frame into fixed-length feature windows, groups them by
the year of their target bar and writes memory-bounded chunk shards under
``<out_dir>/ForexUniverse<year>/<pair>/<timeframe>/``. Existing shards are
skipped so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .pipeline_config import CHUNK_SIZE, FEATURES
from .shard_table import ShardLabels, build_shard_table, open_shard_writer

__all__ = [
    "ShardLabels",
    "ShardTarget",
    "build_windows_and_save_by_year",
    "consolidate_year_parquet",
    "save_shards",
    "year_chunk_complete",
]


@dataclass(frozen=True)
class ShardTarget:
    """Where a pair/timeframe's shards go: dataset output folder, pair and timeframe (minutes)."""

    out_dir: str
    pair: str
    timeframe_min: int


@dataclass(frozen=True)
class _WindowSource:
    """Arrays of a labelled frame that windows and targets are cut from."""

    features: np.ndarray
    direction: np.ndarray
    tp: np.ndarray
    sl: np.ndarray
    timestamps: np.ndarray
    seq_len: int


def _shard_suffix(chunk_suffix):
    return f"_chunk{chunk_suffix}" if chunk_suffix is not None else ""


def _shard_dir(out_dir, year, pair, tf):
    return os.path.join(out_dir, f"ForexUniverse{year}", pair, str(tf))


def year_chunk_complete(out_dir, year, pair, tf, chunk_suffix=None):
    """True when the year's Parquet file or the shard for this pair, timeframe and chunk exists."""
    year_parquet = os.path.join(out_dir, f"ForexUniverse{year}.parquet")
    if os.path.isfile(year_parquet):
        return True
    shard_dir = _shard_dir(out_dir, year, pair, tf)
    if not os.path.isdir(shard_dir):
        return False
    suffix = _shard_suffix(chunk_suffix)
    if os.path.isfile(os.path.join(shard_dir, f"shard{suffix}.parquet")):
        return True
    legacy_files = (f"X{suffix}.npy", f"y_dir{suffix}.npy", f"y_mag{suffix}.npy")
    return all(os.path.isfile(os.path.join(shard_dir, name)) for name in legacy_files)


def save_shards(features, labels: ShardLabels, target: ShardTarget, year_split, chunk_suffix=None):
    """Write one Parquet shard of feature windows and labels in its year/pair/timeframe folder."""
    shard_dir = _shard_dir(target.out_dir, year_split, target.pair, target.timeframe_min)
    os.makedirs(shard_dir, exist_ok=True)
    parquet_path = os.path.join(shard_dir, f"shard{_shard_suffix(chunk_suffix)}.parquet")

    table = build_shard_table(target.pair, target.timeframe_min, features, labels)
    pq.write_table(table, parquet_path, compression="zstd", compression_level=3)
    print(f" [MT5Pipeline] Parquet Shard Written -> {parquet_path} ({len(features)} samples)")


def consolidate_year_parquet(out_dir, year):
    """Consolidates all pair/tf parquet shards into single ForexUniverse{year}.parquet."""
    year_dir = os.path.join(out_dir, f"ForexUniverse{year}")
    final_parquet = os.path.join(out_dir, f"ForexUniverse{year}.parquet")
    if not os.path.isdir(year_dir):
        return
    shard_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(year_dir)
        for f in files
        if f.endswith(".parquet")
    ]
    if not shard_files:
        return
    print(f" [MT5Pipeline] Consolidating {len(shard_files)} shards into {final_parquet}...")
    writer = open_shard_writer(final_parquet)
    for sf in sorted(shard_files):
        tbl = pq.read_table(sf)
        writer.write_table(tbl, row_group_size=5000)
    writer.close()
    shutil.rmtree(year_dir, ignore_errors=True)
    print(f" [MT5Pipeline] Consolidated -> {final_parquet}")


def _window_source(df, seq_len):
    available = [c for c in FEATURES if c in df.columns]
    arrays = {
        "features": np.asarray(df[available], dtype=np.float32),
        "direction": np.asarray(df['direction'], dtype=np.int64),
        "tp": np.asarray(df['tp_pips'], dtype=np.float32),
        "sl": np.asarray(df['sl_pips'], dtype=np.float32),
    }
    if 'time' not in df.columns:
        raise ValueError("DataFrame missing explicit chronological 'time' axis.")
    df_time = pd.Series(pd.to_datetime(df['time'], utc=True))
    # Unix epoch seconds (UTC): datetime64[ns] integers divided by 10**9.
    timestamps = np.asarray(df_time.astype(np.int64) // 10**9, dtype=np.int64)
    return _WindowSource(timestamps=timestamps, seq_len=seq_len, **arrays), df_time


def _cut_windows(src: _WindowSource, chunk_indices):
    windows = np.empty((len(chunk_indices), src.seq_len, src.features.shape[1]), dtype=np.float32)
    for out_i, idx in enumerate(chunk_indices):
        windows[out_i] = src.features[idx : idx + src.seq_len]
    return windows


def _chunk_labels(src: _WindowSource, chunk_targets):
    return ShardLabels(
        y_dir=src.direction[chunk_targets],
        y_mag=np.stack([src.tp[chunk_targets], src.sl[chunk_targets]], axis=1),
        timestamps=src.timestamps[chunk_targets],
    )


def _save_year(src: _WindowSource, target: ShardTarget, year, year_indices):
    """Save the windows of one target year, in chunks of ``CHUNK_SIZE`` samples."""
    num_samples = len(year_indices)
    num_chunks = int(np.ceil(num_samples / CHUNK_SIZE))
    for c in range(num_chunks):
        chunk_indices = year_indices[c * CHUNK_SIZE : min((c + 1) * CHUNK_SIZE, num_samples)]
        chunk_suffix = c if num_chunks > 1 else None

        if year_chunk_complete(
            target.out_dir, str(year), target.pair, target.timeframe_min, chunk_suffix
        ):
            print(
                f" [RESUME] Skipping existing chunk {year}-{c} for"
                f" {target.pair}@{target.timeframe_min}"
            )
            continue

        save_shards(
            _cut_windows(src, chunk_indices),
            _chunk_labels(src, chunk_indices + src.seq_len),
            target,
            year_split=str(year),
            chunk_suffix=chunk_suffix,
        )


def build_windows_and_save_by_year(
    df, target: ShardTarget, seq_len, max_samples=50000, stride=None
):
    """Cut ``df`` into ``seq_len`` windows and save them as shards grouped by target year."""
    src, df_time = _window_source(df, seq_len)

    n_total = len(df) - seq_len
    if n_total <= 0:
        return

    if stride is None:
        stride = max(1, int(np.ceil(n_total / max_samples))) if max_samples > 0 else 1

    indices = np.arange(0, n_total, stride)
    sample_years = pd.Series(df_time.iloc[indices + seq_len]).dt.year.to_numpy()

    for year in np.unique(sample_years):
        year_indices = indices[sample_years == year]
        if len(year_indices) == 0:
            continue
        _save_year(src, target, year, year_indices)
