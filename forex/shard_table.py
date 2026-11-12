"""
Arrow table and Parquet writer helpers for Forex manifold shards.

Single place that turns feature windows plus labels into a ``PARQUET_SCHEMA``
table, shared by the MT5 pipeline sharding and the legacy .npy converter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import PARQUET_SCHEMA

__all__ = [
    "DEFAULT_N_FEATURES",
    "ShardLabels",
    "build_shard_table",
    "open_shard_writer",
    "to_epoch_seconds",
]

DEFAULT_N_FEATURES = 14
# Largest plausible epoch value in seconds (year 5138); larger values are in a finer unit.
_SECONDS_CEILING = 10**11
_UNIT_STEP = 1000


def to_epoch_seconds(timestamps: Any) -> np.ndarray:
    """Whole Unix epoch seconds from timestamps in s, ms, us or ns (int64 array).

    The ``timestamp`` column is always seconds (``forex.columns.TIMESTAMP_UNIT``);
    legacy ``timestamps.npy`` files in a finer unit are scaled down.
    """
    values = np.asarray(timestamps, dtype=np.int64)
    if values.size == 0:
        return values
    peak = int(np.abs(values).max())
    divisor = 1
    while peak // divisor >= _SECONDS_CEILING:
        divisor *= _UNIT_STEP
    return values // divisor if divisor > 1 else values


@dataclass(frozen=True)
class ShardLabels:
    """Per-sample targets of a shard: direction, (tp, sl) magnitudes and epoch timestamps.

    ``timestamps`` may be in any epoch unit; the table stores seconds.
    """

    y_dir: Any
    y_mag: Any
    timestamps: Any = None


def build_shard_table(pair, timeframe, features, labels: ShardLabels, default_seq_len=168):
    """Build a ``PARQUET_SCHEMA`` table from ``(samples, seq_len, n_features)`` windows.

    Timestamps are stored as Unix epoch seconds; missing ones as zeros.
    """
    num_samples = len(features)
    seq_len = features.shape[1] if len(features.shape) > 1 else default_seq_len
    n_features = features.shape[2] if len(features.shape) > 2 else DEFAULT_N_FEATURES
    if labels.timestamps is None:
        timestamps = np.zeros(num_samples, dtype=np.int64)
    else:
        timestamps = to_epoch_seconds(labels.timestamps)

    x_bytes = [sample.tobytes() for sample in features]
    return pa.Table.from_arrays(
        [
            pa.array([pair] * num_samples, type=pa.string()),
            pa.array([timeframe] * num_samples, type=pa.int32()),
            pa.array(timestamps, type=pa.int64()),
            pa.array(labels.y_dir, type=pa.int8()),
            pa.array(labels.y_mag[:, 0], type=pa.float32()),
            pa.array(labels.y_mag[:, 1], type=pa.float32()),
            pa.array([seq_len] * num_samples, type=pa.int16()),
            pa.array([n_features] * num_samples, type=pa.int16()),
            pa.array(x_bytes, type=pa.binary()),
        ],
        schema=PARQUET_SCHEMA,
    )


def open_shard_writer(path, **options):
    """Open a zstd-3 ``ParquetWriter`` with ``PARQUET_SCHEMA`` at ``path``."""
    return pq.ParquetWriter(
        str(path), schema=PARQUET_SCHEMA, compression="zstd", compression_level=3, **options
    )
