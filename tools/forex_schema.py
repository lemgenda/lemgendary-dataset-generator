"""
LemGendary Forex Manifold Schema & Column Descriptors Specification
===================================================================
Backwards-compatible shim module re-exporting from forex.schema.
"""

from forex.schema import (
    COLUMN_DESCRIPTIONS,
    EXTENDED_PAIRS,
    PARQUET_SCHEMA,
    TIMEFRAME_LOOKBACK,
)

__all__ = [
    "COLUMN_DESCRIPTIONS",
    "EXTENDED_PAIRS",
    "PARQUET_SCHEMA",
    "TIMEFRAME_LOOKBACK",
]
