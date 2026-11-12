"""
LemGendary Forex Manifold Schema & Column Descriptors Specification
===================================================================
Standardized PyArrow schema, data types, and column descriptions
shared across Forex conversion and metadata injection pipelines.
"""

import json
from pathlib import Path

import pyarrow as pa

from .columns import COLUMN_DESCRIPTIONS, TIMESTAMP_UNIT

TIMEFRAME_LOOKBACK = {
    1: 512,
    5: 288,
    15: 192,
    60: 168,
    240: 90,
    1440: 252,
}

EXTENDED_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD",
    "USDCAD", "USDCHF", "AUDUSD", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP",
    "XAGUSD", "USOIL",
    "US500", "NAS100", "DE40"
]

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("pair", pa.string(), metadata={"description": COLUMN_DESCRIPTIONS["pair"]}),
        pa.field(
            "timeframe", pa.int32(), metadata={"description": COLUMN_DESCRIPTIONS["timeframe"]}
        ),
        pa.field(
            "timestamp", pa.int64(), metadata={"description": COLUMN_DESCRIPTIONS["timestamp"]}
        ),
        pa.field("y_dir", pa.int8(), metadata={"description": COLUMN_DESCRIPTIONS["y_dir"]}),
        pa.field("tp_pips", pa.float32(), metadata={"description": COLUMN_DESCRIPTIONS["tp_pips"]}),
        pa.field("sl_pips", pa.float32(), metadata={"description": COLUMN_DESCRIPTIONS["sl_pips"]}),
        pa.field("seq_len", pa.int16(), metadata={"description": COLUMN_DESCRIPTIONS["seq_len"]}),
        pa.field(
            "n_features", pa.int16(), metadata={"description": COLUMN_DESCRIPTIONS["n_features"]}
        ),
        pa.field(
            "features", pa.binary(), metadata={"description": COLUMN_DESCRIPTIONS["features"]}
        ),
    ],
    metadata={
        b"description": b"LemGendary Forex Universe High-Fidelity OHLCV Temporal Manifold",
        b"columns": json.dumps(COLUMN_DESCRIPTIONS).encode("utf-8"),
        b"domain": b"Financial & Time-Series",
        b"timestamp_unit": TIMESTAMP_UNIT.encode("utf-8"),
        b"task": b"forex_prediction",
        b"timeframe_rungs": b"[1, 5, 15, 60, 240, 1440]",
        b"features_list": (
            b'["open", "high", "low", "close", "volume", "rsi", "macd", "macd_signal", "atr",'
            b' "bb_width", "session_sin", "session_cos", "atr_percentile", "bar_range_ratio"]'
        ),
        b"author": b"LemGendary AI",
        b"created_by": b"LemGendary MT5 Compiler Pipeline",
    },
)


def default_manifold_dir() -> Path:
    """Default ``LemGendizedForexUniverse`` folder next to the workspace repos.

    Falls back to the legacy ``LemGendizedForexUniverseLarge`` folder when only
    that one exists.
    """
    datasets_root = Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets"
    default_dir = datasets_root / "LemGendizedForexUniverse"
    legacy_dir = datasets_root / "LemGendizedForexUniverseLarge"
    if not default_dir.exists() and legacy_dir.exists():
        return legacy_dir
    return default_dir
