"""
LemGendary Forex Manifold Schema & Column Descriptors Specification
===================================================================
Backwards-compatible shim module re-exporting from forex.schema.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
