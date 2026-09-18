#!/usr/bin/env python3
"""
LemGendary MT5 Data Pipeline.
Backwards-compatible shim module delegating to forex.pipeline.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forex.pipeline import (
    connect_mt5,
    disconnect_mt5,
    download_bars,
    resolve_symbol,
    compute_indicators,
    generate_labels,
    normalize_ohlcv,
    run_download_pipeline,
    main,
)

__all__ = [
    "connect_mt5",
    "disconnect_mt5",
    "download_bars",
    "resolve_symbol",
    "compute_indicators",
    "generate_labels",
    "normalize_ohlcv",
    "run_download_pipeline",
    "main",
]

if __name__ == "__main__":
    main()
