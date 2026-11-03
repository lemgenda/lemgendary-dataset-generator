#!/usr/bin/env python3
"""
LemGendary MT5 Data Pipeline.
Backwards-compatible shim module delegating to forex.pipeline.
"""

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
