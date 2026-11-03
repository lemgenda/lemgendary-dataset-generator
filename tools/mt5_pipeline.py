#!/usr/bin/env python3
"""
LemGendary MT5 Data Pipeline.
Backwards-compatible shim module delegating to forex.pipeline.
"""

from forex.pipeline import (
    compute_indicators,
    connect_mt5,
    disconnect_mt5,
    download_bars,
    generate_labels,
    main,
    normalize_ohlcv,
    resolve_symbol,
    run_download_pipeline,
)

__all__ = [
    "compute_indicators",
    "connect_mt5",
    "disconnect_mt5",
    "download_bars",
    "generate_labels",
    "main",
    "normalize_ohlcv",
    "resolve_symbol",
    "run_download_pipeline",
]

if __name__ == "__main__":
    main()
