"""
LemGendary Forex Manifold Pipeline Package
==========================================
Clean, encapsulated package providing PyArrow schemas, MetaTrader5
bridges, Parquet converters, metadata injectors, and historical
manifold compilation pipelines.
"""

from __future__ import annotations

from .schema import (
    COLUMN_DESCRIPTIONS,
    EXTENDED_PAIRS,
    PARQUET_SCHEMA,
    TIMEFRAME_LOOKBACK,
)
from .bridge import (
    TIMEFRAME_D1,
    TIMEFRAME_H1,
    TIMEFRAME_H4,
    TIMEFRAME_M1,
    TIMEFRAME_M5,
    TIMEFRAME_M15,
    account_info,
    copy_rates_from_pos,
    copy_rates_range,
    initialize,
    last_error,
    shutdown,
    symbol_info,
    symbol_select,
    symbols_get,
)
from .converter import (
    convert_year,
    get_dir_size_gb,
    get_file_size_gb,
)
from .injector import (
    process_parquet_file,
)
from .pipeline import (
    connect_mt5,
    disconnect_mt5,
    download_bars,
    resolve_symbol,
    compute_indicators,
    generate_labels,
    normalize_ohlcv,
    run_download_pipeline,
    main as run_pipeline_main,
)

__all__ = [
    "COLUMN_DESCRIPTIONS",
    "EXTENDED_PAIRS",
    "PARQUET_SCHEMA",
    "TIMEFRAME_LOOKBACK",
    "TIMEFRAME_M1",
    "TIMEFRAME_M5",
    "TIMEFRAME_M15",
    "TIMEFRAME_H1",
    "TIMEFRAME_H4",
    "TIMEFRAME_D1",
    "initialize",
    "shutdown",
    "last_error",
    "account_info",
    "symbols_get",
    "symbol_info",
    "symbol_select",
    "copy_rates_range",
    "copy_rates_from_pos",
    "convert_year",
    "get_dir_size_gb",
    "get_file_size_gb",
    "process_parquet_file",
    "connect_mt5",
    "disconnect_mt5",
    "download_bars",
    "resolve_symbol",
    "compute_indicators",
    "generate_labels",
    "normalize_ohlcv",
    "run_download_pipeline",
    "run_pipeline_main",
]
