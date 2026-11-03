"""
LemGendary Forex Manifold Pipeline Package
==========================================
Clean, encapsulated package providing PyArrow schemas, MetaTrader5
bridges, Parquet converters, metadata injectors, and historical
manifold compilation pipelines.

The package-level names below are loaded lazily (PEP 562): importing
``forex.schema`` or ``forex.converter`` does not pull in pandas through the
download pipeline. ``from forex import <name>`` keeps working for every name.
"""

from __future__ import annotations

import importlib
from typing import Any

# submodule -> names re-exported at package level under the same name
_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "schema": ("COLUMN_DESCRIPTIONS", "EXTENDED_PAIRS", "PARQUET_SCHEMA", "TIMEFRAME_LOOKBACK"),
    "bridge": (
        "TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_H1", "TIMEFRAME_H4",
        "TIMEFRAME_D1", "initialize", "shutdown", "last_error", "account_info", "symbols_get",
        "symbol_info", "symbol_select", "copy_rates_range", "copy_rates_from_pos",
    ),
    "converter": ("convert_year", "get_dir_size_gb", "get_file_size_gb"),
    "injector": ("process_parquet_file",),
    "pipeline": (
        "connect_mt5", "disconnect_mt5", "download_bars", "resolve_symbol",
        "compute_indicators", "generate_labels", "normalize_ohlcv", "run_download_pipeline",
    ),
}

# package-level name -> (submodule, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    name: (module, name) for module, names in _SUBMODULE_EXPORTS.items() for name in names
}
_EXPORTS["run_pipeline_main"] = ("pipeline", "main")

__all__: list[str] = []
__all__.extend(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _EXPORTS[name]
    value = getattr(importlib.import_module(f".{module_name}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
