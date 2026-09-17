"""
MetaTrader5 IPC bridge.

Backwards-compatible shim module re-exporting from forex.bridge.
"""

from forex.bridge import (
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

__all__ = [
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
]