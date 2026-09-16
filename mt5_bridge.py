"""
MetaTrader5 IPC bridge.

MetaTrader5 ships no PEP 561 stubs — its compiled extension provides only
partial metadata that Pyrefly cannot fully introspect. This module is the
single import site; consumers should import typed helpers from here rather
than re-importing MetaTrader5.

`_mt5` is deliberately typed as `Any`: it lets the module's helper functions
carry concrete return types (bool, int, etc.) while bypassing Pyrefly's
incomplete view of the underlying MT5 binary.

Phase 1.5.7 of the 2026 modernization roadmap.
"""

from __future__ import annotations

from typing import Any

try:
    import MetaTrader5 as _mt5_module  # type: ignore[import-untyped]  # MT5 ships no PEP 561 stubs
    _mt5: Any = _mt5_module
except ImportError:
    _mt5 = None  # sentinel when MT5 is not installed on this host


# Timeframe constants — re-exported so consumers never touch _mt5 directly
TIMEFRAME_M1: int = int(getattr(_mt5, "TIMEFRAME_M1", 1))
TIMEFRAME_M5: int = int(getattr(_mt5, "TIMEFRAME_M5", 5))
TIMEFRAME_M15: int = int(getattr(_mt5, "TIMEFRAME_M15", 15))
TIMEFRAME_H1: int = int(getattr(_mt5, "TIMEFRAME_H1", 60))
TIMEFRAME_H4: int = int(getattr(_mt5, "TIMEFRAME_H4", 240))
TIMEFRAME_D1: int = int(getattr(_mt5, "TIMEFRAME_D1", 1440))


def initialize() -> bool:
    """Initialize the MT5 terminal connection. Returns False if MT5 is unavailable."""
    if _mt5 is None:
        return False
    return bool(_mt5.initialize())


def shutdown() -> None:
    """Cleanly close the MT5 terminal connection."""
    if _mt5 is not None:
        _mt5.shutdown()


def last_error() -> Any:
    """Return the last MT5 error tuple, or None if MT5 is unavailable."""
    return _mt5.last_error() if _mt5 is not None else None


def account_info() -> Any:
    return _mt5.account_info() if _mt5 is not None else None


def symbols_get(*args: Any, **kwargs: Any) -> Any:
    return _mt5.symbols_get(*args, **kwargs) if _mt5 is not None else None


def symbol_info(symbol: str) -> Any:
    return _mt5.symbol_info(symbol) if _mt5 is not None else None


def symbol_select(symbol: str, enable: bool = True) -> bool:
    if _mt5 is None:
        return False
    return bool(_mt5.symbol_select(symbol, enable))


def copy_rates_range(symbol: str, timeframe: int, start: Any, end: Any) -> Any:
    return _mt5.copy_rates_range(symbol, timeframe, start, end) if _mt5 is not None else None


def copy_rates_from_pos(symbol: str, timeframe: int, pos: int, count: int) -> Any:
    return _mt5.copy_rates_from_pos(symbol, timeframe, pos, count) if _mt5 is not None else None