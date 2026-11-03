"""
Historical OHLCV bar download from MetaTrader 5.

Tries every candidate symbol for a pair (mapped, logical, resolved), first via
the range API and then via chunked ``copy_rates_from_pos`` as a fallback,
with retries and exponential back-off.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from . import bridge as mt5_bridge
from .mt5_session import mt5_call, remember_symbol, resolve_symbol
from .pipeline_config import MT5_SYMBOL_MAP, MT5_TIMEFRAMES

__all__ = ["download_bars"]

logger = logging.getLogger(__name__)

_DOWNLOAD_ERRORS = (RuntimeError, AttributeError, OSError, KeyError, ValueError, IndexError)
_FALLBACK_CHUNK_BARS = 100000
_FALLBACK_MAX_BARS = 5_000_000


@dataclass(frozen=True)
class _BarRequest:
    """One pair/timeframe download: MT5 timeframe constant and the date range."""

    pair: str
    timeframe_min: int
    start_date: str
    tf: int
    dt_from: datetime
    dt_to: datetime

    @property
    def dt_from_utc(self) -> pd.Timestamp:
        """``dt_from`` as a UTC timestamp, comparable with the bars' tz-aware ``time``.

        ``dt_from`` stays naive for the MT5 range API; the downloaded bars are
        converted with ``utc=True``.
        """
        return pd.Timestamp(self.dt_from, tz="UTC")


def _candidate_symbols(pair):
    """Mapped MT5 name first, then the logical name, then the dynamically resolved one."""
    preferred = MT5_SYMBOL_MAP.get(pair, pair)
    candidates = [preferred]
    if preferred != pair:
        candidates.append(pair)
    resolved = resolve_symbol(pair)
    if resolved and resolved not in candidates:
        candidates.append(resolved)
    return candidates


def _symbol_available(mt5_symbol):
    """Add ``mt5_symbol`` to Market Watch; False when the broker does not know it at all."""
    if _ensure_selected(mt5_symbol):
        return True
    if mt5_call("symbol_info", mt5_symbol) is None:
        print(f" [MT5] Symbol '{mt5_symbol}' not found. Trying next candidate.")
        return False
    print(
        f" [MT5] Symbol '{mt5_symbol}' exists but not in Market Watch; will attempt to"
        " fetch data anyway."
    )
    return True


def _ensure_selected(mt5_symbol):
    return bool(mt5_call("symbol_select", mt5_symbol, True))


def _rates_to_frame(rates):
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['volume'] = df['tick_volume']
    return df[['time', 'open', 'high', 'low', 'close', 'volume']]


def _span_text(df):
    time_s = pd.Series(df['time'])
    return f"{time_s.iloc[0].strftime('%Y-%m-%d')} -> {time_s.iloc[-1].strftime('%Y-%m-%d')}"


def _fetch_range(mt5_symbol, req):
    """Fetch the whole date range at once; None when the range API returns nothing."""
    print(
        f" [MT5] Fetching historical range for {mt5_symbol} {req.timeframe_min}min via"
        " Range API..."
    )
    rates = mt5_call("copy_rates_range", mt5_symbol, req.tf, req.dt_from, req.dt_to)
    if rates is None or len(rates) == 0:
        return None
    df = _rates_to_frame(rates)
    print(
        f" [MT5] Downloaded {len(df)} bars for {req.pair} ({mt5_symbol})"
        f" {req.timeframe_min}min (Spans {_span_text(df)})"
    )
    return df


def _collect_chunks(mt5_symbol, req):
    """Page backwards from the newest bar until ``dt_from`` or the bar cap is reached."""
    all_rates = []
    start_pos = 0
    total_fetched = 0
    while total_fetched < _FALLBACK_MAX_BARS:
        chunk = mt5_call(
            "copy_rates_from_pos", mt5_symbol, req.tf, start_pos, _FALLBACK_CHUNK_BARS
        )
        if chunk is None or len(chunk) == 0:
            break
        all_rates.append(chunk)
        total_fetched += len(chunk)
        start_pos += _FALLBACK_CHUNK_BARS
        oldest_in_chunk = pd.to_datetime(chunk[-1][0], unit='s', utc=True)
        print(
            f"   [MT5] Downloaded chunk: {len(chunk)} bars (total {total_fetched}), "
            f"oldest: {oldest_in_chunk.strftime('%Y-%m-%d')}",
            end='\r',
        )
        if oldest_in_chunk <= req.dt_from_utc:
            break
    return all_rates


def _fetch_chunked(mt5_symbol, req):
    """Fallback download through ``copy_rates_from_pos``; raises RuntimeError on no data."""
    print(
        f" [MT5] Range returned empty. Falling back to chunked from_pos "
        f"for {mt5_symbol} {req.timeframe_min}min..."
    )
    all_rates = _collect_chunks(mt5_symbol, req)
    if not all_rates:
        err = mt5_call("last_error")
        raise RuntimeError(f"No data returned from fallback. MT5 Error: {err}")

    rates = np.concatenate(all_rates)
    df = _rates_to_frame(rates)
    df = df[df['time'] >= req.dt_from_utc]
    if len(df) == 0:
        print(
            f" [MT5] No bars after filtering from {req.start_date}; using all available"
            " data."
        )
        df = _rates_to_frame(rates)

    print(
        f"\n [MT5] Downloaded {len(df)} bars for {req.pair} ({mt5_symbol})"
        f" {req.timeframe_min}min via fallback (Spans {_span_text(df)})"
    )
    return df


def _fetch_with_retries(mt5_symbol, req, retries):
    """Download bars for one symbol, retrying with back-off; None after the last failure."""
    for attempt in range(retries):
        try:
            df = _fetch_range(mt5_symbol, req)
            if df is None:
                df = _fetch_chunked(mt5_symbol, req)
            remember_symbol(req.pair, mt5_symbol)
            return df
        except _DOWNLOAD_ERRORS as e:
            logger.warning(
                "MT5 download attempt %d for %s failed: %s", attempt + 1, mt5_symbol, e
            )
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(
                    f" [MT5] Download attempt {attempt + 1} failed: {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                print(
                    f" [MT5] Failed to download for {mt5_symbol} after {retries} attempts: {e}"
                )
    return None


def download_bars(pair, timeframe_min, start_date="2019-01-01", retries=3):
    """Download one pair and timeframe of OHLCV bars from MT5 since ``start_date``, with retries."""
    tf_attr = MT5_TIMEFRAMES.get(timeframe_min)
    if tf_attr is None:
        raise ValueError(f"Unsupported timeframe: {timeframe_min}min")
    req = _BarRequest(
        pair=pair,
        timeframe_min=timeframe_min,
        start_date=start_date,
        tf=mt5_bridge.get_timeframe_const(tf_attr),
        dt_from=datetime.strptime(start_date, "%Y-%m-%d"),
        dt_to=datetime.now(),
    )

    for mt5_symbol in _candidate_symbols(pair):
        print(f" [MT5] Trying symbol '{mt5_symbol}' for pair '{pair}'")
        if not _symbol_available(mt5_symbol):
            continue
        df = _fetch_with_retries(mt5_symbol, req, retries)
        if df is not None:
            return df

    print(f" [MT5] WARNING: No symbol found for pair {pair}. Skipping.")
    return None
