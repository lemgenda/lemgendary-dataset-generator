"""
Technical indicator features for the MT5 download pipeline.

Uses ``pandas_ta_classic`` when it is installed and falls back to the
manual NumPy/pandas implementations below otherwise. All price-scale
indicators are divided by the close price so they are scale-free.
"""

from __future__ import annotations

import importlib
import logging

import numpy as np
import pandas as pd

__all__ = ["compute_indicators"]

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _first_column(columns, predicate):
    return next((c for c in columns if predicate(c.upper())), None)


def _is_macd_line(name):
    return 'MACD_' in name and 'H' not in name and 'S' not in name


def _is_atr(name):
    return name.startswith(('ATRR_', 'ATR'))


_TA_COLUMN_MATCHERS = {
    "rsi": lambda name: name.startswith('RSI_'),
    "macd": _is_macd_line,
    "macds": lambda name: 'MACDS_' in name,
    "atr": _is_atr,
    "bbu": lambda name: 'BBU_' in name,
    "bbl": lambda name: 'BBL_' in name,
}


def _as_f64(values):
    return np.asarray(values, dtype=np.float64)


def _ta_indicators(df, close):
    """Indicator columns from ``pandas_ta_classic``, or None when a column is missing.

    Raises ImportError when the package is not installed.
    """
    # Importing the optional package registers the `DataFrame.ta` accessor.
    importlib.import_module("pandas_ta_classic")
    df_ta = df.copy()
    df_ta.ta.rsi(length=14, append=True)
    df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)
    df_ta.ta.atr(length=14, append=True)
    df_ta.ta.bbands(length=20, std=2, append=True)

    cols = {
        key: _first_column(df_ta.columns, matcher) for key, matcher in _TA_COLUMN_MATCHERS.items()
    }
    if not all(cols.values()):
        return None
    scale = close + _EPS
    return {
        'rsi': (_as_f64(df_ta[cols["rsi"]]) / 100.0).tolist(),
        'macd': (_as_f64(df_ta[cols["macd"]]) / scale).tolist(),
        'macd_signal': (_as_f64(df_ta[cols["macds"]]) / scale).tolist(),
        'atr': (_as_f64(df_ta[cols["atr"]]) / scale).tolist(),
        'bb_width': (
            (_as_f64(df_ta[cols["bbu"]]) - _as_f64(df_ta[cols["bbl"]])) / scale
        ).tolist(),
    }


def _try_ta_indicators(df, close):
    try:
        return _ta_indicators(df, close)
    except ImportError as err:
        logger.debug("pandas_ta_classic unavailable, using manual indicators: %s", err)
    except (AttributeError, KeyError, ValueError, TypeError) as err:
        logger.warning(
            "pandas_ta_classic indicator computation failed, using manual indicators: %s", err
        )
    return None


def _manual_indicators(close, high, low):
    scale = close + _EPS
    macd_line, sig = _macd_manual(close, 12, 26, 9)
    return {
        'rsi': _rsi_manual(close, 14).tolist(),
        'macd': (macd_line / scale).tolist(),
        'macd_signal': (sig / scale).tolist(),
        'atr': (_atr_manual(high, low, close, 14) / scale).tolist(),
        'bb_width': (_bbwidth_manual(close, 20, 2) / scale).tolist(),
    }


def _clean(df):
    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)


def _add_session_features(df):
    if 'time' in df.columns:
        hours = pd.Series(pd.to_datetime(df['time'], utc=True)).dt.hour
    else:
        hours = pd.Series([0.0] * len(df))
    hour_norm = _as_f64(hours) / 23.0 * 2.0 * np.pi
    df['session_sin'] = np.sin(hour_norm).tolist()
    df['session_cos'] = np.cos(hour_norm).tolist()


def _add_volatility_features(df):
    high_arr = _as_f64(df['high'])
    low_arr = _as_f64(df['low'])
    raw_atr = _atr_manual(high_arr, low_arr, _as_f64(df['close']), 14)
    atr_series = pd.Series([float(v) for v in raw_atr])
    atr_pct = atr_series.rolling(100, min_periods=1).apply(
        lambda x: float(np.sum(x <= x[-1])) / float(len(x)), raw=True
    )
    df['atr_percentile'] = np.asarray(atr_pct, dtype=np.float32).tolist()
    df['bar_range_ratio'] = ((high_arr - low_arr) / (raw_atr + _EPS)).astype(np.float32).tolist()


def compute_indicators(df):
    """Return a copy of ``df`` with the technical indicator columns added."""
    df = df.copy()
    close = _as_f64(df['close'])

    indicators = _try_ta_indicators(df, close)
    if indicators is None:
        indicators = _manual_indicators(close, _as_f64(df['high']), _as_f64(df['low']))
    for name, values in indicators.items():
        df[name] = values

    _clean(df)
    _add_session_features(df)
    _add_volatility_features(df)
    _clean(df)
    return df


def _rsi_manual(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0).astype(np.float64)
    loss = np.where(delta < 0, -delta, 0.0).astype(np.float64)
    avg_g = _as_f64(pd.Series(gain.tolist()).ewm(alpha=1 / period, min_periods=period).mean())
    avg_l = _as_f64(pd.Series(loss.tolist()).ewm(alpha=1 / period, min_periods=period).mean())
    rs = avg_g / (avg_l + _EPS)
    return np.asarray(1.0 - 1.0 / (1.0 + rs), dtype=np.float64)


def _ema_manual(arr, span):
    return _as_f64(pd.Series(arr.tolist()).ewm(span=span, adjust=False).mean())


def _macd_manual(close, fast, slow, signal):
    macd_line = _ema_manual(close, fast) - _ema_manual(close, slow)
    sig_line = _ema_manual(macd_line, signal)
    return macd_line, sig_line


def _atr_manual(high, low, close, period=14):
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _as_f64(pd.Series(tr.tolist()).ewm(span=period, adjust=False).mean())


def _bbwidth_manual(close, period=20, std_mult=2.0):
    s = pd.Series(close.tolist())
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return _as_f64(upper - lower)
