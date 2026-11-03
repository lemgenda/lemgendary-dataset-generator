"""
Training targets and OHLCV normalisation for the MT5 download pipeline.

``generate_labels`` adds the direction class and take-profit / stop-loss pip
targets over a forward horizon; ``normalize_ohlcv`` turns prices into log
returns and volume into a rolling z-score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline_config import (
    DIRECTION_THRESHOLD_PIPS,
    LABEL_HORIZON_BARS,
    PAIR_SPREADS_PIPS,
)

__all__ = ["generate_labels", "normalize_ohlcv", "pip_size_for"]

# (substrings, pip size) checked in order; the first pair that contains a substring wins.
_PIP_SIZE_RULES = (
    (("JPY",), 0.01),
    (("XAU",), 0.1),
    (("US500", "NAS100", "DE40"), 1.0),
    (("USOIL", "XAG"), 0.01),
)
_DEFAULT_PIP_SIZE = 0.0001


def pip_size_for(pair):
    """Price change of one pip for ``pair``."""
    for keys, size in _PIP_SIZE_RULES:
        if any(key in pair for key in keys):
            return size
    return _DEFAULT_PIP_SIZE


def _future_moves(df, horizon, pip_size):
    """Best up and down move in pips over the next ``horizon`` bars for each labelled bar."""
    close_arr = np.asarray(df['close'], dtype=np.float64)
    high_arr = np.asarray(df['high'], dtype=np.float64)
    low_arr = np.asarray(df['low'], dtype=np.float64)
    for i in range(len(df) - horizon):
        entry = close_arr[i]
        fut_high = high_arr[i + 1 : i + 1 + horizon].max()
        fut_low = low_arr[i + 1 : i + 1 + horizon].min()
        yield i, (fut_high - entry) / pip_size, (entry - fut_low) / pip_size


def generate_labels(df, pair, horizon=LABEL_HORIZON_BARS, threshold_pips=DIRECTION_THRESHOLD_PIPS):
    """Add direction (0 down, 1 flat, 2 up), TP and SL pip labels over ``horizon`` bars."""
    spread_pips = PAIR_SPREADS_PIPS.get(pair, 2.0)
    n = len(df)
    directions = np.ones(n, dtype=np.int64)
    tp_pips = np.zeros(n, dtype=np.float32)
    sl_pips = np.zeros(n, dtype=np.float32)

    for i, up_move, down_move in _future_moves(df, horizon, pip_size_for(pair)):
        tp_pips[i] = max(0.0, float(up_move) - spread_pips)
        sl_pips[i] = max(0.0, float(down_move) + spread_pips)

        if up_move >= threshold_pips and up_move > down_move:
            directions[i] = 2
        elif down_move >= threshold_pips and down_move > up_move:
            directions[i] = 0

    df = df.copy()
    df['direction'] = directions.tolist()
    df['tp_pips'] = tp_pips.tolist()
    df['sl_pips'] = sl_pips.tolist()
    return df


def normalize_ohlcv(df):
    """Return a copy of ``df`` with prices as log returns and volume as a rolling z-score."""
    df = df.copy()
    close = np.asarray(df['close'], dtype=np.float64)
    prev = np.roll(close, 1)
    prev[0] = close[0]

    for col in ['open', 'high', 'low', 'close']:
        df[col] = np.log(np.asarray(df[col], dtype=np.float64) / (prev + 1e-8)).tolist()

    vol = np.asarray(df['volume'], dtype=np.float64)
    vol_s = pd.Series(vol.tolist())
    df['volume'] = np.asarray(
        (vol_s - vol_s.rolling(50).mean()) / (vol_s.rolling(50).std() + 1e-8),
        dtype=np.float64,
    ).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df
