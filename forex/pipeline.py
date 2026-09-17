"""
LemGendary MT5 Data Pipeline v5.4 (Final - NAS100 renamed, Disk-Error Resilient)
====================================================================================
- Uses dynamic symbol resolution (exact, alternatives, fuzzy).
- Falls back through multiple candidates (mapped, logical, resolved).
- Caches resolved symbols per pair.
- Skips gracefully if symbol unavailable.
- Resumes compilation by checking existing year chunks.
- Memory-safe chunking (CHUNK_SIZE = 20000).
- Handles disk write errors (OSError) with retries and skip.
- Logical pair "NAS100" is now called "NAS100".
"""

import os
import sys
import json
import importlib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, List, Dict, Optional
import time
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import COLUMN_DESCRIPTIONS, PARQUET_SCHEMA
FEATURES = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal", "atr", "bb_width",
    "session_sin", "session_cos", "atr_percentile", "bar_range_ratio"
]

LABEL_HORIZON_BARS = 20
DIRECTION_THRESHOLD_PIPS = 5

PAIR_SPREADS_PIPS = {
    "EURUSD": 1.2,  "GBPUSD": 1.5,  "USDJPY": 1.3,  "XAUUSD": 35.0,
    "USDCAD": 1.8,  "USDCHF": 1.5,  "AUDUSD": 1.4,  "NZDUSD": 2.0,
    "EURGBP": 1.5,  "EURJPY": 1.8,  "GBPJPY": 2.5,  "USOIL":  40.0,
    "US500":  10.0, "NAS100": 20.0,
    "DE40":   10.0,  "XAGUSD": 20.0,
}

TIMEFRAME_LOOKBACK = {1: 512, 5: 288, 15: 192, 60: 168, 240: 90, 1440: 252}
MT5_TIMEFRAMES = {
    1: "TIMEFRAME_M1", 5: "TIMEFRAME_M5", 15: "TIMEFRAME_M15",
    60: "TIMEFRAME_H1", 240: "TIMEFRAME_H4", 1440: "TIMEFRAME_D1"
}

# ─── Symbol Mapping (logical -> preferred MT5 name) ─────────────────────
MT5_SYMBOL_MAP = {
    "USOIL": "WTI",      # oil is WTI in this broker
    "NAS100": "NAS100",  # optional, explicit
    # add others if needed
}

# ─── Alternative keywords for fuzzy resolution ──────────────────────────
ALTERNATIVE_KEYWORDS = {
    "USOIL": ["WTI", "OIL"],
    "US500": ["SPX", "SP500", "SPX500"],
    "NAS100": ["US100", "NDX"],   # logical name is now NAS100
    "DE40": ["DAX", "GER40"],
    "XAGUSD": ["SILVER", "XAG"],
    "XAUUSD": ["GOLD", "XAU"],
}

CHUNK_SIZE = 20000

# ─── MT5 Connection ────────────────────────────────────────────────────────

def _mt5_call(method_name: str, *args: Any, **kwargs: Any) -> Any:
    try:
        mt5_mod = importlib.import_module("MetaTrader5")
    except ImportError as exc:
        raise RuntimeError(
            "[MT5] MetaTrader5 package not installed. Run: pip install MetaTrader5"
        ) from exc
    func = getattr(mt5_mod, method_name, None)
    if func is None:
        raise AttributeError(f"[MT5] Method '{method_name}' not found on MetaTrader5")
    return func(*args, **kwargs)


def connect_mt5(login=None, password=None, server=None, api_key=None):
    try:
        if _mt5_call("initialize"):
            info = _mt5_call("account_info")
            if info:
                print(f" [MT5] Connected -> Account: {info.login} | Server: {info.server} | Balance: {info.balance} {info.currency}")
                return True
    except Exception as e:
        print(f" [MT5] Failed to attach to terminal: {e}")

    if login is None and "MT5_LOGIN" in os.environ:
        try:
            login = int(os.environ["MT5_LOGIN"])
        except ValueError as err:
            print(f" [MT5] Warning: Invalid MT5_LOGIN integer format: {err}")
    if password is None:
        password = os.environ.get("MT5_PASSWORD")
    if server is None:
        server = os.environ.get("MT5_SERVER")
    if api_key is None:
        api_key = os.environ.get("MT5_API_KEY")

    init_kwargs = {}
    if login and password and server:
        init_kwargs.update({"login": login, "password": password, "server": server})
        try:
            if _mt5_call("initialize", **init_kwargs):
                info = _mt5_call("account_info")
                if info:
                    print(f" [MT5] Connected -> Account: {info.login} | Server: {info.server} | Balance: {info.balance} {info.currency}")
                    return True
                raise RuntimeError("[MT5] Connected but no account info.")
            err = _mt5_call("last_error")
            raise RuntimeError(f"[MT5] Initialize failed: {err}")
        except Exception as e:
            raise RuntimeError(f"[MT5] Initialize exception: {e}") from e
    else:
        raise RuntimeError(
            "[MT5] Could not connect to MetaTrader 5.\n"
            "Please ensure:\n"
            "  1. MetaTrader 5 is running and logged in\n"
            "  2. If using a demo account, it is active\n"
            "  3. If you need explicit login, set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER environment variables"
        )


def disconnect_mt5():
    try:
        _mt5_call("shutdown")
        print(" [MT5] Disconnected.")
    except Exception as err:
        print(f" [MT5] Warning during shutdown: {err}")


# ─── Symbol Resolution (dynamic, caches results) ────────────────────────

_SYMBOL_CACHE = {}  # pair -> resolved symbol name


def _find_matching_symbol(all_symbols, candidates):
    for sym in all_symbols:
        if sym.name.upper() in candidates:
            return sym.name

    for alt in candidates:
        if len(alt) < 3:
            continue
        for sym in all_symbols:
            name_upper = sym.name.upper()
            if name_upper.startswith(alt) and (len(name_upper) == len(alt) or name_upper[len(alt)] in ('.', '_', '-')):
                return sym.name

    for alt in candidates:
        if len(alt) < 3:
            continue
        for sym in all_symbols:
            if alt in sym.name.upper():
                return sym.name
    return None


def resolve_symbol(pair: str) -> Optional[str]:
    """Find the actual MT5 symbol name for a given logical pair."""
    if pair in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[pair]

    try:
        all_symbols = _mt5_call("symbols_get")
    except Exception:
        return None

    if not all_symbols:
        return None

    pair_upper = pair.upper()
    candidates = [pair_upper] + [alt.upper() for alt in ALTERNATIVE_KEYWORDS.get(pair, [])]
    matched = _find_matching_symbol(all_symbols, candidates)
    if matched:
        _SYMBOL_CACHE[pair] = matched
        return matched
    return None


# ─── Data Download with Fallbacks ────────────────────────────────────────

def download_bars(pair, timeframe_min, start_date="2019-01-01", retries=3):
    tf_attr = MT5_TIMEFRAMES.get(timeframe_min)
    if tf_attr is None:
        raise ValueError(f"Unsupported timeframe: {timeframe_min}min")
    try:
        import MetaTrader5 as mt5
        tf = getattr(mt5, tf_attr)
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package not installed. Run: pip install MetaTrader5") from exc

    dt_from = datetime.strptime(start_date, "%Y-%m-%d")
    dt_to = datetime.now()

    # Build candidate list: mapped -> logical -> resolved
    preferred = MT5_SYMBOL_MAP.get(pair, pair)
    candidates = [preferred]
    if preferred != pair:
        candidates.append(pair)
    resolved = resolve_symbol(pair)
    if resolved and resolved not in candidates:
        candidates.append(resolved)

    # Try each candidate until we get data
    for mt5_symbol in candidates:
        print(f" [MT5] Trying symbol '{mt5_symbol}' for pair '{pair}'")
        selected = _mt5_call("symbol_select", mt5_symbol, True)
        if not selected:
            info = _mt5_call("symbol_info", mt5_symbol)
            if info is None:
                print(f" [MT5] Symbol '{mt5_symbol}' not found. Trying next candidate.")
                continue
            print(f" [MT5] Symbol '{mt5_symbol}' exists but not in Market Watch; will attempt to fetch data anyway.")

        # Try to fetch data
        for attempt in range(retries):
            try:
                print(f" [MT5] Fetching historical range for {mt5_symbol} {timeframe_min}min via Range API...")
                rates = _mt5_call("copy_rates_range", mt5_symbol, tf, dt_from, dt_to)

                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                    df['volume'] = df['tick_volume']
                    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
                    time_s = pd.Series(df['time'])
                    print(f" [MT5] Downloaded {len(df)} bars for {pair} ({mt5_symbol}) {timeframe_min}min "
                          f"(Spans {time_s.iloc[0].strftime('%Y-%m-%d')} -> {time_s.iloc[-1].strftime('%Y-%m-%d')})")
                    # Cache the successful symbol for this pair
                    _SYMBOL_CACHE[pair] = mt5_symbol
                    return df

                # Fallback: chunked download
                print(f" [MT5] Range returned empty. Falling back to chunked from_pos for {mt5_symbol} {timeframe_min}min...")
                all_rates = []
                start_pos = 0
                chunk_size = 100000
                total_fetched = 0
                max_bars = 5_000_000

                while total_fetched < max_bars:
                    chunk = _mt5_call("copy_rates_from_pos", mt5_symbol, tf, start_pos, chunk_size)
                    if chunk is None or len(chunk) == 0:
                        break
                    all_rates.append(chunk)
                    total_fetched += len(chunk)
                    start_pos += chunk_size
                    oldest_in_chunk = pd.to_datetime(chunk[-1][0], unit='s', utc=True)
                    print(f"   [MT5] Downloaded chunk: {len(chunk)} bars (total {total_fetched}), oldest: {oldest_in_chunk.strftime('%Y-%m-%d')}", end='\r')
                    if oldest_in_chunk <= dt_from:
                        break

                if not all_rates:
                    err = _mt5_call("last_error")
                    raise RuntimeError(f"No data returned from fallback. MT5 Error: {err}")

                rates = np.concatenate(all_rates)
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df['volume'] = df['tick_volume']
                df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
                df = df[df['time'] >= dt_from]
                if len(df) == 0:
                    print(f" [MT5] No bars after filtering from {start_date}; using all available data.")
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                    df['volume'] = df['tick_volume']
                    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

                time_s = pd.Series(df['time'])
                print(f"\n [MT5] Downloaded {len(df)} bars for {pair} ({mt5_symbol}) {timeframe_min}min via fallback "
                      f"(Spans {time_s.iloc[0].strftime('%Y-%m-%d')} -> {time_s.iloc[-1].strftime('%Y-%m-%d')})")
                _SYMBOL_CACHE[pair] = mt5_symbol
                return df

            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f" [MT5] Download attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f" [MT5] Failed to download for {mt5_symbol} after {retries} attempts: {e}")
                    # Break out of retry loop to try next candidate
                    break

    # If all candidates fail
    print(f" [MT5] WARNING: No symbol found for pair {pair}. Skipping.")
    return None


# ─── Technical Indicators ────────────────────────────────────────────────

def compute_indicators(df):
    df = df.copy()
    close = np.asarray(df['close'], dtype=np.float64)
    high = np.asarray(df['high'], dtype=np.float64)
    low = np.asarray(df['low'], dtype=np.float64)

    computed_ta = False
    try:
        import pandas_ta_classic as ta
        df_ta = df.copy()
        df_ta.ta.rsi(length=14, append=True)
        df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)
        df_ta.ta.atr(length=14, append=True)
        df_ta.ta.bbands(length=20, std=2, append=True)

        rsi_col = next((c for c in df_ta.columns if c.upper().startswith('RSI_')), None)
        macd_col = next((c for c in df_ta.columns if 'MACD_' in c.upper() and 'H' not in c.upper() and 'S' not in c.upper()), None)
        macds_col = next((c for c in df_ta.columns if 'MACDS_' in c.upper()), None)
        atr_col = next((c for c in df_ta.columns if c.upper().startswith('ATRR_') or c.upper().startswith('ATR')), None)
        bbu_col = next((c for c in df_ta.columns if 'BBU_' in c.upper()), None)
        bbl_col = next((c for c in df_ta.columns if 'BBL_' in c.upper()), None)

        if rsi_col and macd_col and macds_col and atr_col and bbu_col and bbl_col:
            df['rsi'] = (np.asarray(df_ta[rsi_col], dtype=np.float64) / 100.0).tolist()
            df['macd'] = (np.asarray(df_ta[macd_col], dtype=np.float64) / (close + 1e-8)).tolist()
            df['macd_signal'] = (np.asarray(df_ta[macds_col], dtype=np.float64) / (close + 1e-8)).tolist()
            df['atr'] = (np.asarray(df_ta[atr_col], dtype=np.float64) / (close + 1e-8)).tolist()
            df['bb_width'] = ((np.asarray(df_ta[bbu_col], dtype=np.float64) - np.asarray(df_ta[bbl_col], dtype=np.float64)) / (close + 1e-8)).tolist()
            computed_ta = True
    except Exception:
        computed_ta = False

    if not computed_ta:
        df['rsi'] = _rsi_manual(close, 14).tolist()
        macd_line, sig = _macd_manual(close, 12, 26, 9)
        df['macd'] = (macd_line / (close + 1e-8)).tolist()
        df['macd_signal'] = (sig / (close + 1e-8)).tolist()
        df['atr'] = (_atr_manual(high, low, close, 14) / (close + 1e-8)).tolist()
        df['bb_width'] = (_bbwidth_manual(close, 20, 2) / (close + 1e-8)).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)

    if 'time' in df.columns:
        hours = pd.to_datetime(df['time'], utc=True).dt.hour
    else:
        hours = pd.Series([0.0] * len(df))
    hour_norm = np.asarray(hours, dtype=np.float64) / 23.0 * 2.0 * np.pi
    df['session_sin'] = np.sin(hour_norm).tolist()
    df['session_cos'] = np.cos(hour_norm).tolist()

    raw_atr = _atr_manual(np.asarray(df['high'], dtype=np.float64),
                          np.asarray(df['low'], dtype=np.float64),
                          np.asarray(df['close'], dtype=np.float64), 14)
    atr_series = pd.Series(list(raw_atr), dtype=float)
    atr_pct = atr_series.rolling(100, min_periods=1).apply(
        lambda x: float(np.sum(x <= x[-1])) / float(len(x)), raw=True
    )
    df['atr_percentile'] = np.asarray(atr_pct, dtype=np.float32).tolist()

    high_arr = np.asarray(df['high'], dtype=np.float64)
    low_arr = np.asarray(df['low'], dtype=np.float64)
    df['bar_range_ratio'] = ((high_arr - low_arr) / (raw_atr + 1e-8)).astype(np.float32).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df


def _rsi_manual(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0).astype(np.float64)
    loss = np.where(delta < 0, -delta, 0.0).astype(np.float64)
    avg_g = np.asarray(pd.Series(gain.tolist()).ewm(alpha=1/period, min_periods=period).mean(), dtype=np.float64)
    avg_l = np.asarray(pd.Series(loss.tolist()).ewm(alpha=1/period, min_periods=period).mean(), dtype=np.float64)
    rs = avg_g / (avg_l + 1e-8)
    return np.asarray(1.0 - 1.0 / (1.0 + rs), dtype=np.float64)


def _ema_manual(arr, span):
    return np.asarray(pd.Series(arr.tolist()).ewm(span=span, adjust=False).mean(), dtype=np.float64)


def _macd_manual(close, fast, slow, signal):
    macd_line = _ema_manual(close, fast) - _ema_manual(close, slow)
    sig_line = _ema_manual(macd_line, signal)
    return macd_line, sig_line


def _atr_manual(high, low, close, period=14):
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return np.asarray(pd.Series(tr.tolist()).ewm(span=period, adjust=False).mean(), dtype=np.float64)


def _bbwidth_manual(close, period=20, std_mult=2.0):
    s = pd.Series(close.tolist())
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return np.asarray((upper - lower), dtype=np.float64)


# ─── Labels ────────────────────────────────────────────────────────────────

def generate_labels(df, pair, horizon=LABEL_HORIZON_BARS, threshold_pips=DIRECTION_THRESHOLD_PIPS):
    if any(x in pair for x in ["JPY"]):
        pip_size = 0.01
    elif "XAU" in pair:
        pip_size = 0.1
    elif any(x in pair for x in ["US500", "NAS100", "DE40"]):
        pip_size = 1.0
    elif "USOIL" in pair or "XAG" in pair:
        pip_size = 0.01
    else:
        pip_size = 0.0001

    spread_pips = PAIR_SPREADS_PIPS.get(pair, 2.0)

    close_arr = np.asarray(df['close'], dtype=np.float64)
    high_arr = np.asarray(df['high'], dtype=np.float64)
    low_arr = np.asarray(df['low'], dtype=np.float64)
    n = len(df)

    directions = np.ones(n, dtype=np.int64)
    tp_pips = np.zeros(n, dtype=np.float32)
    sl_pips = np.zeros(n, dtype=np.float32)

    for i in range(n - horizon):
        entry = close_arr[i]
        fut_high = high_arr[i+1 : i+1+horizon].max()
        fut_low = low_arr[i+1 : i+1+horizon].min()
        up_move = (fut_high - entry) / pip_size
        down_move = (entry - fut_low) / pip_size

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


# ─── Normalisation ────────────────────────────────────────────────────────

def normalize_ohlcv(df):
    df = df.copy()
    close = np.asarray(df['close'], dtype=np.float64)
    prev = np.roll(close, 1)
    prev[0] = close[0]

    for col in ['open', 'high', 'low', 'close']:
        df[col] = np.log(np.asarray(df[col], dtype=np.float64) / (prev + 1e-8)).tolist()

    vol = np.asarray(df['volume'], dtype=np.float64)
    vol_s = pd.Series(vol.tolist())
    df['volume'] = np.asarray((vol_s - vol_s.rolling(50).mean()) / (vol_s.rolling(50).std() + 1e-8), dtype=np.float64).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df


# ─── Sharding with Resumption and Parquet Compilation ────────────────────

def year_chunk_complete(out_dir, year, pair, tf, chunk_suffix=None):
    year_parquet = os.path.join(out_dir, f"ForexUniverse{year}.parquet")
    if os.path.isfile(year_parquet):
        return True
    shard_dir = os.path.join(out_dir, f"ForexUniverse{year}", pair, str(tf))
    if not os.path.isdir(shard_dir):
        return False
    suffix = f"_chunk{chunk_suffix}" if chunk_suffix is not None else ""
    parquet_shard = os.path.join(shard_dir, f"shard{suffix}.parquet")
    if os.path.isfile(parquet_shard):
        return True
    x_file = os.path.join(shard_dir, f"X{suffix}.npy")
    y_dir_file = os.path.join(shard_dir, f"y_dir{suffix}.npy")
    y_mag_file = os.path.join(shard_dir, f"y_mag{suffix}.npy")
    return os.path.isfile(x_file) and os.path.isfile(y_dir_file) and os.path.isfile(y_mag_file)


def save_shards(X, y_dir, y_mag, out_dir, pair, timeframe_min, year_split,
                timestamps=None, chunk_suffix=None):
    shard_dir = os.path.join(out_dir, f"ForexUniverse{year_split}", pair, str(timeframe_min))
    os.makedirs(shard_dir, exist_ok=True)

    suffix = f"_chunk{chunk_suffix}" if chunk_suffix is not None else ""
    parquet_path = os.path.join(shard_dir, f"shard{suffix}.parquet")

    num_samples = len(X)
    seq_len = X.shape[1] if len(X.shape) > 1 else 168
    n_features = X.shape[2] if len(X.shape) > 2 else 14
    ts_arr = timestamps if timestamps is not None else np.zeros(num_samples, dtype=np.int64)

    x_bytes = [sample.tobytes() for sample in X]
    table = pa.Table.from_arrays([
        pa.array([pair] * num_samples, type=pa.string()),
        pa.array([timeframe_min] * num_samples, type=pa.int32()),
        pa.array(ts_arr, type=pa.int64()),
        pa.array(y_dir, type=pa.int8()),
        pa.array(y_mag[:, 0], type=pa.float32()),
        pa.array(y_mag[:, 1], type=pa.float32()),
        pa.array([seq_len] * num_samples, type=pa.int16()),
        pa.array([n_features] * num_samples, type=pa.int16()),
        pa.array(x_bytes, type=pa.binary()),
    ], schema=PARQUET_SCHEMA)

    pq.write_table(table, parquet_path, compression="zstd", compression_level=3)
    print(f" [MT5Pipeline] Parquet Shard Written -> {parquet_path} ({num_samples} samples)")


def consolidate_year_parquet(out_dir, year):
    """Consolidates all pair/tf parquet shards into single ForexUniverse{year}.parquet."""
    year_dir = os.path.join(out_dir, f"ForexUniverse{year}")
    final_parquet = os.path.join(out_dir, f"ForexUniverse{year}.parquet")
    if not os.path.isdir(year_dir):
        return
    shard_files = []
    for root, _, files in os.walk(year_dir):
        for f in files:
            if f.endswith(".parquet"):
                shard_files.append(os.path.join(root, f))
    if not shard_files:
        return
    print(f" [MT5Pipeline] Consolidating {len(shard_files)} shards into {final_parquet}...")
    writer = pq.ParquetWriter(final_parquet, schema=PARQUET_SCHEMA, compression="zstd", compression_level=3)
    for sf in sorted(shard_files):
        tbl = pq.read_table(sf)
        writer.write_table(tbl, row_group_size=5000)
    writer.close()
    import shutil
    shutil.rmtree(year_dir, ignore_errors=True)
    print(f" [MT5Pipeline] Consolidated -> {final_parquet}")


def build_windows_and_save_by_year(df, seq_len, out_dir, pair, tf,
                                   max_samples=50000, stride=None):
    available = [c for c in FEATURES if c in df.columns]
    feat_arr = np.asarray(df[available], dtype=np.float32)
    dir_arr = np.asarray(df['direction'], dtype=np.int64)
    tp_arr = np.asarray(df['tp_pips'], dtype=np.float32)
    sl_arr = np.asarray(df['sl_pips'], dtype=np.float32)

    if 'time' in df.columns:
        df_time = pd.to_datetime(df['time'], utc=True)
        ts_arr = np.asarray(df_time.astype(np.int64) // 10**9, dtype=np.int64)
    else:
        raise ValueError("DataFrame missing explicit chronological 'time' axis.")

    n_total = len(df) - seq_len
    if n_total <= 0:
        return

    if stride is None:
        stride = max(1, int(np.ceil(n_total / max_samples))) if max_samples > 0 else 1

    indices = np.arange(0, n_total, stride)
    target_indices = indices + seq_len
    target_times = df_time.iloc[target_indices]
    sample_years = target_times.dt.year.values
    unique_years = np.unique(sample_years)

    for year in unique_years:
        year_mask = (sample_years == year)
        year_indices = indices[year_mask]
        year_targets = target_indices[year_mask]

        num_samples = len(year_indices)
        if num_samples == 0:
            continue

        chunk_size = CHUNK_SIZE
        num_chunks = int(np.ceil(num_samples / chunk_size))
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, num_samples)
            chunk_indices = year_indices[start:end]
            chunk_targets = year_targets[start:end]
            chunk_num = len(chunk_indices)

            chunk_suffix = c if num_chunks > 1 else None

            if year_chunk_complete(out_dir, str(year), pair, tf, chunk_suffix):
                print(f" [RESUME] Skipping existing chunk {year}-{c} for {pair}@{tf}")
                continue

            X_chunk = np.empty((chunk_num, seq_len, len(available)), dtype=np.float32)
            for out_i, idx in enumerate(chunk_indices):
                X_chunk[out_i] = feat_arr[idx : idx + seq_len]

            y_dir_chunk = dir_arr[chunk_targets]
            y_mag_chunk = np.stack([tp_arr[chunk_targets], sl_arr[chunk_targets]], axis=1)
            timestamps_chunk = ts_arr[chunk_targets]

            save_shards(X_chunk, y_dir_chunk, y_mag_chunk, out_dir, pair, tf,
                        year_split=str(year), timestamps=timestamps_chunk,
                        chunk_suffix=chunk_suffix)

            del X_chunk, y_dir_chunk, y_mag_chunk, timestamps_chunk


# ─── Load Forex Datasets from YAML ────────────────────────────────────────

def load_forex_datasets(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    datasets = []
    for name, spec in cfg.get("datasets", {}).items():
        if spec.get("dataset_type") == "forex":
            ds = {
                "name": name,
                "pairs": spec.get("pairs", []),
                "timeframes": spec.get("timeframe_rungs", []),
                "start_date": spec.get("start_date", "2019-01-01"),
                "lookback": spec.get("lookback_bars", 168),
            }
            if ds["pairs"] and ds["timeframes"]:
                datasets.append(ds)
    return datasets


# ─── Full Pipeline with Resumption and Symbol Resolution ─────────────────

def run_download_pipeline(
    dataset_defs: List[Dict],
    out_dir: str,
    login=None,
    password=None,
    server=None,
    api_key=None,
):
    mt5_active = connect_mt5(login=login, password=password, server=server, api_key=api_key)
    if not mt5_active:
        raise RuntimeError("[FATAL] Cannot connect to MetaTrader 5.")

    tasks = {}
    for ds in dataset_defs:
        for pair in ds["pairs"]:
            for tf in ds["timeframes"]:
                start = ds["start_date"]
                key = (pair, tf, start)
                if key not in tasks:
                    tasks[key] = []
                tasks[key].append(ds["name"])

    current_year = datetime.now().year

    try:
        for (pair, tf, start_date), dataset_names in tasks.items():
            missing_datasets = []
            for ds_name in set(dataset_names):
                all_years_present = True
                for year in range(int(start_date.split('-')[0]), current_year + 1):
                    year_parquet = os.path.join(out_dir, ds_name, f"ForexUniverse{year}.parquet")
                    if os.path.isfile(year_parquet):
                        continue
                    shard_dir = os.path.join(out_dir, ds_name, f"ForexUniverse{year}", pair, str(tf))
                    if not os.path.isdir(shard_dir):
                        all_years_present = False
                        break
                    any_file = any(
                        (f.endswith('.npy') or f.endswith('.parquet'))
                        for f in os.listdir(shard_dir)
                        if os.path.isfile(os.path.join(shard_dir, f))
                    )
                    if not any_file:
                        all_years_present = False
                        break
                if not all_years_present:
                    missing_datasets.append(ds_name)

            if not missing_datasets:
                print(f" [RESUME] {pair} @ {tf}min already complete for all datasets. Skipping download.")
                continue

            print(f"\n [MT5Pipeline] Processing {pair} @ {tf}min (lookback={TIMEFRAME_LOOKBACK.get(tf, 168)}) for datasets: {missing_datasets}")

            if tf == 1:
                current_max_samples = 2_000_000
            elif tf == 5:
                current_max_samples = 500_000
            else:
                current_max_samples = 50_000
            print(f"   [MAX-SAMPLES] Set to {current_max_samples:,} for {tf}min timeframe.")

            df = download_bars(pair, tf, start_date=start_date, retries=3)
            if df is None:
                print(f" [SKIP] {pair} @ {tf}min - symbol not available. Skipping all datasets.")
                continue

            df = compute_indicators(df)
            df = generate_labels(df, pair)
            df = normalize_ohlcv(df)

            for ds_name in missing_datasets:
                dataset_out_dir = os.path.join(out_dir, ds_name)
                print(f"   Saving for dataset: {ds_name} -> {dataset_out_dir}")
                build_windows_and_save_by_year(
                    df,
                    seq_len=TIMEFRAME_LOOKBACK.get(tf, 168),
                    out_dir=dataset_out_dir,
                    pair=pair,
                    tf=tf,
                    max_samples=current_max_samples
                )

    finally:
        disconnect_mt5()

    print("\n [MT5Pipeline] Download pipeline complete.")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LemGendary MT5 Data Pipeline (Final - Resilient)")
    parser.add_argument("--config", default="unified_data.yaml", help="Path to the unified_data.yaml manifest")
    parser.add_argument("--out_dir", default="data/forex", help="Output directory for .npy shards")
    parser.add_argument("--login", type=int, default=None, help="MT5 demo account login ID")
    parser.add_argument("--password", type=str, default=None, help="MT5 demo account password")
    parser.add_argument("--server", type=str, default=None, help="MT5 demo account server name")
    parser.add_argument("--api_key", type=str, default=None, help="MT5 demo account API key")
    parser.add_argument("--pairs", nargs="+", default=None, help="Override: list of pairs to process")
    parser.add_argument("--timeframes", nargs="+", type=int, default=None, help="Override: timeframes in minutes")
    parser.add_argument("--start_date", type=str, default="2019-01-01", help="Override: start date")

    args = parser.parse_args()

    if args.pairs is not None:
        print(" [MT5Pipeline] Override mode: using CLI pairs/timeframes/start_date")
        tf_list = args.timeframes if args.timeframes is not None else [1, 5, 15, 60, 240, 1440]
        dataset_defs = [{
            "name": "CustomOverride",
            "pairs": args.pairs,
            "timeframes": tf_list,
            "start_date": args.start_date,
        }]
    else:
        print(f" [MT5Pipeline] Loading forex datasets from: {args.config}")
        dataset_defs = load_forex_datasets(args.config)
        if not dataset_defs:
            print(" [MT5Pipeline] No forex datasets found in YAML. Exiting.")
            sys.exit(0)

    print(f" [MT5Pipeline] Processing {len(dataset_defs)} dataset(s):")
    for ds in dataset_defs:
        print(f"   - {ds['name']}: {len(ds['pairs'])} pairs, {len(ds['timeframes'])} timeframes, start={ds['start_date']}")

    run_download_pipeline(
        dataset_defs=dataset_defs,
        out_dir=args.out_dir,
        login=args.login,
        password=args.password,
        server=args.server,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
