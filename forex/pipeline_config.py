"""
Constants shared by the MT5 download pipeline modules.

Feature column order, label settings, per-pair spreads, MT5 timeframe names,
symbol mapping and the shard chunk size live here so every stage of the
pipeline reads the same values. ``TIMEFRAME_LOOKBACK`` lives in ``forex.schema``.
"""

FEATURES = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal", "atr", "bb_width",
    "session_sin", "session_cos", "atr_percentile", "bar_range_ratio",
]

LABEL_HORIZON_BARS = 20
DIRECTION_THRESHOLD_PIPS = 5

PAIR_SPREADS_PIPS = {
    "EURUSD": 1.2,  "GBPUSD": 1.5,  "USDJPY": 1.3,  "XAUUSD": 35.0,
    "USDCAD": 1.8,  "USDCHF": 1.5,  "AUDUSD": 1.4,  "NZDUSD": 2.0,
    "EURGBP": 1.5,  "EURJPY": 1.8,  "GBPJPY": 2.5,  "USOIL":  40.0,
    "US500":  10.0, "NAS100": 20.0,
    "DE40":   10.0, "XAGUSD": 20.0,
}

MT5_TIMEFRAMES = {
    1: "TIMEFRAME_M1", 5: "TIMEFRAME_M5", 15: "TIMEFRAME_M15",
    60: "TIMEFRAME_H1", 240: "TIMEFRAME_H4", 1440: "TIMEFRAME_D1",
}

# ─── Symbol Mapping (logical -> preferred MT5 name) ─────────────────────
MT5_SYMBOL_MAP = {
    "USOIL": "WTI",      # oil is WTI in this broker
    "NAS100": "NAS100",  # optional, explicit
}

# ─── Alternative keywords for fuzzy resolution ──────────────────────────
ALTERNATIVE_KEYWORDS = {
    "USOIL": ["WTI", "OIL"],
    "US500": ["SPX", "SP500", "SPX500"],
    "NAS100": ["US100", "NDX"],
    "DE40": ["DAX", "GER40"],
    "XAGUSD": ["SILVER", "XAG"],
    "XAUUSD": ["GOLD", "XAU"],
}

CHUNK_SIZE = 20000
