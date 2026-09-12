"""
LemGendary Forex Manifold Schema & Column Descriptors Specification
===================================================================
Standardized PyArrow schema, data types, and column descriptions
shared across Forex conversion and metadata injection pipelines.
"""

import json
import pyarrow as pa

TIMEFRAME_LOOKBACK = {
    1: 512,
    5: 288,
    15: 192,
    60: 168,
    240: 90,
    1440: 252,
}

EXTENDED_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD",
    "USDCAD", "USDCHF", "AUDUSD", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP",
    "XAGUSD", "USOIL",
    "US500", "NAS100", "DE40"
]

COLUMN_DESCRIPTIONS = {
    "pair": "Asset / currency pair / commodity symbol identifier (e.g., EURUSD, GBPUSD, USDJPY, XAUUSD, NAS100, DE40, USOIL, US500).",
    "timeframe": "Bar aggregation timeframe rung in minutes: 1=M1 (1min), 5=M5 (5min), 15=M15 (15min), 60=H1 (60min), 240=H4 (240min), 1440=D1 (1440min).",
    "timestamp": "Millisecond Unix epoch timestamp of the sequence prediction anchor / candle close.",
    "y_dir": "Causal directional classification target label over forward horizon: 0=SELL (Down), 1=HOLD (Sideways/Neutral), 2=BUY (Up).",
    "tp_pips": "Optimal forward Take-Profit target excursion magnitude in pips.",
    "sl_pips": "Maximum adverse excursion Stop-Loss safety threshold in pips.",
    "seq_len": "Historical lookback sequence length in bars (e.g., 168 for H1 macro, 512 for M1 microstructure).",
    "n_features": "Number of input feature dimensions per timestep (14 channels: OHLCV, RSI, MACD, MACD Signal, ATR, Bollinger Band Width, Session Sin/Cos, ATR Percentile, Bar Range Ratio).",
    "features": "Serialized float32 binary tensor representing the normalized [seq_len, n_features] temporal feature matrix."
}

PARQUET_SCHEMA = pa.schema([
    pa.field("pair", pa.string(), metadata={"description": COLUMN_DESCRIPTIONS["pair"]}),
    pa.field("timeframe", pa.int32(), metadata={"description": COLUMN_DESCRIPTIONS["timeframe"]}),
    pa.field("timestamp", pa.int64(), metadata={"description": COLUMN_DESCRIPTIONS["timestamp"]}),
    pa.field("y_dir", pa.int8(), metadata={"description": COLUMN_DESCRIPTIONS["y_dir"]}),
    pa.field("tp_pips", pa.float32(), metadata={"description": COLUMN_DESCRIPTIONS["tp_pips"]}),
    pa.field("sl_pips", pa.float32(), metadata={"description": COLUMN_DESCRIPTIONS["sl_pips"]}),
    pa.field("seq_len", pa.int16(), metadata={"description": COLUMN_DESCRIPTIONS["seq_len"]}),
    pa.field("n_features", pa.int16(), metadata={"description": COLUMN_DESCRIPTIONS["n_features"]}),
    pa.field("features", pa.binary(), metadata={"description": COLUMN_DESCRIPTIONS["features"]}),
], metadata={
    b"description": b"LemGendary Forex Universe High-Fidelity OHLCV Temporal Manifold",
    b"columns": json.dumps(COLUMN_DESCRIPTIONS).encode("utf-8"),
    b"domain": b"Financial & Time-Series",
    b"task": b"forex_prediction",
    b"timeframe_rungs": b"[1, 5, 15, 60, 240, 1440]",
    b"features_list": b'["open", "high", "low", "close", "volume", "rsi", "macd", "macd_signal", "atr", "bb_width", "session_sin", "session_cos", "atr_percentile", "bar_range_ratio"]',
    b"author": b"LemGendary AI",
    b"created_by": b"LemGendary MT5 Compiler Pipeline"
})
