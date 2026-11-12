"""Forex manifold column specification: names, documented types and descriptions.

Single source of truth for the column descriptions shared by the Parquet
schema (:mod:`forex.schema`), the metadata injector and the Kaggle
metadata generator (:mod:`core.docs.metadata`). It imports nothing heavy,
so documentation code can use it without loading pyarrow.

Timestamps are whole seconds since the Unix epoch (UTC), the unit MT5
reports bar times in (owner decision, R2-D4).
"""

from __future__ import annotations

__all__ = ["COLUMN_DESCRIPTIONS", "COLUMN_DOC_TYPES", "TIMESTAMP_UNIT", "column_fields"]

TIMESTAMP_UNIT = "seconds"

COLUMN_DESCRIPTIONS: dict[str, str] = {
    "pair": (
        "Asset / currency pair / commodity symbol identifier (e.g., EURUSD, GBPUSD, USDJPY, XAUUSD,"
        " NAS100, DE40, USOIL, US500)."
    ),
    "timeframe": (
        "Bar aggregation timeframe rung in minutes: 1=M1 (1min), 5=M5 (5min), 15=M15 (15min), 60=H1"
        " (60min), 240=H4 (240min), 1440=D1 (1440min)."
    ),
    "timestamp": (
        "Unix epoch timestamp in seconds (UTC) of the sequence prediction anchor / candle close."
    ),
    "y_dir": (
        "Causal directional classification target label over forward horizon: 0=SELL (Down), 1=HOLD"
        " (Sideways/Neutral), 2=BUY (Up)."
    ),
    "tp_pips": "Optimal forward Take-Profit target excursion magnitude in pips.",
    "sl_pips": "Maximum adverse excursion Stop-Loss safety threshold in pips.",
    "seq_len": (
        "Historical lookback sequence length in bars (e.g., 168 for H1 macro, 512 for M1"
        " microstructure)."
    ),
    "n_features": (
        "Number of input feature dimensions per timestep (14 channels: OHLCV, RSI, MACD, MACD"
        " Signal, ATR, Bollinger Band Width, Session Sin/Cos, ATR Percentile, Bar Range Ratio)."
    ),
    "features": (
        "Serialized float32 binary tensor representing the normalized [seq_len, n_features]"
        " temporal feature matrix."
    ),
}

# Column types as documented in Kaggle / datapackage metadata.
COLUMN_DOC_TYPES: dict[str, str] = {
    "pair": "string",
    "timeframe": "integer",
    "timestamp": "integer",
    "y_dir": "integer",
    "tp_pips": "number",
    "sl_pips": "number",
    "seq_len": "integer",
    "n_features": "integer",
    "features": "bytes",
}


def column_fields() -> list[dict[str, str]]:
    """``[{"name", "type", "description"}, ...]`` in column order, for metadata files."""
    return [
        {"name": name, "type": COLUMN_DOC_TYPES[name], "description": description}
        for name, description in COLUMN_DESCRIPTIONS.items()
    ]
