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

Facade module: the implementation lives in the focused modules below, and
every public name stays importable from ``forex.pipeline``.

- ``pipeline_config``: feature list, label settings, spreads, symbol maps.
- ``mt5_session``: MT5 connect/disconnect and symbol resolution.
- ``mt5_download``: bar download with range and chunked fallbacks.
- ``indicators``: technical indicator features.
- ``labels``: direction/TP/SL labels and OHLCV normalisation.
- ``sharding``: windowing and resumable Parquet shards.
- ``pipeline_runner``: YAML dataset loading, orchestration and CLI.
"""

from .indicators import compute_indicators
from .labels import generate_labels, normalize_ohlcv
from .mt5_download import download_bars
from .mt5_session import MT5Credentials, connect_mt5, disconnect_mt5, resolve_symbol
from .pipeline_config import (
    ALTERNATIVE_KEYWORDS,
    CHUNK_SIZE,
    DIRECTION_THRESHOLD_PIPS,
    FEATURES,
    LABEL_HORIZON_BARS,
    MT5_SYMBOL_MAP,
    MT5_TIMEFRAMES,
    PAIR_SPREADS_PIPS,
)
from .pipeline_runner import load_forex_datasets, main, run_download_pipeline
from .schema import TIMEFRAME_LOOKBACK
from .sharding import (
    ShardLabels,
    ShardTarget,
    build_windows_and_save_by_year,
    consolidate_year_parquet,
    save_shards,
    year_chunk_complete,
)

__all__ = [
    "ALTERNATIVE_KEYWORDS",
    "CHUNK_SIZE",
    "DIRECTION_THRESHOLD_PIPS",
    "FEATURES",
    "LABEL_HORIZON_BARS",
    "MT5_SYMBOL_MAP",
    "MT5_TIMEFRAMES",
    "PAIR_SPREADS_PIPS",
    "TIMEFRAME_LOOKBACK",
    "MT5Credentials",
    "ShardLabels",
    "ShardTarget",
    "build_windows_and_save_by_year",
    "compute_indicators",
    "connect_mt5",
    "consolidate_year_parquet",
    "disconnect_mt5",
    "download_bars",
    "generate_labels",
    "load_forex_datasets",
    "main",
    "normalize_ohlcv",
    "resolve_symbol",
    "run_download_pipeline",
    "save_shards",
    "year_chunk_complete",
]


if __name__ == "__main__":
    main()
