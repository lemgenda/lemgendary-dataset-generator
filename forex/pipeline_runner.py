"""
End-to-end MT5 download pipeline: dataset definitions, orchestration and CLI.

For every unique (pair, timeframe, start date) across the requested datasets
it downloads bars once, adds indicators, labels and normalisation, and writes
year shards for each dataset that is still missing them.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

import yaml

from .indicators import compute_indicators
from .labels import generate_labels, normalize_ohlcv
from .mt5_download import download_bars
from .mt5_session import MT5Credentials, connect_mt5, disconnect_mt5
from .schema import TIMEFRAME_LOOKBACK
from .sharding import ShardTarget, build_windows_and_save_by_year

__all__ = ["load_forex_datasets", "main", "run_download_pipeline"]

_DEFAULT_TIMEFRAMES = [1, 5, 15, 60, 240, 1440]
# Sample cap per timeframe (minutes); every other timeframe uses _DEFAULT_MAX_SAMPLES.
_MAX_SAMPLES_BY_TF = {1: 2_000_000, 5: 500_000}
_DEFAULT_MAX_SAMPLES = 50_000


# ─── Load Forex Datasets from YAML ────────────────────────────────────────


def load_forex_datasets(yaml_path):
    """Load the Forex dataset definitions from a unified_data.yaml file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")

    with open(yaml_path, 'r', encoding="utf-8") as f:
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


def _group_tasks(dataset_defs):
    """Map each (pair, timeframe, start date) to the dataset names that need it."""
    tasks: dict[tuple, list[str]] = {}
    for ds in dataset_defs:
        for pair in ds["pairs"]:
            for tf in ds["timeframes"]:
                tasks.setdefault((pair, tf, ds["start_date"]), []).append(ds["name"])
    return tasks


def _year_present(dataset_dir, year, pair, tf):
    """True when the year Parquet exists or the pair/timeframe shard folder holds a shard."""
    if os.path.isfile(os.path.join(dataset_dir, f"ForexUniverse{year}.parquet")):
        return True
    shard_dir = os.path.join(dataset_dir, f"ForexUniverse{year}", pair, str(tf))
    if not os.path.isdir(shard_dir):
        return False
    return any(
        f.endswith(('.npy', '.parquet'))
        for f in os.listdir(shard_dir)
        if os.path.isfile(os.path.join(shard_dir, f))
    )


def _missing_datasets(out_dir, task, dataset_names):
    """Dataset names that lack at least one year of shards for ``task``."""
    pair, tf, start_date = task
    years = range(int(start_date.split('-')[0]), datetime.now().year + 1)
    return [
        ds_name
        for ds_name in set(dataset_names)
        if not all(_year_present(os.path.join(out_dir, ds_name), y, pair, tf) for y in years)
    ]


def _process_task(task, missing_datasets, out_dir):
    """Download, label and shard one (pair, timeframe, start date) for ``missing_datasets``."""
    pair, tf, start_date = task
    seq_len = TIMEFRAME_LOOKBACK.get(tf, 168)
    print(
        f"\n [MT5Pipeline] Processing {pair} @ {tf}min "
        f"(lookback={seq_len}) for datasets: {missing_datasets}"
    )

    max_samples = _MAX_SAMPLES_BY_TF.get(tf, _DEFAULT_MAX_SAMPLES)
    print(f"   [MAX-SAMPLES] Set to {max_samples:,} for {tf}min timeframe.")

    df = download_bars(pair, tf, start_date=start_date, retries=3)
    if df is None:
        print(f" [SKIP] {pair} @ {tf}min - symbol not available. Skipping all datasets.")
        return

    df = normalize_ohlcv(generate_labels(compute_indicators(df), pair))

    for ds_name in missing_datasets:
        dataset_out_dir = os.path.join(out_dir, ds_name)
        print(f"   Saving for dataset: {ds_name} -> {dataset_out_dir}")
        build_windows_and_save_by_year(
            df,
            ShardTarget(out_dir=dataset_out_dir, pair=pair, timeframe_min=tf),
            seq_len=seq_len,
            max_samples=max_samples,
        )


def run_download_pipeline(
    dataset_defs: list[dict],
    out_dir: str,
    *,
    credentials: MT5Credentials | None = None,
    **credential_fields: Any,
):
    """Download, label and shard every pair and timeframe in ``dataset_defs`` into ``out_dir``.

    Login details come from ``credentials`` or from the ``login``, ``password``,
    ``server`` and ``api_key`` keywords (an unknown keyword raises TypeError).
    """
    creds = credentials if credentials is not None else MT5Credentials(**credential_fields)
    mt5_active = connect_mt5(
        login=creds.login, password=creds.password, server=creds.server, api_key=creds.api_key
    )
    if not mt5_active:
        raise RuntimeError("[FATAL] Cannot connect to MetaTrader 5.")

    try:
        for task, dataset_names in _group_tasks(dataset_defs).items():
            missing = _missing_datasets(out_dir, task, dataset_names)
            if not missing:
                print(
                    f" [RESUME] {task[0]} @ {task[1]}min already complete for all datasets."
                    " Skipping download."
                )
                continue
            _process_task(task, missing, out_dir)
    finally:
        disconnect_mt5()

    print("\n [MT5Pipeline] Download pipeline complete.")


# ─── CLI ────────────────────────────────────────────────────────────────────


def _build_parser():
    parser = argparse.ArgumentParser(description="LemGendary MT5 Data Pipeline (Final - Resilient)")
    parser.add_argument(
        "--config", default="unified_data.yaml", help="Path to the unified_data.yaml manifest"
    )
    parser.add_argument("--out_dir", default="data/forex", help="Output directory for .npy shards")
    parser.add_argument("--login", type=int, default=None, help="MT5 demo account login ID")
    parser.add_argument("--password", type=str, default=None, help="MT5 demo account password")
    parser.add_argument("--server", type=str, default=None, help="MT5 demo account server name")
    parser.add_argument("--api_key", type=str, default=None, help="MT5 demo account API key")
    parser.add_argument(
        "--pairs", nargs="+", default=None, help="Override: list of pairs to process"
    )
    parser.add_argument(
        "--timeframes", nargs="+", type=int, default=None, help="Override: timeframes in minutes"
    )
    parser.add_argument("--start_date", type=str, default="2019-01-01", help="Override: start date")
    return parser


def _dataset_defs_from_args(args):
    if args.pairs is not None:
        print(" [MT5Pipeline] Override mode: using CLI pairs/timeframes/start_date")
        tf_list = args.timeframes if args.timeframes is not None else list(_DEFAULT_TIMEFRAMES)
        return [
            {
                "name": "CustomOverride",
                "pairs": args.pairs,
                "timeframes": tf_list,
                "start_date": args.start_date,
            }
        ]
    print(f" [MT5Pipeline] Loading forex datasets from: {args.config}")
    dataset_defs = load_forex_datasets(args.config)
    if not dataset_defs:
        print(" [MT5Pipeline] No forex datasets found in YAML. Exiting.")
        sys.exit(0)
    return dataset_defs


def main():
    """Run the MT5 download pipeline (CLI entry point)."""
    args = _build_parser().parse_args()
    dataset_defs = _dataset_defs_from_args(args)

    print(f" [MT5Pipeline] Processing {len(dataset_defs)} dataset(s):")
    for ds in dataset_defs:
        print(
            f"   - {ds['name']}: {len(ds['pairs'])} pairs, {len(ds['timeframes'])} timeframes,"
            f" start={ds['start_date']}"
        )

    run_download_pipeline(
        dataset_defs=dataset_defs,
        out_dir=args.out_dir,
        credentials=MT5Credentials(
            login=args.login, password=args.password, server=args.server, api_key=args.api_key
        ),
    )
