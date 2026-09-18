"""Forex manifold scanner for LemGendary Dataset Compiler.

Scans unified Parquet archives and legacy multi-rung NPY directory hierarchies
to gather shard inventory, pair distributions, and temporal sample counts.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np


def scan_forex_manifold(root_path: Path) -> dict[str, Any]:
    """Scan root_path for Forex Parquet shards or legacy directory chunks."""
    result: dict[str, Any] = {
        "years": [],
        "pairs": [],
        "timeframes": [],
        "samples_per_year": {},
        "samples_per_pair": {},
        "samples_per_tf": {},
        "details": [],
    }

    year_set: set[int] = set()
    pair_set: set[str] = set()
    tf_set: set[int] = set()

    # 1. Scan unified Parquet files first (ForexUniverseYYYY.parquet)
    for pq_file in sorted(root_path.glob("ForexUniverse*.parquet")):
        year_str = pq_file.stem.replace("ForexUniverse", "")
        try:
            year = int(year_str)
        except ValueError:
            continue
        year_set.add(year)

        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(str(pq_file))
            tbl = pf.read(columns=["pair", "timeframe"])
            p_arr = tbl["pair"].to_numpy(zero_copy_only=False)
            tf_arr = tbl["timeframe"].to_numpy()

            counts = Counter(zip(p_arr, tf_arr))
            for (pair, tf), count in counts.items():
                pair_set.add(str(pair))
                tf_set.add(int(tf))
                entry = {
                    "year": year,
                    "pair": str(pair),
                    "timeframe": int(tf),
                    "count": int(count),
                }
                result["details"].append(entry)
                result["samples_per_year"][year] = result["samples_per_year"].get(year, 0) + int(count)
                result["samples_per_pair"][str(pair)] = result["samples_per_pair"].get(str(pair), 0) + int(count)
                result["samples_per_tf"][int(tf)] = result["samples_per_tf"].get(int(tf), 0) + int(count)
        except Exception as e:
            print(f"[WARNING] Error scanning {pq_file.name}: {e}")

    # 2. Scan legacy directories if not already scanned as Parquet
    for chunk_dir in root_path.glob("ForexUniverse*"):
        if not chunk_dir.is_dir():
            continue
        year_str = chunk_dir.name.replace("ForexUniverse", "")
        try:
            year = int(year_str)
        except ValueError:
            continue

        if year in year_set:
            continue
        year_set.add(year)

        for pair_dir in chunk_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            pair = pair_dir.name
            pair_set.add(pair)

            for tf_dir in pair_dir.iterdir():
                if not tf_dir.is_dir():
                    continue
                tf_str = tf_dir.name
                try:
                    tf = int(tf_str)
                except ValueError:
                    continue
                tf_set.add(tf)

                total_samples = 0
                for npy_file in tf_dir.glob("X*.npy"):
                    try:
                        arr = np.load(npy_file, mmap_mode="r")
                        total_samples += arr.shape[0]
                    except Exception as exc:
                        print(f"[DEBUG] Could not load npy file {npy_file}: {exc}")

                if total_samples > 0:
                    entry = {
                        "year": year,
                        "pair": pair,
                        "timeframe": tf,
                        "count": total_samples,
                    }
                    result["details"].append(entry)
                    result["samples_per_year"][year] = result["samples_per_year"].get(year, 0) + total_samples
                    result["samples_per_pair"][pair] = result["samples_per_pair"].get(pair, 0) + total_samples
                    result["samples_per_tf"][tf] = result["samples_per_tf"].get(tf, 0) + total_samples

    result["years"] = sorted(year_set)
    result["pairs"] = sorted(pair_set)
    result["timeframes"] = sorted(tf_set)
    return result
