"""Forex manifold scanner for LemGendary Dataset Compiler.

Scans unified Parquet archives and legacy multi-rung NPY directory hierarchies
to gather shard inventory, pair distributions, and temporal sample counts.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.error_types import PARQUET_READ_ERRORS
from core.lazy_imports import load_parquet

logger = logging.getLogger(__name__)


@dataclass
class _ForexScan:
    """Accumulates shard details and per-year/pair/timeframe sample counts."""

    result: dict[str, Any] = field(
        default_factory=lambda: {
            "years": [],
            "pairs": [],
            "timeframes": [],
            "samples_per_year": {},
            "samples_per_pair": {},
            "samples_per_tf": {},
            "details": [],
        }
    )
    years: set[int] = field(default_factory=set)
    pairs: set[str] = field(default_factory=set)
    timeframes: set[int] = field(default_factory=set)

    def add(self, year: int, pair: str, tf: int, count: int) -> None:
        """Record ``count`` samples for one (year, pair, timeframe) shard."""
        self.result["details"].append(
            {"year": year, "pair": pair, "timeframe": tf, "count": count}
        )
        for key, value in (
            ("samples_per_year", year),
            ("samples_per_pair", pair),
            ("samples_per_tf", tf),
        ):
            self.result[key][value] = self.result[key].get(value, 0) + count

    def finish(self) -> dict[str, Any]:
        """Return the scan result with sorted year, pair and timeframe lists."""
        self.result["years"] = sorted(self.years)
        self.result["pairs"] = sorted(self.pairs)
        self.result["timeframes"] = sorted(self.timeframes)
        return self.result


def _year_suffix(path: Path, stem: str) -> int | None:
    """Return the year encoded after 'ForexUniverse' in ``stem``, or None."""
    try:
        return int(stem.replace("ForexUniverse", ""))
    except ValueError:
        logger.debug("Ignoring %s: no year suffix", path.name)
        return None


def _scan_parquet_file(scan: _ForexScan, pq_file: Path, year: int) -> None:
    """Count samples per (pair, timeframe) in one unified yearly Parquet file."""
    try:
        pq = load_parquet()
        pf = pq.ParquetFile(str(pq_file))
        tbl = pf.read(columns=["pair", "timeframe"])
        p_arr = tbl["pair"].to_numpy(zero_copy_only=False)
        tf_arr = tbl["timeframe"].to_numpy()

        counts = Counter(zip(p_arr, tf_arr))
        for (pair, tf), count in counts.items():
            scan.pairs.add(str(pair))
            scan.timeframes.add(int(tf))
            scan.add(year, str(pair), int(tf), int(count))
    except (*PARQUET_READ_ERRORS, KeyError) as e:
        logger.warning("Error scanning %s: %s", pq_file.name, e)


def _count_npy_samples(tf_dir: Path) -> int:
    """Sum the leading dimension of every X*.npy array in a timeframe folder."""
    total_samples = 0
    for npy_file in tf_dir.glob("X*.npy"):
        try:
            arr = np.load(npy_file, mmap_mode="r")
            total_samples += arr.shape[0]
        except (OSError, ValueError) as exc:
            logger.warning("Could not load npy file %s: %s", npy_file, exc)
    return total_samples


def _scan_pair_dir(scan: _ForexScan, pair_dir: Path, year: int) -> None:
    """Scan the numeric timeframe folders of one legacy pair directory."""
    pair = pair_dir.name
    scan.pairs.add(pair)
    for tf_dir in pair_dir.iterdir():
        if not tf_dir.is_dir():
            continue
        try:
            tf = int(tf_dir.name)
        except ValueError:
            logger.debug("Ignoring %s: timeframe folder is not numeric", tf_dir)
            continue
        scan.timeframes.add(tf)

        total_samples = _count_npy_samples(tf_dir)
        if total_samples > 0:
            scan.add(year, pair, tf, total_samples)


def scan_forex_manifold(root_path: Path) -> dict[str, Any]:
    """Scan root_path for Forex Parquet shards or legacy directory chunks."""
    scan = _ForexScan()

    # 1. Scan unified Parquet files first (ForexUniverseYYYY.parquet)
    for pq_file in sorted(root_path.glob("ForexUniverse*.parquet")):
        year = _year_suffix(pq_file, pq_file.stem)
        if year is None:
            continue
        scan.years.add(year)
        _scan_parquet_file(scan, pq_file, year)

    # 2. Scan legacy directories if not already scanned as Parquet
    for chunk_dir in root_path.glob("ForexUniverse*"):
        if not chunk_dir.is_dir():
            continue
        year = _year_suffix(chunk_dir, chunk_dir.name)
        if year is None or year in scan.years:
            continue
        scan.years.add(year)

        for pair_dir in chunk_dir.iterdir():
            if pair_dir.is_dir():
                _scan_pair_dir(scan, pair_dir, year)

    return scan.finish()
