"""Forex manifold reduction for :mod:`tools.manifold_reduce`.

Split out of ``tools/manifold_reduce.py``: keeps a year range and a subset of
timeframes of a compiled forex manifold (all pairs), then rewrites its
``dataset_info.yaml`` and copies the companion files.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.manifold_reduce_select import (
    ABORTED_MESSAGE,
    DEFAULT_SUFFIX,
    ReduceRequest,
    ReductionAborted,
    prompt_multiselect,
    sample_count,
)

__all__ = ["reduce_forex_dataset"]

logger = logging.getLogger(__name__)

_COMPANION_GLOBS = ("*.txt", "*.md", "*.ipynb")


@dataclass(frozen=True)
class _ForexInventory:
    """Pairs, timeframes and years present in a compiled forex folder."""

    pairs: list[str]
    timeframes: list[int]
    years: list[int]


@dataclass(frozen=True)
class _ForexPlan:
    """What a forex reduction keeps and where it writes."""

    forex_src: Path
    target_root: Path
    target_name: str
    kept_years: list[int]
    kept_tfs: set[int]
    kept_pairs: set[str]

    def describe(self) -> str:
        """One-line summary of the kept years, timeframes and pairs."""
        return (
            f"  Years: {self.kept_years[0]} – {self.kept_years[-1]}  |  Timeframes:"
            f" {sorted(self.kept_tfs)}  |  Pairs: all ({len(self.kept_pairs)})"
        )


def _numeric_dirs(parent: Path):
    """Yield ``(dir, int(name))`` for sub-directories with numeric names (others logged)."""
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        try:
            yield child, int(child.name)
        except ValueError:
            logger.debug("Skipping non-numeric timeframe directory %s", child)


def _year_dirs(tf_dir: Path):
    """Yield ``(dir, year)`` for the digit-named year folders under a timeframe."""
    for year_dir in sorted(tf_dir.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            yield year_dir, int(year_dir.name)


def _scan_forex(forex_src: Path) -> _ForexInventory:
    """Discover available pairs, timeframes and years from the forex directory structure."""
    if not forex_src.exists():
        return _ForexInventory([], [], [])
    pair_dirs = [d for d in sorted(forex_src.iterdir()) if d.is_dir()]
    tfs: set[int] = set()
    years: set[int] = set()
    for pair_dir in pair_dirs:
        for tf_dir, tf in _numeric_dirs(pair_dir):
            tfs.add(tf)
            # Now look for year subdirectories under the timeframe
            years.update(year for _, year in _year_dirs(tf_dir))
    return _ForexInventory([d.name for d in pair_dirs], sorted(tfs), sorted(years))


def _ask_year(label: str, default: int) -> int:
    """Prompt for a year; empty input keeps ``default`` (ValueError on bad input)."""
    raw = input(f"Enter {label} year [Default: {default}]: ").strip()
    return int(raw) if raw else default


def _choose_year_range(
    avail_years: list[int], request: ReduceRequest
) -> tuple[int, int]:
    """Return ``(start, end)``; non-interactive keeps the most recent ``fraction`` of years."""
    # Non-interactive default keeps every year (or the most recent share of them).
    default_years = avail_years[-sample_count(len(avail_years), request.fraction):]
    default_start, default_end = default_years[0], default_years[-1]
    if not request.interactive:
        return default_start, default_end
    # Ask user for year range to keep.
    print(f"\nAvailable years: {avail_years}")
    try:
        return _ask_year("start", default_start), _ask_year("end", default_end)
    except ValueError:
        return default_start, default_end
    except KeyboardInterrupt as exc:
        raise ReductionAborted from exc


def _choose_timeframes(avail_tfs: list[int], interactive: bool) -> set[int]:
    """Return the timeframes to keep (all of them when non-interactive)."""
    tf_labels = [str(tf) for tf in avail_tfs]
    if not interactive:
        return {int(x) for x in tf_labels}
    try:
        kept_tf_strs = prompt_multiselect("Select timeframes to include:", tf_labels)
    except KeyboardInterrupt as exc:
        raise ReductionAborted from exc
    return {int(x) for x in kept_tf_strs}


def _choose_suffix(request: ReduceRequest) -> str:
    """Return the output suffix (override, prompt, or the default)."""
    if request.suffix_override is not None:
        return str(request.suffix_override)
    if not request.interactive:
        return DEFAULT_SUFFIX
    try:
        raw_suffix = input("\nNew suffix [Default: Reduced]: ").strip()
    except KeyboardInterrupt as exc:
        raise ReductionAborted from exc
    return raw_suffix if raw_suffix else DEFAULT_SUFFIX


def _plan_forex(
    source_root: Path, base_name: str, request: ReduceRequest, out_parent: Path
) -> _ForexPlan | None:
    """Resolve what to keep (prompting when interactive); None after a printed error."""
    forex_src = source_root / "forex"
    inventory = _scan_forex(forex_src)
    if not inventory.pairs or not inventory.timeframes or not inventory.years:
        print(
            f"[ERROR] No compiled forex data found under {forex_src} (missing pairs, timeframes, or"
            " year folders)."
        )
        return None

    start_year, end_year = _choose_year_range(inventory.years, request)
    if start_year > end_year:
        print("[ERROR] Start year must be <= end year.")
        return None

    # Filter years that actually exist within the range.
    kept_years = [y for y in inventory.years if start_year <= y <= end_year]
    if not kept_years:
        print("[ERROR] No years in the specified range exist in the source.")
        return None

    kept_tfs = _choose_timeframes(inventory.timeframes, request.interactive)
    target_name = f"{base_name}{_choose_suffix(request)}"
    return _ForexPlan(
        forex_src=forex_src,
        target_root=out_parent / target_name,
        target_name=target_name,
        kept_years=kept_years,
        kept_tfs=kept_tfs,
        # All pairs are kept (unchanged)
        kept_pairs=set(inventory.pairs),
    )


def _copy_forex(plan: _ForexPlan) -> set[str]:
    """Copy the kept pair/timeframe/year folders; return the pairs actually copied."""
    forex_dst = plan.target_root / "forex"
    actual_copied_pairs: set[str] = set()
    if not plan.forex_src.exists():
        return actual_copied_pairs
    for pair_dir in sorted(plan.forex_src.iterdir()):
        if not pair_dir.is_dir() or pair_dir.name not in plan.kept_pairs:
            continue
        actual_copied_pairs.add(pair_dir.name)
        for tf_dir, tf in _numeric_dirs(pair_dir):
            if tf not in plan.kept_tfs:
                continue
            # Copy only the years we want.
            for year_dir, year in _year_dirs(tf_dir):
                if year not in plan.kept_years:
                    continue
                dst_year = forex_dst / pair_dir.name / tf_dir.name / year_dir.name
                dst_year.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(year_dir, dst_year, dirs_exist_ok=True)
    return actual_copied_pairs


def _write_outputs(
    plan: _ForexPlan, source_root: Path, info: dict[str, Any], copied_pairs: set[str]
) -> None:
    """Write the updated dataset_info.yaml and copy the companion flat files."""
    # Write updated dataset_info.yaml with the new year range.
    new_info = info.copy()
    new_info["pairs"] = sorted(copied_pairs)
    new_info["timeframe_rungs"] = sorted(plan.kept_tfs)
    # Adjust start_date to the first kept year (January 1st)
    new_info["start_date"] = f"{plan.kept_years[0]}-01-01"

    with open(plan.target_root / "dataset_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(new_info, f, sort_keys=False)

    # Copy companion flat files from source.
    for pattern in _COMPANION_GLOBS:
        for src_file in source_root.glob(pattern):
            shutil.copy2(src_file, plan.target_root / src_file.name)


def reduce_forex_dataset(
    source_root: Path,
    base_name: str,
    info: dict[str, Any],
    request: ReduceRequest,
    out_parent: Path,
) -> int:
    """Create a reduced forex manifold under ``out_parent``; return an exit code."""
    print(f"\n--- Forex Reduction: {source_root.name} ---")
    try:
        plan = _plan_forex(source_root, base_name, request, out_parent)
    except ReductionAborted:
        print(ABORTED_MESSAGE)
        return 1
    if plan is None:
        return 1

    if request.dry_run:
        print(
            f"\n[DRY-RUN] Would reduce forex {source_root.name} -> {plan.target_name}."
            " Nothing written."
        )
        print(plan.describe())
        return 0

    print(f"\n[FOREX REDUCING] {source_root.name} -> {plan.target_name}")
    print(plan.describe())

    copied_pairs = _copy_forex(plan)
    _write_outputs(plan, source_root, info, copied_pairs)
    print(f"\n[SUCCESS] Reduced Forex manifold created at {plan.target_root.name}")
    return 0
