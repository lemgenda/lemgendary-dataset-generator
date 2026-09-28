"""
LemGendary Dataset Compiler — Unified Progress Bar Architecture SSOT.

Enforces strictly uniform progress bar visual formatting, terminal line-wrap prevention,
and granular item-by-item updates across all compilation, sharding, and ingestion utilities.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Iterator, TypeVar
from tqdm import tqdm

T = TypeVar("T")

# Unified single-line bar format across the entire LemGendary ecosystem:
STANDARD_BAR_FORMAT = "{desc:<26} |{bar:28}| {percentage:5.1f}% [{n_fmt}/{total_fmt}, {rate_fmt}]"


def supports_unicode() -> bool:
    """Detect whether standard output safely supports unicode block characters."""
    try:
        encoding = getattr(sys.stdout, "encoding", "") or ""
        if "utf" in encoding.lower():
            return True
        # Check PYTHONIOENCODING or modern Windows Terminal
        if os.environ.get("WT_SESSION") or os.environ.get("PYTHONIOENCODING", "").lower().startswith("utf"):
            return True
        return False
    except Exception:
        return False


def create_progress_bar(
    total: int | None = None,
    desc: str = "PROCESSING",
    unit: str = "img",
    ncols: int | None = 90,
    ascii_blocks: bool | None = None,
    mininterval: float = 0.25,
    **kwargs: Any,
) -> tqdm:
    """Create a standardized tqdm progress bar guaranteed to render identically across all tools."""
    kwargs.setdefault("bar_format", STANDARD_BAR_FORMAT)
    kwargs.setdefault("mininterval", mininterval)
    kwargs.setdefault("file", sys.stdout)

    if ascii_blocks is None:
        kwargs.setdefault("ascii", not supports_unicode())
    else:
        kwargs.setdefault("ascii", ascii_blocks)

    if ncols is not None:
        kwargs["ncols"] = ncols
        kwargs["dynamic_ncols"] = False
    else:
        kwargs.setdefault("dynamic_ncols", True)

    return tqdm(total=total, desc=desc, unit=unit, **kwargs)


def track_progress(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "PROCESSING",
    unit: str = "img",
    ncols: int | None = 90,
    **kwargs: Any,
) -> Iterator[T]:
    """Iterate over an iterable with standardized real-time single-item progress bar updates."""
    count = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
    pbar = create_progress_bar(
        total=count,
        desc=desc,
        unit=unit,
        ncols=ncols,
        **kwargs,
    )
    with pbar:
        for item in iterable:
            yield item
            pbar.update(1)

