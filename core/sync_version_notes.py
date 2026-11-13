"""Version notes for a new Kaggle dataset version.

Kaggle shows the notes next to each dataset version. They name the compiler
release that built the manifold and the sample counts its index.json (else
dataset_info.yaml) records, so every published version is labelled. A push
may add a short description of what changed (``sync push --notes``); it
follows the generated part.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from core.cli_args import __version__
from utils.shard_audit import ExpectedCounts, expected_counts

logger = logging.getLogger(__name__)

_SPLIT_ORDER = ("train", "val", "test")


def _count_text(counts: ExpectedCounts) -> str | None:
    """``"22,000 samples (train 19,790, val 2,210)"``; None when nothing is recorded."""
    splits = counts.index_splits
    if splits:
        ordered = [s for s in _SPLIT_ORDER if s in splits]
        ordered += sorted(s for s in splits if s not in _SPLIT_ORDER)
        detail = ", ".join(f"{s} {splits[s]:,}" for s in ordered)
        return f"{sum(splits.values()):,} samples ({detail})"
    if counts.info_count is not None:
        return f"{counts.info_count:,} samples"
    return None


def _generated_notes(manifold_dir: Path) -> str:
    """Compiler release plus the recorded sample counts (release only when unreadable)."""
    notes = f"LemGendary Dataset Compiler v{__version__}"
    try:
        count_text = _count_text(expected_counts(manifold_dir))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.warning("Sample counts of %s unreadable for the version notes: %s",
                       manifold_dir, exc)
        return notes
    return f"{notes}; {count_text}" if count_text else notes


def build_version_notes(manifold_dir: Path, extra: str | None = None) -> str:
    """Version notes for uploading ``manifold_dir``: compiler release and sample counts.

    ``extra`` (what changed in this version) follows them after ``". "``;
    blank text adds nothing.
    """
    notes = _generated_notes(manifold_dir)
    extra_text = " ".join((extra or "").split())
    return f"{notes}. {extra_text}" if extra_text else notes
