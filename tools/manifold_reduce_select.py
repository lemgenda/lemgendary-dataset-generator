"""Manifold discovery and interactive prompts for :mod:`tools.manifold_reduce`.

Split out of ``tools/manifold_reduce.py``: finds reducible manifolds, parses
numbered selections, and asks for the size cap and output suffix.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from core.compiler_core import CONFIG

__all__ = [
    "ABORTED_MESSAGE",
    "DEFAULT_MAX_GB",
    "DEFAULT_SUFFIX",
    "ReduceRequest",
    "ReductionAborted",
    "find_manifold",
    "has_manifold_data",
    "prompt_multiselect",
    "prompt_reduction_params",
    "sample_count",
    "select_manifolds",
]

logger = logging.getLogger(__name__)

DEFAULT_MAX_GB = 190.0
DEFAULT_SUFFIX = "Reduced"
ABORTED_MESSAGE = "\n[ABORTED] Operation cancelled by user."


class ReductionAborted(Exception):
    """The user cancelled an interactive prompt (Ctrl+C)."""


@dataclass(frozen=True)
class ReduceRequest:
    """How one reduction run should behave (shared by every target manifold)."""

    max_gb_override: float | None = None
    suffix_override: str | None = None
    fraction: float | None = None
    interactive: bool = True
    dry_run: bool = False


# ─── Phase 0: Data-presence helper ──────────────────────────────────────────
def has_manifold_data(path: Path) -> bool:
    """Return True if the manifold has actual data (works for both legacy
    `*Large` names and modern suffix-free names)."""
    # Forex: any *.parquet directly in the folder
    try:
        if any(path.glob("*.parquet")):
            return True
    except OSError as exc:
        logger.debug("Error globbing parquet files in %s: %s", path, exc)

    # Image / target / mask manifolds: at least one file in a split folder
    for split in ("train", "val", "test"):
        for sub in ("images", "targets", "masks"):
            d = path / sub / split
            if d.exists() and _dir_has_file(d):
                return True
    return False


def _dir_has_file(d: Path) -> bool:
    """Return True when ``d`` directly contains at least one file."""
    try:
        return any(f.is_file() for f in d.iterdir())
    except OSError as exc:
        logger.debug("Error reading directory %s: %s", d, exc)
        return False


def _parse_index(token: str, count: int) -> int | None:
    """Return the 0-based index for a 1-based ``token``, or None when it is invalid."""
    try:
        idx = int(token.strip()) - 1
    except ValueError:
        return None
    return idx if 0 <= idx < count else None


def prompt_multiselect(label, options, default_all=True):
    """
    Present a numbered list of options and return the user-selected subset.
    Accepts comma-separated numbers or 'a' for all.
    Returns a list of selected option values.
    """
    print(f"\n{label}")
    for i, opt in enumerate(options):
        print(f"  {i + 1}. {opt}")
    prompt = "Select (comma-separated numbers, or 'a' for all)"
    if default_all:
        prompt += " [Default: all]: "
    else:
        prompt += ": "

    raw = input(prompt).strip().lower()
    if not raw or raw == 'a':
        return list(options)

    selected = []
    tokens = [token.strip() for token in raw.split(',')]
    for token in filter(None, tokens):
        idx = _parse_index(token, len(options))
        if idx is None:
            print(f"[WARNING] Ignoring invalid selection: '{token}'")
        else:
            selected.append(options[idx])
    return selected if selected else list(options)


def select_manifolds(manifolds):
    """Ask which listed manifolds to reduce; return their indices, or None."""
    try:
        sel = input(
            "\nSelect manifold to reduce (number, comma-separated, or 'a' for all): "
        ).strip()
        if not sel:
            return None
        if sel.lower() == 'a':
            return list(range(len(manifolds)))
        target_indices = []
        for part in sel.split(','):
            idx = _parse_index(part, len(manifolds))
            if idx is None:
                raise ValueError(part)
            target_indices.append(idx)
        return target_indices
    except ValueError:
        print("[ERROR] Invalid selection.")
        return None
    except KeyboardInterrupt:
        print(ABORTED_MESSAGE)
        return None


def prompt_reduction_params(max_gb_override=None, suffix_override=None, interactive=True):
    """Return ``(max_gb, suffix)``, prompting for unset values; ``(None, None)`` on error."""
    if not interactive:
        max_gb = float(max_gb_override) if max_gb_override is not None else DEFAULT_MAX_GB
        suffix = str(suffix_override) if suffix_override is not None else DEFAULT_SUFFIX
        return max_gb, suffix
    try:
        if max_gb_override is not None:
            max_gb = float(max_gb_override)
        else:
            raw_gb = input("Target max size in GB [Default: 190.0]: ").strip()
            max_gb = float(raw_gb) if raw_gb else 190.0
        if suffix_override is not None:
            suffix = str(suffix_override)
        else:
            raw_suffix = input("New suffix [Default: Reduced]: ").strip()
            suffix = raw_suffix if raw_suffix else "Reduced"
        return max_gb, suffix
    except ValueError:
        print("[ERROR] Invalid input.")
        return None, None
    except KeyboardInterrupt:
        print(ABORTED_MESSAGE)
        return None, None


def find_manifold(manifolds, dataset):
    """Return the manifold folder named ``dataset`` (case-insensitive; legacy suffix allowed)."""
    wanted = dataset.strip().lower()
    old_suffix = str(CONFIG.get("name_suffix", "Large")).lower()
    for m in manifolds:
        name = m.name.lower()
        if name == wanted or (old_suffix and name == f"{wanted}{old_suffix}"):
            return m
    return None


def sample_count(total, fraction):
    """Number of samples kept from ``total`` for a sampling ``fraction`` (None keeps all)."""
    if fraction is None or total == 0:
        return total
    return max(1, math.ceil(total * fraction))
