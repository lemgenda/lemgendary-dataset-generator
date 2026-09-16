"""
LemGendary Dataset Compiler — Shared source-fetcher helpers.

Small utility module consumed by sources/hf.py, sources/gh.py, sources/gd.py,
and sources/kaggle.py. Holds the cross-cutting concerns:

    - Auth token discovery (env vars, .kaggle_token, .huggingface_token)
    - Common argparse scaffolding for `--repo_id` / `--output_dir`
    - Uniform progress-bar construction
    - Structured error reporting for the hub PS1

Phase 1.2 of the 2026 modernization roadmap. No runtime side effects —
importing this module does not touch the network or the filesystem.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# ─── Auth token discovery ───────────────────────────────────────────────────
def read_token_file(filename: str) -> str | None:
    """Read a token file from the project root. Returns None if missing/empty."""
    p = PROJECT_ROOT / filename
    if not p.exists():
        return None
    content = p.read_text(encoding="utf-8").strip()
    return content or None


def load_kaggle_credentials(default_username: str = "lemtreursi") -> bool:
    """
    Populate KAGGLE_USERNAME / KAGGLE_KEY / KAGGLE_API_TOKEN from env or
    .kaggle_token. Returns True if any credential is available.
    """
    token = read_token_file(".kaggle_token")
    if token and "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = token
    if "KAGGLE_USERNAME" not in os.environ and default_username:
        os.environ["KAGGLE_USERNAME"] = default_username
    return bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or os.environ.get("KAGGLE_KEY")
        or os.environ.get("KAGGLE_USERNAME")
    )


def load_hf_credentials() -> bool:
    """Populate HF_TOKEN from env or .huggingface_token."""
    if "HF_TOKEN" in os.environ or "HUGGING_FACE_HUB_TOKEN" in os.environ:
        return True
    token = read_token_file(".huggingface_token")
    if token:
        os.environ["HF_TOKEN"] = token
        return True
    return False


# ─── Progress bars ──────────────────────────────────────────────────────────
def make_progress_bar(total: int, desc: str, colour: str = "cyan") -> tqdm:
    """Standard byte-metered tqdm factory used across all fetchers."""
    return tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        colour=colour,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.25,
    )


# ─── Structured status reporting (parsed by the hub PS1) ────────────────────
def emit_status(status: str) -> None:
    """Hub PS1 watches stdout for lines matching `STATUS:<value>`."""
    print(f"STATUS:{status}", flush=True)


def emit_result(result: str) -> None:
    """Hub PS1 watches stdout for lines matching `RESULT:<value>`.

    Valid values: DOWNLOADED, COMPLETED, FAILED, FOUND.
    """
    print(f"RESULT:{result}", flush=True)


def emit_notification(message: str) -> None:
    """Hub PS1 buffers these into a rolling event log."""
    print(f"NOTIFICATION:{message}", flush=True)


# ─── Common argparse helpers ────────────────────────────────────────────────
def add_common_args(parser) -> None:
    """Attach the args shared by every fetcher's CLI."""
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Repository identifier (backend-specific).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Local destination directory.")
    parser.add_argument("--token", type=str, default=None,
                        help="Optional auth token override.")


def summarize_output(output_dir: str | Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) for a completed fetch. Used to
    decide COMPLETED vs FAILED in the hub status reporter."""
    root = Path(output_dir)
    if not root.exists():
        return 0, 0
    count = 0
    total = 0
    for entry in root.rglob("*"):
        try:
            if entry.is_file():
                count += 1
                total += entry.stat().st_size
        except OSError:
            continue
    return count, total