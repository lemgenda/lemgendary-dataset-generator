"""
LemGendary Dataset Compiler — Shared CLI arguments and env-manager discovery.

Phase 1.5 of the 2026 modernization roadmap, extended in Phases 3, 4, and 5.

This module is the SSOT for:
  * The shared argparse parser consumed by `compiler_core.py` and
    `manifold_compile.py`.
  * Discovery of the `lem-env` executable (the LemGendary Environment
    Manager CLI).

Downstream convention:
    from cli_args import build_parser, resolve_lem_env, PROJECT_NAME
    parser = build_parser()
    args = parser.parse_args()
"""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import sys
from pathlib import Path


# ─── Constants ──────────────────────────────────────────────────────────────
PROJECT_NAME = "lemgendary-datasets"
__version__ = "16.7.0"


# ─── Shared argparse parser ─────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser shared by every compiler entry point.

    Flags are the union of what the hub PS1 can pass and what the CLI
    sub-commands forward to `manifold_compile.py`.
    """
    parser = argparse.ArgumentParser(
        prog="lemgendary",
        description="LemGendary Dataset Compiler Suite",
    )

    # ── Dataset identity ────────────────────────────────────────────────────
    parser.add_argument("--name", type=str, default=None,
                        help="Output folder name override (default: from registry metadata)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific dataset model key to compile")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Override the manifold name suffix (default: from registry metadata)")

    # ── Size and parallelism ────────────────────────────────────────────────
    parser.add_argument("--max_gb", type=float, default=None,
                        help="Override max_size_gb for this run")
    parser.add_argument("--workers", type=int,
                        default=max(1, multiprocessing.cpu_count() - 2),
                        help="Number of parallel workers (default: cpu_count - 2)")

    # ── Mode selectors ──────────────────────────────────────────────────────
    parser.add_argument("--reduce", action="store_true",
                        help="Start in Reduce mode")
    parser.add_argument("--cleanup", action="store_true",
                        help="Start in Cleanup mode")
    parser.add_argument("--finalize", action="store_true",
                        help="Sharding + README only — skip re-processing")

    # ── Quality gate toggles ────────────────────────────────────────────────
    parser.add_argument("--no-vetting", dest="no_vetting", action="store_true",
                        help="Disable NIMA quality gate (Pass-Through mode)")
    parser.add_argument("--no-labeling", dest="no_labeling", action="store_true",
                        help="Disable YOLO auto-labeling (High-Speed mode)")
    parser.add_argument("--no-hash", dest="no_hash", action="store_true",
                        help="Disable deduplication hash for maximum I/O speed")

    # ── Transcode controls (Phase 3) ────────────────────────────────────────
    parser.add_argument("--image-format", dest="image_format", type=str,
                        choices=["webp", "jpeg", "png", "keep"], default=None,
                        help="Image output format (default: webp)")
    parser.add_argument("--image-quality", dest="image_quality", type=int, default=None,
                        help="Quality for images 1-100 (default: 92)")
    parser.add_argument("--target-quality", dest="target_quality", type=int, default=None,
                        help="Quality for restoration targets (default: 95)")
    parser.add_argument("--mask-format", dest="mask_format", type=str,
                        choices=["webp-lossless", "png"], default=None,
                        help="Format for segmentation masks (default: webp-lossless)")

    # ── Container-format controls (Phase 4) ─────────────────────────────────
    parser.add_argument("--also-format", dest="also_format", type=str, default=None,
                        help="Comma-separated container formats to write in addition "
                             "to the canonical directory layout (e.g. 'mds' or 'mds,litdata')")
    parser.add_argument("--force-duplicate", dest="force_duplicate", action="store_true",
                        help="Proceed with container write despite WARN-tier hardlink fraction")
    parser.add_argument("--accept-space-loss", dest="accept_space_loss", action="store_true",
                        help="Proceed with container write despite BLOCK-tier hardlink fraction")

    # ── Smart-generation strategy overrides (Phase 5) ───────────────────────
    parser.add_argument("--label-strategy", dest="label_strategy", type=str, default=None,
                        choices=["blip_caption", "clip_zeroshot", "yolo_detection",
                                 "parsenet_segmentation", "nima_quality"],
                        help="Label generation strategy (default: inferred from task)")
    parser.add_argument("--prompt-strategy", dest="prompt_strategy", type=str, default=None,
                        help="Prompt template name (default: diffusers-v1)")
    parser.add_argument("--mask-strategy", dest="mask_strategy", type=str, default=None,
                        choices=["parsenet", "sam", "modnet"],
                        help="Mask generation strategy (default: parsenet for segmentation)")

    # ── Compiler Preset Profiles (Phase 8) ─────────────────────────────────
    parser.add_argument("--preset", dest="preset", type=str, default=None,
                        help="Compiler preset profile name from presets.yaml (e.g. quality-vision, restoration-hardlinked)")

    return parser




# ─── lem-env discovery ──────────────────────────────────────────────────────
_LEM_ENV_CACHE: str | None = None


def resolve_lem_env() -> str:
    """Locate the `lem-env` executable. Raises FileNotFoundError if absent.

    Resolution order:
      1. `shutil.which("lem-env")` — hits if the venv is activated or
         lem-env is installed on PATH.
      2. `../lemgendary-env-manager/.venv/Scripts/lem-env.exe` — Windows venv.
      3. `../lemgendary-env-manager/.venv/Scripts/lem-env` — rare shebang case.
      4. `../lemgendary-env-manager/.venv/bin/lem-env` — POSIX venv.
    """
    global _LEM_ENV_CACHE
    if _LEM_ENV_CACHE is not None:
        return _LEM_ENV_CACHE

    found = shutil.which("lem-env")
    if found:
        _LEM_ENV_CACHE = found
        return found

    here = Path(__file__).parent.resolve()
    parent = here.parent
    candidates = [
        parent / "lemgendary-env-manager" / ".venv" / "Scripts" / "lem-env.exe",
        parent / "lemgendary-env-manager" / ".venv" / "Scripts" / "lem-env",
        parent / "lemgendary-env-manager" / ".venv" / "bin" / "lem-env",
    ]
    for candidate in candidates:
        if candidate.exists():
            _LEM_ENV_CACHE = str(candidate)
            return _LEM_ENV_CACHE

    raise FileNotFoundError(
        "lem-env executable not found. Install the LemGendary Environment "
        "Manager by running:\n"
        "    cd ..\\lemgendary-env-manager\n"
        "    .\\.venv\\Scripts\\python.exe -m pip install -e .\n"
        "Or ensure ../lemgendary-env-manager/.venv/Scripts/lem-env.exe exists."
    )


def venv_python() -> str:
    """Return this project's venv Python path, or the current interpreter."""
    here = Path(__file__).parent.resolve()
    for candidate in (
        here / ".venv" / "Scripts" / "python.exe",
        here / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable