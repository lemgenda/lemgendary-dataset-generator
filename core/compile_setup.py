"""Run-wide setup for ``core.manifold_compile``: CLI parsing, pre-flight and the worker pool.

Everything here happens once per compile run, before the per-manifold loop:
parsing the CLI (with preset defaults), caching the models the run needs,
sizing the worker pool and indexing the DPED ground-truth frames.
"""

import argparse
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, cast

from core import presets
from core.cli_args import build_parser
from core.compiler_config import install_args
from core.compiler_core import (
    CONFIG,
    DATASETS_META,
    INPUT_ROOT,
    META,
    detect_task,
    get_device_info,
    init_worker,
)
from core.lazy_imports import load_model_module
from runtime.environment import load_torch

# torch is loaded after the runtime env vars are set (see runtime.environment).
torch = load_torch()

# Hard cap on auto-detected workers; --workers overrides it.
_AUTO_WORKER_CAP = 8


# ─── ARGUMENT PARSER ─────────────────────────────────────────────────────────
# 2026 Phase 1.5: Parser is the SSOT from cli_args.py. This module previously
# declared its own parser that had to stay manually in sync with
# compiler_core.py's. Both now share `build_parser()`.

def _apply_preset(parsed: argparse.Namespace, preset_cfg: Any) -> None:
    """Fill the options the user left unset from the preset."""
    if parsed.image_format is None:
        parsed.image_format = preset_cfg.image_format
    if parsed.image_quality is None:
        parsed.image_quality = preset_cfg.image_quality
    if parsed.target_quality is None and preset_cfg.target_quality is not None:
        parsed.target_quality = preset_cfg.target_quality
    if parsed.mask_format is None and preset_cfg.mask_format is not None:
        parsed.mask_format = preset_cfg.mask_format
    if not preset_cfg.vetting_enabled and not parsed.no_vetting:
        parsed.no_vetting = True
    if not preset_cfg.labeling_enabled and not parsed.no_labeling:
        parsed.no_labeling = True
    if preset_cfg.containers and parsed.also_format is None:
        parsed.also_format = ",".join(preset_cfg.containers)


def parse_compile_args(cmd_args: list[str] | None = None) -> argparse.Namespace:
    """Parse compile CLI arguments cleanly without top-level module side-effects."""
    parser = build_parser()
    parsed = parser.parse_args(cmd_args)

    if parsed.preset:
        try:
            _apply_preset(parsed, presets.get_preset(parsed.preset))
        except KeyError as exc:
            print(f"[ERROR] Invalid preset: {exc}")
            sys.exit(1)
    return parsed


# ────────────────────────────────────────────────────────────────────────────

def install_interrupt_handler() -> None:
    """2026 Resilience: Force-Kill Handler for Windows (SIGINT v1.1)."""
    if os.name == 'nt':
        def signal_handler(_sig, _frame):
            print("\n[INTERRUPT] Emergency termination requested. Mission aborted.")
            os._exit(1)
        signal.signal(signal.SIGINT, signal_handler)


@dataclass
class CompileSettings:
    """Run-wide limits and naming shared by every manifold of one compile."""

    min_gb: float
    max_gb: float
    prefix_str: str
    suffix_str: str
    max_workers: int = 1
    dped_canon_paths: set[str] = field(default_factory=set)


def _selected_models(args: argparse.Namespace):
    """Yield ``(model_key, model_config)`` for the models this run compiles."""
    for model_key, model_config in DATASETS_META.items():
        if args.model and model_key != args.model:
            continue
        yield model_key, model_config


def _preflight_models(args: argparse.Namespace) -> None:
    """Pre-load models globally once to prevent multiprocess race conditions on HF cache."""
    print("[PRE-FLIGHT] Analyzing task requirements...")
    diffusion = load_model_module("diffusion")
    encoder = load_model_module("encoder")

    # Analyze if any target models need AI augmentation
    needs_captioning = False
    needs_styling = False

    for model_key, _ in _selected_models(args):
        task = detect_task(model_key)
        if task == "diffusion":
            needs_captioning = True
        if task == "diffusion" or model_key == "nima_aesthetic":
            needs_styling = True

    if needs_captioning and not args.no_vetting:
        print("[PRE-FLIGHT] Caching CaptionSentry (BLIP)...")
        tmp = diffusion.CaptionSentry(device="cpu")
        del tmp

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if needs_styling and not args.no_vetting and needs_captioning:
        print("[PRE-FLIGHT] Caching CLIPManifold...")
        _ = encoder.CLIPManifold(device="cpu")

    print("[PRE-FLIGHT] Pre-flight analysis complete.")


def _worker_count(args: argparse.Namespace) -> int:
    """2026 Resilience: Adaptive worker scaling (v5.1)."""
    final_workers = args.workers if args.workers else CONFIG.get("num_workers", 4)

    if not args.workers and final_workers > _AUTO_WORKER_CAP:
        print(
            "[RESILIENCE] Capping auto-detected workers to 8 for stability. Use --workers to"
            " override."
        )
        final_workers = _AUTO_WORKER_CAP
    return final_workers


def _cache_canon_root(canon_root: Any, slug: str, paths: set[str]) -> None:
    """Add every file under one DPED ``canon`` folder to ``paths``."""
    print(f"[DPED] Caching ground truth manifold for {slug} ({canon_root.parent.name})...")
    for r, _, f_list in os.walk(canon_root):
        for f in f_list:
            paths.add(os.path.join(r, f).replace("\\", "/").lower())


def _dped_canon_paths(args: argparse.Namespace) -> set[str]:
    """2026 DPED Optimization: Pre-cache canon paths to avoid O(N) exists() calls."""
    dped_canon_paths: set[str] = set()
    for _, model_config in _selected_models(args):
        for ref_entry in cast("list[dict[str, Any]]", model_config.get("refs", [])):
            if "dped" not in ref_entry["ref"].lower():
                continue
            slug = ref_entry["ref"].split("/")[-1].lower()
            for canon_root in (
                INPUT_ROOT / slug / "iphone2canon" / "train" / "canon",
                INPUT_ROOT / slug / "iphone2canon" / "test" / "canon",
            ):
                if canon_root.exists():
                    _cache_canon_root(canon_root, slug, dped_canon_paths)
    return dped_canon_paths


def prepare_compile(args: argparse.Namespace) -> CompileSettings:
    """Install the run's CLI args, resolve its limits, pre-load models, size the pool.

    Installing ``args`` first is what makes ``get_args()`` work in the workers
    and applies the CLI overrides to CONFIG.
    """
    install_args(args)
    constraints = META.get("global_constraints", {})
    settings = CompileSettings(
        min_gb=constraints.get("min_size_gb", 0.1),
        max_gb=(
            args.max_gb
            if args.max_gb is not None
            else META.get("global_constraints", {}).get("max_size_gb", 50.0)
        ),
        prefix_str=META.get("name_prefix", ""),
        suffix_str=args.suffix if args.suffix is not None else META.get("name_suffix", ""),
    )

    _preflight_models(args)
    final_workers = _worker_count(args)
    settings.dped_canon_paths = _dped_canon_paths(args)

    settings.max_workers = max(1, final_workers)
    print(f"[PRE-FLIGHT] Python: {sys.executable}")
    print(
        f"[PRE-FLIGHT] Hardware: {get_device_info()} | Active Workers: {settings.max_workers}",
        flush=True,
    )

    if settings.max_workers > 4 and args.no_vetting and args.no_labeling:
        print("[I/O-GEAR] WARNING: High worker count detected for I/O-bound task.")
        print("   -> On mechanical HDDs, this will cause SEVERE thrashing (seeking contention).")
        print("   -> If performance is < 10it/s, restart with --workers 2 or 4.")
    return settings


def make_executor(args: argparse.Namespace, settings: CompileSettings,
                  existing_on_disk: set[str]) -> ThreadPoolExecutor:
    """Build the manifold's worker pool (workers initialise in-process for I/O-only runs)."""
    if args.no_vetting and args.no_labeling:
        init_worker(CONFIG, settings.dped_canon_paths, existing_on_disk, args)
        return ThreadPoolExecutor(max_workers=settings.max_workers)
    return ThreadPoolExecutor(
        max_workers=settings.max_workers,
        initializer=init_worker,
        initargs=(CONFIG, settings.dped_canon_paths, existing_on_disk, args),
    )
