"""Create reduced (sampled) variants of compiled manifolds.

Run as ``python -m tools.manifold_reduce``.

This module is the entry point and orchestration layer; the work lives in
:mod:`tools.manifold_reduce_select` (discovery and prompts),
:mod:`tools.manifold_reduce_vision` (image manifolds) and
:mod:`tools.manifold_reduce_forex` (forex manifolds).
"""

import argparse
import json
import logging
import sys
from typing import TypedDict, Unpack

import yaml

from core.compiler_core import (
    CONFIG,
    OUT_PARENT,
    remove_empty_dirs,
)
from core.doc_generator import generate_dataset_docs
from tools.manifold_reduce_forex import reduce_forex_dataset
from tools.manifold_reduce_select import (
    DEFAULT_MAX_GB,
    DEFAULT_SUFFIX,
    ReduceRequest,
    find_manifold,
    has_manifold_data,
    prompt_reduction_params,
    select_manifolds,
)
from tools.manifold_reduce_vision import copy_reduced_vision

__all__ = [
    "DEFAULT_MAX_GB",
    "DEFAULT_SUFFIX",
    "ReduceDatasetOptions",
    "ReduceRequest",
    "main",
    "purge_ghost_manifolds",
    "reduce_dataset",
]

logger = logging.getLogger(__name__)


class ReduceDatasetOptions(TypedDict, total=False):
    """Keyword options of :func:`reduce_dataset` (all optional)."""

    max_gb_override: float | None
    suffix_override: str | None
    dataset: str | None
    fraction: float | None
    assume_yes: bool
    dry_run: bool


def _request_error(fraction, interactive, dataset):
    """Return the error message for an invalid request, or None when it is valid."""
    if fraction is not None and not 0 < fraction <= 1:
        return f"[ERROR] --fraction must be in (0, 1], got {fraction}."
    if not interactive and not dataset:
        return "[ERROR] Non-interactive reduction requires --dataset."
    return None


def _resolve_targets(manifolds, dataset):
    """Return the manifolds to reduce (named, or picked interactively); [] after an error."""
    if not manifolds:
        print("[ERROR] No valid manifolds found to reduce.")
        return []
    if not dataset:
        return _choose_manifolds_interactively(manifolds)
    match = find_manifold(manifolds, dataset)
    if match is None:
        print(f"[ERROR] Manifold '{dataset}' not found or has no loose data to reduce.")
        return []
    return [match]


def reduce_dataset(**options: Unpack[ReduceDatasetOptions]):
    """Create reduced manifold variant(s).

    Interactive by default. ``assume_yes`` runs without any prompt (API/jobs) and
    then requires ``dataset``. ``fraction`` in (0, 1] keeps that share of samples
    (vision) or of the most recent years (forex). ``dry_run`` writes nothing.
    ``max_gb_override`` / ``suffix_override`` skip the size and suffix prompts.
    Returns a process exit code (0 on success).
    """
    request = ReduceRequest(
        max_gb_override=options.get("max_gb_override"),
        suffix_override=options.get("suffix_override"),
        fraction=options.get("fraction"),
        interactive=not options.get("assume_yes", False),
        dry_run=options.get("dry_run", False),
    )
    dataset = options.get("dataset")
    error = _request_error(request.fraction, request.interactive, dataset)
    if error:
        print(error)
        return 1

    print("\n[SCANNING] Locating existing manifolds in LemGendaryDatasets...")
    # ─── Phase 0: eligibility filter relaxed to support both legacy and modern names ───
    manifolds = [
        d for d in sorted(OUT_PARENT.iterdir())
        if d.is_dir()
        and d.name.startswith("LemGendized")
        and has_manifold_data(d)
    ]
    targets = _resolve_targets(manifolds, dataset)
    if not targets:
        return 1

    for source_root in targets:
        code = _reduce_one(source_root, request)
        if code != 0:
            return code
    return 0


def _choose_manifolds_interactively(manifolds):
    for i, m in enumerate(manifolds):
        # Status display: mark if a KaggleReady / Reduced sibling already exists
        has_reduced = any(
            (m.parent / f"{m.name.rsplit('Large', 1)[0]}{s}").exists()
            for s in ("KaggleReady", "Reduced")
        )
        if has_reduced:
            print(f"\033[92m{i + 1}. {m.name} (reduced variant exists)\033[0m")
        else:
            print(f"\033[93m{i + 1}. {m.name}\033[0m")

    target_indices = select_manifolds(manifolds)
    if not target_indices:
        return []
    return [manifolds[idx] for idx in target_indices]


def _reduce_one(source_root, request):
    """Reduce a single manifold. Returns an exit code."""
    old_suffix = str(CONFIG.get("name_suffix", "Large"))
    # ─── Phase 0: guard against empty suffix (post-modernization state) ───
    if old_suffix and source_root.name.endswith(old_suffix):
        base_name = source_root.name[:-len(old_suffix)]
    else:
        base_name = source_root.name

    dataset_type = "quality"
    info = {}
    info_path = source_root / "dataset_info.yaml"
    if info_path.exists():
        with open(info_path, 'r', encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}
            dataset_type = info.get("dataset_type", "quality")

    if dataset_type == "forex":
        return reduce_forex_dataset(source_root, base_name, info, request, OUT_PARENT)
    return _reduce_vision(source_root, base_name, request)


def _reduce_vision(source_root, base_name, request):
    """Vision / quality manifold flow. Returns an exit code."""
    max_gb, suffix = prompt_reduction_params(
        max_gb_override=request.max_gb_override,
        suffix_override=request.suffix_override,
        interactive=request.interactive,
    )
    if max_gb is None:
        return 1

    target_name = f"{base_name}{suffix}"
    target_root = OUT_PARENT / target_name
    share = f", {request.fraction:.0%} sampling" if request.fraction is not None else ""

    if request.dry_run:
        print(
            f"\n[DRY-RUN] Would reduce {source_root.name} -> {target_name} ({max_gb} GB"
            f" cap{share}). Nothing written."
        )
        return 0

    print(f"\n[REDUCING] {source_root.name} -> {target_name} ({max_gb} GB cap{share})...")
    new_index = copy_reduced_vision(source_root, target_root, max_gb, request.fraction)
    if new_index is None:
        return 1

    with open(target_root / "index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)

    remove_empty_dirs(target_root)
    generate_dataset_docs(target_root, new_index, target_name)
    print(
        "[INFO] Run `lemtrain notebooks generate --all` to regenerate notebooks for reduced"
        " datasets."
    )
    print(f"\n[SUCCESS] Reduced manifold created at {target_root.name}")
    return 0


def purge_ghost_manifolds():
    """
    Ghost Manifold Audit: identifies folders with no manifold data (no images, shards,
    index, or dataset_info). Currently reports only; purge is disabled to prevent
    accidental data loss.
    """
    print("\n[GHOST-AUDIT] Scanning for empty manifold folders in LemGendaryDatasets...")
    ghosts = []
    if not OUT_PARENT.exists():
        return
    for item in OUT_PARENT.iterdir():
        if not item.is_dir() or item.name == ".git":
            continue
        has_data = any(
            (item / d).exists() for d in ["images", "shards", "index.json", "dataset_info.yaml"]
        )
        if not has_data:
            ghosts.append(item)

    if ghosts:
        print(f"  [FOUND] {len(ghosts)} ghost folders identified.")
        print("  [ACTION] Ghost purge disabled. Keeping folders intact.")
    else:
        print("  [OK] No ghost manifolds detected.")


def main(argv=None):
    """Command-line entry point; returns an exit code."""

    parser = argparse.ArgumentParser(description="Manifold downsampling and scale reduction")
    parser.add_argument(
        "--reduce", action="store_true", help="Run dataset reduction flow (default action)"
    )
    parser.add_argument(
        "--dataset", default=None, help="Manifold folder name to reduce (required with --yes)"
    )
    parser.add_argument(
        "--fraction", type=float, default=None, help="Share of samples to keep, in (0, 1]"
    )
    parser.add_argument(
        "--max_gb", "--max-gb", type=float, default=None, help="Target max size in GB"
    )
    parser.add_argument("--suffix", default=None, help="Suffix for reduced manifold")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Non-interactive: never prompt, use defaults"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the plan only; write nothing"
    )
    args = parser.parse_args(argv)
    return reduce_dataset(
        max_gb_override=args.max_gb,
        suffix_override=args.suffix,
        dataset=args.dataset,
        fraction=args.fraction,
        assume_yes=args.yes,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    sys.exit(main())
