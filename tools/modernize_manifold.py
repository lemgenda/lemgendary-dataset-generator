"""
LemGendary Dataset Compiler — Modernize Manifold

Creates modernized manifold variants from legacy sets:
    LemGendizedNimaAestheticLarge -> LemGendizedNimaAesthetic (new folder)

Safety & Architecture:
    - The legacy manifold folder is PRESERVED untouched (never renamed).
    - A new modernized directory is created for the target manifold.
    - Directory layout, labels, annotations, notebooks, and metadata are replicated.
    - Images are transcoded in parallel to modern WebP (or configured format).
    - Modern dataset metadata (dataset_info.yaml, README.md, index.json) is regenerated.
    - Pluggable container formats (WebDataset / MDS / LitData / Parquet) are generated.
    - Remote Kaggle repositories are synchronized if not skipped.
    - name_suffix in unified_data.yaml is updated only when all eligible sets are modernized.

Usage:
    python -m tools.modernize_manifold                                   # interactive selection
    python -m tools.modernize_manifold --dry-run                         # preview plan
    python -m tools.modernize_manifold --all --yes                       # batch creation
    python -m tools.modernize_manifold --datasets yolov8n --also-format webdataset

Public facade and CLI entry point; the implementation lives in the
``tools.modernize`` package (manifold_scan, manifold_select, manifold_build,
manifold_publish, manifold_run).
"""

from __future__ import annotations

import argparse
import sys

from tools.modernize.manifold_build import _create_modernized_batch
from tools.modernize.manifold_publish import (
    _compute_new_kaggle_ref,
    _kaggle_reupload,
    _regen_docs,
    _report,
    _update_kaggle_ref,
    _write_name_suffix,
)
from tools.modernize.manifold_run import run
from tools.modernize.manifold_scan import (
    BRAND_PREFIX,
    LEGACY_SUFFIX,
    REGISTRY_YAML,
    ROOT,
    VENV_PYTHON,
    _enumerate_eligible,
    _find_dataset_entry,
    _find_kaggle_key,
    _has_manifold_data,
    _is_manifold_fully_modernized,
    _load_registry,
    _out_parent,
    _save_registry,
)
from tools.modernize.manifold_select import (
    _confirm,
    _match_query,
    _parse_selection,
    _print_eligible,
    _select_from_datasets_arg,
)

__all__ = [
    "BRAND_PREFIX",
    "LEGACY_SUFFIX",
    "REGISTRY_YAML",
    "ROOT",
    "VENV_PYTHON",
    "_compute_new_kaggle_ref",
    "_confirm",
    "_create_modernized_batch",
    "_enumerate_eligible",
    "_find_dataset_entry",
    "_find_kaggle_key",
    "_has_manifold_data",
    "_is_manifold_fully_modernized",
    "_kaggle_reupload",
    "_load_registry",
    "_match_query",
    "_out_parent",
    "_parse_selection",
    "_print_eligible",
    "_regen_docs",
    "_report",
    "_save_registry",
    "_select_from_datasets_arg",
    "_update_kaggle_ref",
    "_write_name_suffix",
    "build_parser",
    "main",
    "run",
]


# ─── Main ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for manifold modernization."""
    parser = argparse.ArgumentParser(
        description=(
            "LemGendary Manifold Modernization (create modern manifold variants, WebP transcoding,"
            " container formats)"
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan, do not modify anything")
    parser.add_argument("--all", action="store_true", help="Select all eligible manifolds")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation gate")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated list of folder names, base names, or dataset keys")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="Skip Kaggle re-upload (metadata still written)")
    parser.add_argument("--image-format", type=str, default="webp",
                        choices=["webp", "jpeg", "png", "keep"],
                        help="Transcode images during modernization (default: webp)")
    parser.add_argument("--image-quality", type=int, default=92,
                        help="Quality for image transcoding (1-100)")
    parser.add_argument("--target-quality", type=int, default=95,
                        help="Quality for target image transcoding (1-100)")
    parser.add_argument("--mask-format", type=str, default="webp-lossless",
                        choices=["webp-lossless", "png"],
                        help="Format for mask transcoding")
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        help=(
            "Override target format for all selected datasets (default: auto-detect from"
            " unified_data.yaml canonical_format)"
        ),
    )
    parser.add_argument(
        "--also-format",
        type=str,
        default=None,
        help="Alias for --format (e.g., webdataset, mds, litdata, parquet, directory)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help=(
            "Keep intermediate loose images after container conversion (default: false, purge for"
            " zero duplication)"
        ),
    )
    parser.add_argument("--skip-transcode", action="store_true",
                        help="Skip WebP image transcoding")
    parser.add_argument("--skip-container", action="store_true",
                        help="Skip modern container format writing")
    return parser


def main() -> int:
    """Parse the CLI arguments and run the modernization."""
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
