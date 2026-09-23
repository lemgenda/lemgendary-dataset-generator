# 2026 Phase 1.2: Ensure project root is on sys.path so sibling imports
# (common_sync, archive_manager, etc.) resolve when this file is invoked
# directly via `python sources/<name>.py` — which is how the hub PS1 calls it.
# Also prune the script directory (sources/) from sys.path to prevent sources/kaggle.py
# from shadowing the installed PyPI 'kaggle' package.
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir]
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
"""
LemGendary Kaggle Dataset Manager CLI
=====================================
Unified interface for uploading, downloading, extracting, and monitoring Kaggle manifolds.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.getLogger("kagglehub").setLevel(logging.ERROR)

from core.common_sync import (
    CHUNK_SIZE,
    DatasetVersionInfo,
    setup_kaggle_auth,
    robust_staging_cleanup,
    cleanup_temp_archives,
    get_dataset_version_info,
    get_dataset_status,
    track_kaggle_dataset_status,
    copy_with_progress,
    copy_tree_with_progress,
    perform_dataset_upload,
    perform_dataset_download,
)


def _resolve_manifold_paths(clean_repo_id: str, output_dir: str) -> tuple[Path, Path, str, str]:
    """Resolve target manifold directory and root datasets directory."""
    target_dir = Path(output_dir).resolve()
    slug = clean_repo_id.split("/")[-1]
    if target_dir.name.startswith("LemGendized"):
        root_datasets_dir = target_dir.parent
        manifold_name = target_dir.name
    else:
        root_datasets_dir = target_dir
        manifold_name = slug
        try:
            import yaml
            yd_path = Path(__file__).parent / "unified_data.yaml"
            if yd_path.exists():
                with open(yd_path, "r", encoding="utf-8") as yf:
                    ydata = yaml.safe_load(yf) or {}
                prefix = ydata.get("_registry_metadata", {}).get("name_prefix", "LemGendized")
                suffix = ydata.get("_registry_metadata", {}).get("name_suffix", "Large")
                for _, entry in ydata.get("datasets", {}).items():
                    ref = entry.get("kaggle_ref", "")
                    if slug.lower() in ref.lower():
                        manifold_name = f"{prefix}{entry.get('name', '')}{suffix}"
                        break
        except Exception as exc:
            print(f"[DEBUG] Kaggle manifold name lookup fallback: {exc}")
        target_dir = root_datasets_dir / manifold_name
    return target_dir, root_datasets_dir, manifold_name, slug


def main():
    """Main CLI entrypoint for Kaggle dataset operations."""
    parser = argparse.ArgumentParser(description="LemGendary Kaggle Manager")
    parser.add_argument("--repo_id", required=True, help="Kaggle dataset handle or URL")
    parser.add_argument("--output_dir", default="", help="Target local directory")
    parser.add_argument("--is_competition", action="store_true", help="Download from competition")
    parser.add_argument("--action", default="download", choices=["download", "upload", "status"], help="Action to perform")
    parser.add_argument("--no-wait", action="store_true", help="Skip monitoring server-side extraction after upload")
    args = parser.parse_args()

    if args.action in ["download", "upload"] and not args.output_dir:
        parser.error(f"--output_dir is required when --action is '{args.action}'.")

    clean_repo_id = args.repo_id.replace("kaggle://", "")
    owner = clean_repo_id.split("/")[0] if "/" in clean_repo_id else "lemtreursi"
    setup_kaggle_auth(default_user=owner)

    if args.action == "download":
        print(f"PULLING: {clean_repo_id}")
        target_dir, root_datasets_dir, manifold_name, slug = _resolve_manifold_paths(clean_repo_id, args.output_dir)
        success = perform_dataset_download(
            clean_repo_id=clean_repo_id,
            target_dir=target_dir,
            root_datasets_dir=root_datasets_dir,
            manifold_name=manifold_name,
            slug=slug,
            is_competition=args.is_competition
        )
        if not success:
            sys.exit(1)

    elif args.action == "upload":
        print(f"PUSHING: {clean_repo_id}")
        src_path = Path(args.output_dir).resolve()
        success = perform_dataset_upload(
            src_path=src_path,
            clean_repo_id=clean_repo_id,
            no_wait=args.no_wait
        )
        if not success:
            sys.exit(1)

    elif args.action == "status":
        success = track_kaggle_dataset_status(clean_repo_id)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
