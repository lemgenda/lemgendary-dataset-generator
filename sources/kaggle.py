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
    push_kaggle_dataset_metadata,
    ensure_kaggle_dataset_exists,
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
            yd_path = Path(__file__).parent.parent / "unified_data.yaml"
            if not yd_path.exists():
                yd_path = Path(__file__).parent / "unified_data.yaml"
            if yd_path.exists():
                with open(yd_path, "r", encoding="utf-8") as yf:
                    ydata = yaml.safe_load(yf) or {}
                prefix = ydata.get("_registry_metadata", {}).get("name_prefix", "LemGendized")
                suffix = ydata.get("_registry_metadata", {}).get("name_suffix", "")
                for _, entry in ydata.get("datasets", {}).items():
                    ref = entry.get("kaggle_ref", "")
                    if slug.lower() in ref.lower():
                        mod_folder = entry.get("modernized_folder")
                        manifold_name = mod_folder or f"{prefix}{entry.get('name', '')}{suffix}"
                        break
        except Exception as exc:
            print(f"[DEBUG] Kaggle manifold name lookup fallback: {exc}")
        target_dir = root_datasets_dir / manifold_name
    return target_dir, root_datasets_dir, manifold_name, slug


def main():
    """Main CLI entrypoint for Kaggle dataset operations."""
    import socket
    socket.setdefaulttimeout(60.0)
    parser = argparse.ArgumentParser(description="LemGendary Kaggle Manager")
    parser.add_argument("--repo_id", required=True, help="Kaggle dataset handle or URL")
    parser.add_argument("--output_dir", default="", help="Target local directory")
    parser.add_argument("--is_competition", action="store_true", help="Download from competition")
    parser.add_argument("--action", default="download", choices=["download", "upload", "status", "metadata"], help="Action to perform")
    parser.add_argument("--all-datasets", action="store_true", help="Update metadata for all datasets in unified_data.yaml (metadata action only)")
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

    elif args.action == "metadata":
        import yaml
        if args.all_datasets:
            # Update metadata for all datasets defined in unified_data.yaml
            yd_path = Path(__file__).parent.parent / "unified_data.yaml"
            if not yd_path.exists():
                print("[ERROR] unified_data.yaml not found.")
                sys.exit(1)
            with open(yd_path, "r", encoding="utf-8") as yf:
                ydata = yaml.safe_load(yf) or {}
            prefix = ydata.get("_registry_metadata", {}).get("name_prefix", "LemGendized")
            suffix = ydata.get("_registry_metadata", {}).get("name_suffix", "")
            all_ok = True
            for ds_key, ds_entry in ydata.get("datasets", {}).items():
                ref = ds_entry.get("kaggle_ref", "").replace("kaggle://", "").strip()
                if not ref:
                    continue
                mod_folder = ds_entry.get("modernized_folder") or f"{prefix}{ds_entry.get('name', '')}{suffix}"
                # Try resolving metadata file from LemGendaryDatasets or output_dir arg
                base_dir = Path(args.output_dir).resolve() if args.output_dir else Path("../LemGendaryDatasets").resolve()
                meta_candidate = base_dir / mod_folder / "dataset-metadata.json"
                if not meta_candidate.exists():
                    print(f"[SKIP] No dataset-metadata.json found for {ds_key} at {meta_candidate}")
                    continue
                print(f"[META] Updating Kaggle metadata for {ref} ({ds_key})...")
                ok = push_kaggle_dataset_metadata(ref, meta_candidate)
                if ok:
                    print(f"[OK] Metadata updated for {ref}")
                else:
                    print(f"[FAILED] Metadata update failed for {ref}")
                    all_ok = False
            if not all_ok:
                sys.exit(1)
        else:
            # Update metadata for single dataset
            if not args.output_dir:
                parser.error("--output_dir (path to manifold directory with dataset-metadata.json) is required for metadata action.")
            meta_path = Path(args.output_dir).resolve()
            print(f"[META] Updating Kaggle metadata for {clean_repo_id}...")
            success = push_kaggle_dataset_metadata(clean_repo_id, meta_path)
            if not success:
                sys.exit(1)
            print(f"[OK] Metadata successfully updated for {clean_repo_id}")


if __name__ == "__main__":
    main()
