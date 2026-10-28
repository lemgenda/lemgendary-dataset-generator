#!/usr/bin/env python3
"""
LemGendary Kaggle Metadata & Column Descriptors Synchronizer CLI
================================================================
Synchronizes dataset-metadata.json, licenses, keywords, and column-level
schema descriptors directly to Kaggle's backend via the Dataset Settings API.
Can be executed standalone without re-uploading dataset archives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.common_sync import is_kaggle_large_manifold, push_kaggle_dataset_metadata


def is_large_manifold(repo_id: str) -> bool:
    """Detect whether a repository handle points to a legacy *large manifold on Kaggle."""
    return is_kaggle_large_manifold(repo_id)


def sync_metadata(
    repo_id: str | None = None,
    metadata_dir: str | Path | None = None,
    all_datasets: bool = False,
    dry_run: bool = False,
) -> int:
    """Synchronize dataset metadata and column descriptors to Kaggle (returns 0 on success)."""
    datasets_root = Path(metadata_dir).resolve() if (all_datasets and metadata_dir) else (_ROOT.parent / "LemGendaryDatasets")

    if all_datasets:
        if not datasets_root.exists():
            print(f"[ERROR] Datasets directory not found at {datasets_root}")
            return 1

        meta_files = sorted(datasets_root.glob("*/dataset-metadata.json"))
        if not meta_files:
            print(f"[WARNING] No dataset-metadata.json files found in {datasets_root}")
            return 0

        existing_datasets: dict[str, str] = {}
        owner = "lemtreursi"
        if not dry_run:
            from core.common_sync import setup_kaggle_auth
            setup_kaggle_auth()
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            try:
                remote_ds_list = api.dataset_list(user=owner) or []
                existing_datasets = {
                    ds.ref.lower(): ds.ref
                    for ds in remote_ds_list
                    if ds is not None and getattr(ds, "ref", None) is not None
                    and not is_large_manifold(ds.ref)
                }
                print(f"[INFO] Discovered {len(existing_datasets)} active published datasets on Kaggle for @{owner}")
            except Exception as exc:
                print(f"[WARNING] Could not pre-fetch remote dataset list ({exc}); will attempt direct push for all.")
                existing_datasets = {}

        print(f"Discovered {len(meta_files)} manifolds with local metadata in {datasets_root}")
        successes = 0
        skipped = 0
        failures = 0
        synced_slugs: set[str] = set()

        for mf in meta_files:
            manifold_dir = mf.parent
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
                raw_id = meta.get("id") or f"lemtreursi/{manifold_dir.name.lower().replace('_', '')}"

                target_repo_id = None
                if existing_datasets:
                    if raw_id.lower() in existing_datasets:
                        target_repo_id = existing_datasets[raw_id.lower()]
                    if not target_repo_id:
                        from core.docs.context import find_unified_entry
                        u_entry = find_unified_entry(manifold_dir.name)
                        if u_entry and u_entry.get("kaggle_ref"):
                            clean_kref = u_entry["kaggle_ref"].replace("kaggle://", "").lower()
                            if clean_kref in existing_datasets:
                                target_repo_id = existing_datasets[clean_kref]
                            elif not is_large_manifold(clean_kref):
                                target_repo_id = clean_kref

                if not target_repo_id:
                    target_repo_id = raw_id

                if is_large_manifold(target_repo_id):
                    print(f"[SKIP] Refusing to touch legacy *large manifold: {target_repo_id} ({manifold_dir.name})")
                    skipped += 1
                    continue

                if target_repo_id in synced_slugs:
                    continue
                synced_slugs.add(target_repo_id)

                if dry_run:
                    print(f"\n[DRY-RUN] Would synchronize {manifold_dir.name} -> {target_repo_id}:")
                    print(f"  Title: {meta.get('title')}")
                    print(f"  Subtitle: {meta.get('subtitle')}")
                    print(f"  Author: {meta.get('author')}")
                    print(f"  Header Image: {meta.get('headerImage')}")
                    print(f"  License: {meta.get('licenses')}")
                    print(f"  Keywords: {meta.get('keywords')}")
                    print(f"  Provenance Sources: {meta.get('provenanceSources')}")
                    print(f"  Citations: {meta.get('citations')}")
                    print(f"  Resources: {len(meta.get('resources', []))} items registered")
                    successes += 1
                    continue

                print(f"\n[SYNCING] {manifold_dir.name} -> {target_repo_id}...")
                ok = push_kaggle_dataset_metadata(target_repo_id, manifold_dir)
                if ok:
                    successes += 1
                else:
                    failures += 1
            except Exception as exc:
                print(f"[ERROR] Failed to push metadata for {manifold_dir.name}: {exc}")
                failures += 1

        action_word = "Simulated" if dry_run else "Completed"
        print(f"\n{action_word} synchronization: {successes} succeeded, {skipped} skipped, {failures} failed.")
        return 1 if failures > 0 else 0

    meta_path: Path
    if metadata_dir:
        candidate = Path(metadata_dir)
        if candidate.exists():
            meta_path = candidate
        else:
            from tools.manifold_sync import resolve_manifold_folder
            folder_name = resolve_manifold_folder(str(metadata_dir))
            meta_path = datasets_root / folder_name
    else:
        meta_path = datasets_root / "LemGendizedForexUniverse"

    if not repo_id:
        from tools.manifold_sync import resolve_kaggle_ref
        resolved_ref = resolve_kaggle_ref(meta_path.name)
        if resolved_ref:
            repo_id = resolved_ref
        elif (meta_path / "dataset-metadata.json").exists():
            try:
                mdata = json.loads((meta_path / "dataset-metadata.json").read_text(encoding="utf-8"))
                repo_id = mdata.get("id")
            except (OSError, json.JSONDecodeError):
                repo_id = None
        if not repo_id:
            repo_id = f"lemtreursi/{meta_path.name.lower()}"

    clean_repo_id = repo_id.replace("kaggle://", "").strip()
    if is_large_manifold(clean_repo_id):
        print(f"[ERROR] Refusing to touch legacy *large manifold: {clean_repo_id}")
        return 1

    if dry_run:
        print(f"[DRY-RUN] Would synchronize {meta_path} -> {clean_repo_id}")
        return 0

    success = push_kaggle_dataset_metadata(clean_repo_id, meta_path)
    return 0 if success else 1


def main() -> None:
    """CLI entrypoint to push dataset metadata to Kaggle."""
    parser = argparse.ArgumentParser(description="Synchronize dataset metadata and column descriptors to Kaggle.")
    parser.add_argument("--repo_id", default=None, help="Kaggle dataset handle (owner/slug)")
    parser.add_argument("--metadata_dir", default=None, help="Directory containing dataset-metadata.json")
    parser.add_argument("--all", action="store_true", help="Synchronize metadata for all manifolds in LemGendaryDatasets")
    parser.add_argument("--dry-run", action="store_true", help="Simulate metadata synchronization without making Kaggle API mutations")
    args = parser.parse_args()

    code = sync_metadata(
        repo_id=args.repo_id,
        metadata_dir=args.metadata_dir,
        all_datasets=args.all,
        dry_run=args.dry_run,
    )
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
