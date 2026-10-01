#!/usr/bin/env python3
"""
LemGendary Kaggle Metadata & Column Descriptors Synchronizer CLI
================================================================
Synchronizes dataset-metadata.json, licenses, keywords, and column-level
schema descriptors directly to Kaggle's backend via the Dataset Settings API.
Can be executed standalone without re-uploading dataset archives.
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.common_sync import push_kaggle_dataset_metadata


def main():
    """CLI entrypoint to push dataset metadata to Kaggle."""
    parser = argparse.ArgumentParser(description="Synchronize dataset metadata and column descriptors to Kaggle.")
    parser.add_argument("--repo_id", default=None, help="Kaggle dataset handle (owner/slug)")
    parser.add_argument("--metadata_dir", default=None, help="Directory containing dataset-metadata.json")
    parser.add_argument("--all", action="store_true", help="Synchronize metadata for all manifolds in LemGendaryDatasets")
    parser.add_argument("--dry-run", action="store_true", help="Simulate metadata synchronization without making Kaggle API mutations")
    args = parser.parse_args()

    datasets_root = _ROOT.parent / "LemGendaryDatasets"

    if args.all:
        if not datasets_root.exists():
            print(f"[ERROR] Datasets directory not found at {datasets_root}")
            sys.exit(1)

        meta_files = sorted(datasets_root.glob("*/dataset-metadata.json"))
        if not meta_files:
            print(f"[WARNING] No dataset-metadata.json files found in {datasets_root}")
            sys.exit(0)

        # Pre-fetch existing remote datasets to match published handles
        from core.common_sync import setup_kaggle_auth
        setup_kaggle_auth()
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        owner = "lemtreursi"
        existing_datasets: dict[str, str] = {}
        try:
            remote_ds_list = api.dataset_list(user=owner) or []
            existing_datasets = {
                ds.ref.lower(): ds.ref
                for ds in remote_ds_list
                if ds is not None and getattr(ds, "ref", None) is not None
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
                    elif (raw_id + "large").lower() in existing_datasets:
                        target_repo_id = existing_datasets[(raw_id + "large").lower()]
                else:
                    target_repo_id = raw_id

                if not target_repo_id:
                    print(f"[SKIP] {manifold_dir.name} -> {raw_id} not published on Kaggle yet.")
                    skipped += 1
                    continue

                if target_repo_id in synced_slugs:
                    # Avoid duplicate updates for paired regular/Large folders
                    continue
                synced_slugs.add(target_repo_id)

                if args.dry_run:
                    print(f"\n[DRY-RUN] Would synchronize {manifold_dir.name} -> {target_repo_id}:")
                    print(f"  Title: {meta.get('title')}")
                    print(f"  Subtitle: {meta.get('subtitle')}")
                    print(f"  License: {meta.get('licenses')}")
                    print(f"  Keywords: {meta.get('keywords')}")
                    print(f"  Sources: {meta.get('provenanceSources')}")
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

        action_word = "Simulated" if args.dry_run else "Completed"
        print(f"\n{action_word} synchronization: {successes} succeeded, {skipped} skipped, {failures} failed.")
        if failures > 0:
            sys.exit(1)
    else:
        metadata_dir = args.metadata_dir or r"LemGendaryDatasets\LemGendizedForexUniverse"
        repo_id = args.repo_id or "lemtreursi/lemgendizedforexuniverselarge"
        meta_path = Path(metadata_dir)

        if args.dry_run:
            print(f"[DRY-RUN] Would synchronize {meta_path} -> {repo_id}")
            sys.exit(0)

        success = push_kaggle_dataset_metadata(repo_id, meta_path)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
