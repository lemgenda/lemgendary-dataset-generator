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
import logging
import sys
from pathlib import Path

from core.common_sync import (
    is_kaggle_large_manifold,
    push_kaggle_dataset_metadata,
    setup_kaggle_auth,
)
from core.docs.context import find_unified_entry
from core.kaggle_identity import DEFAULT_KAGGLE_OWNER
from core.lazy_imports import load_kaggle_api
from tools.manifold_sync import ROOT, resolve_kaggle_ref, resolve_manifold_folder

logger = logging.getLogger(__name__)

_OWNER = DEFAULT_KAGGLE_OWNER
_SYNC_ERRORS = (OSError, ValueError, KeyError, AttributeError, RuntimeError, ImportError)


def is_large_manifold(repo_id: str) -> bool:
    """Detect whether a repository handle points to a legacy *large manifold on Kaggle."""
    return is_kaggle_large_manifold(repo_id)


def _fetch_existing_datasets(owner: str) -> dict[str, str]:
    """Published (non-*large) dataset refs of ``owner``, keyed by lower-case ref."""
    setup_kaggle_auth()
    api = load_kaggle_api()()
    api.authenticate()
    try:
        remote_ds_list = api.dataset_list(user=owner) or []
        existing_datasets = {
            ds.ref.lower(): ds.ref
            for ds in remote_ds_list
            if ds is not None and getattr(ds, "ref", None) is not None
            and not is_large_manifold(ds.ref)
        }
        print(
            f"[INFO] Discovered {len(existing_datasets)} active published datasets on"
            f" Kaggle for @{owner}"
        )
        return existing_datasets
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Could not pre-fetch Kaggle dataset list for %s: %s", owner, exc)
        print(
            f"[WARNING] Could not pre-fetch remote dataset list ({exc}); will attempt"
            " direct push for all."
        )
        return {}


def _registry_repo_id(manifold_dir: Path, existing_datasets: dict[str, str]) -> str | None:
    """Kaggle ref from unified_data.yaml, preferring the published spelling."""
    u_entry = find_unified_entry(manifold_dir.name)
    if not (u_entry and u_entry.get("kaggle_ref")):
        return None
    clean_kref = u_entry["kaggle_ref"].replace("kaggle://", "").lower()
    if clean_kref in existing_datasets:
        return existing_datasets[clean_kref]
    if not is_large_manifold(clean_kref):
        return clean_kref
    return None


def _target_repo_id(
    meta: dict, manifold_dir: Path, existing_datasets: dict[str, str]
) -> str:
    """Kaggle handle to push a manifold's metadata to."""
    raw_id = meta.get("id") or f"{_OWNER}/{manifold_dir.name.lower().replace('_', '')}"
    if existing_datasets:
        target = existing_datasets.get(raw_id.lower()) or _registry_repo_id(
            manifold_dir, existing_datasets
        )
        if target:
            return target
    return raw_id


def _print_dry_run(manifold_dir: Path, target_repo_id: str, meta: dict) -> None:
    """Show what a metadata push would send."""
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


def _sync_one_manifold(
    mf: Path, existing_datasets: dict[str, str], synced_slugs: set[str], dry_run: bool
) -> str | None:
    """Push one manifold's metadata; returns "success" / "skipped" / "failure" or None (dup)."""
    manifold_dir = mf.parent
    try:
        meta = json.loads(mf.read_text(encoding="utf-8"))
        target_repo_id = _target_repo_id(meta, manifold_dir, existing_datasets)

        if is_large_manifold(target_repo_id):
            print(
                f"[SKIP] Refusing to touch legacy *large manifold: {target_repo_id}"
                f" ({manifold_dir.name})"
            )
            return "skipped"

        if target_repo_id in synced_slugs:
            return None
        synced_slugs.add(target_repo_id)

        if dry_run:
            _print_dry_run(manifold_dir, target_repo_id, meta)
            return "success"

        print(f"\n[SYNCING] {manifold_dir.name} -> {target_repo_id}...")
        ok = push_kaggle_dataset_metadata(target_repo_id, manifold_dir)
        return "success" if ok else "failure"
    except _SYNC_ERRORS as exc:
        logger.error("Failed to push metadata for %s: %s", manifold_dir, exc)
        print(f"[ERROR] Failed to push metadata for {manifold_dir.name}: {exc}")
        return "failure"


def _sync_all_metadata(datasets_root: Path, dry_run: bool) -> int:
    """Push metadata for every manifold folder under ``datasets_root``."""
    if not datasets_root.exists():
        print(f"[ERROR] Datasets directory not found at {datasets_root}")
        return 1

    meta_files = sorted(datasets_root.glob("*/dataset-metadata.json"))
    if not meta_files:
        print(f"[WARNING] No dataset-metadata.json files found in {datasets_root}")
        return 0

    existing_datasets = {} if dry_run else _fetch_existing_datasets(_OWNER)

    print(f"Discovered {len(meta_files)} manifolds with local metadata in {datasets_root}")
    counts = {"success": 0, "skipped": 0, "failure": 0}
    synced_slugs: set[str] = set()
    for mf in meta_files:
        outcome = _sync_one_manifold(mf, existing_datasets, synced_slugs, dry_run)
        if outcome:
            counts[outcome] += 1

    action_word = "Simulated" if dry_run else "Completed"
    print(
        f"\n{action_word} synchronization: {counts['success']} succeeded, {counts['skipped']}"
        f" skipped, {counts['failure']} failed."
    )
    return 1 if counts["failure"] > 0 else 0


def _single_meta_path(metadata_dir: str | Path | None, datasets_root: Path) -> Path:
    """Folder holding the metadata for a single-manifold sync."""
    if not metadata_dir:
        return datasets_root / "LemGendizedForexUniverse"
    candidate = Path(metadata_dir)
    if candidate.exists():
        return candidate
    return datasets_root / resolve_manifold_folder(str(metadata_dir))


def _single_repo_id(meta_path: Path) -> str:
    """Kaggle handle for a manifold folder: registry ref, metadata id, or derived slug."""
    resolved_ref = resolve_kaggle_ref(meta_path.name)
    if resolved_ref:
        return resolved_ref
    repo_id = None
    meta_file = meta_path / "dataset-metadata.json"
    if meta_file.exists():
        try:
            repo_id = json.loads(meta_file.read_text(encoding="utf-8")).get("id")
        except (OSError, json.JSONDecodeError):
            repo_id = None
    return repo_id or f"{_OWNER}/{meta_path.name.lower()}"


def sync_metadata(
    repo_id: str | None = None,
    metadata_dir: str | Path | None = None,
    all_datasets: bool = False,
    dry_run: bool = False,
) -> int:
    """Synchronize dataset metadata and column descriptors to Kaggle (returns 0 on success)."""
    datasets_root = (
        Path(metadata_dir).resolve()
        if (all_datasets and metadata_dir)
        else (ROOT.parent / "LemGendaryDatasets")
    )

    if all_datasets:
        return _sync_all_metadata(datasets_root, dry_run)

    meta_path = _single_meta_path(metadata_dir, datasets_root)
    if not repo_id:
        repo_id = _single_repo_id(meta_path)

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
    parser = argparse.ArgumentParser(
        description="Synchronize dataset metadata and column descriptors to Kaggle."
    )
    parser.add_argument("--repo_id", default=None, help="Kaggle dataset handle (owner/slug)")
    parser.add_argument(
        "--metadata_dir", default=None, help="Directory containing dataset-metadata.json"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Synchronize metadata for all manifolds in LemGendaryDatasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate metadata synchronization without making Kaggle API mutations",
    )
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
