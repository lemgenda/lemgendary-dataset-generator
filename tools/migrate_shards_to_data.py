"""Atomically migrate WebDataset shard archives from .tar to obfuscated .data extension.

Safeguards Kaggle dataset ingestion against archive sniffing and automatic tar extraction.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from utils.shard_offsets import write_shard_offsets


def _rename_shards(shards_dir: Path, from_ext: str, to_ext: str, dry_run: bool) -> tuple[int, int]:
    """Rename every ``from_ext`` shard under ``shards_dir``; return (count, bytes)."""
    renamed = 0
    total_bytes = 0
    # Find all matching shard files (direct or split subdirectories)
    for src in sorted(shards_dir.glob(f"**/*{from_ext}")):
        total_bytes += src.stat().st_size
        if not dry_run:
            src.rename(src.with_suffix(to_ext))
        renamed += 1
    return renamed, total_bytes


def _verify_migration(datasets_root: Path, from_ext: str) -> int:
    """Post-migration check that no ``from_ext`` shards remain; return exit code."""
    remaining_tars = list(datasets_root.glob(f"**/shards/**/*{from_ext}"))
    if remaining_tars:
        print(f"[WARNING] {len(remaining_tars)} shard(s) still remain with {from_ext}!")
        return 1
    print(f"[VERIFIED] 0 remaining {from_ext} shards. Ingestion obfuscation active.")
    return 0


def migrate_shards(
    datasets_root: Path,
    dry_run: bool = False,
    reverse: bool = False,
) -> int:
    """Rename WebDataset shards under ``datasets_root`` .tar to .data, or back (``reverse``)."""
    datasets_root = datasets_root.resolve()
    if not datasets_root.exists() or not datasets_root.is_dir():
        print(f"[ERROR] Datasets directory not found: {datasets_root}")
        return 1

    from_ext = ".data" if reverse else ".tar"
    to_ext = ".tar" if reverse else ".data"

    print("=" * 80)
    print(f"WebDataset Shard Obfuscation Migration: {from_ext} -> {to_ext}")
    print(f"Target Root: {datasets_root}")
    print(f"Dry Run: {dry_run}")
    print("=" * 80)

    start_time = time.time()
    total_renamed = 0
    total_bytes = 0
    dataset_summary: dict[str, int] = {}

    for manifold_dir in sorted(datasets_root.iterdir()):
        shards_dir = manifold_dir / "shards"
        if not manifold_dir.is_dir() or not shards_dir.is_dir():
            continue

        manifold_renamed, manifold_bytes = _rename_shards(shards_dir, from_ext, to_ext, dry_run)
        if not manifold_renamed:
            continue
        if not dry_run:
            write_shard_offsets(shards_dir)

        total_renamed += manifold_renamed
        total_bytes += manifold_bytes
        dataset_summary[manifold_dir.name] = manifold_renamed
        print(f"[OK] {manifold_dir.name}: {manifold_renamed} shard(s) {from_ext} -> {to_ext}")

    elapsed = time.time() - start_time
    gb_processed = total_bytes / (1024 ** 3)

    print("-" * 80)
    print(
        f"Migration Complete: {total_renamed} shard(s) migrated across {len(dataset_summary)}"
        " dataset(s)."
    )
    print(f"Total Shard Volume: {gb_processed:.2f} GB")
    print(f"Elapsed Time: {elapsed:.2f}s")
    print("=" * 80)

    if not dry_run:
        return _verify_migration(datasets_root, from_ext)
    return 0


def main() -> None:
    """Parse the CLI arguments and run the shard rename."""
    parser = argparse.ArgumentParser(description="Migrate WebDataset shards .tar <-> .data")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets",
        help="Path to LemGendaryDatasets root directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate renames without writing changes"
    )
    parser.add_argument("--reverse", action="store_true", help="Reverse migration (.data -> .tar)")
    args = parser.parse_args()

    exit_code = migrate_shards(args.root, dry_run=args.dry_run, reverse=args.reverse)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
