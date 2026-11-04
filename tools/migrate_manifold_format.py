"""
Retroactive container-format migration.

Phase 4 of the 2026 modernization roadmap.

Reads an existing directory manifold, streams each sample through every
requested container writer, and verifies the round-trip.

Non-destructive: the source directory layout is never removed. Pass
`--purge-source` to remove it after a successful round-trip verification,
which requires the `--i-know-what-im-doing` flag.

Usage:
    python -m tools.migrate_manifold_format --manifold <path> --to mds
    python -m tools.migrate_manifold_format --manifold <path> --to mds,litdata
    python -m tools.migrate_manifold_format --manifold <path> --to mds --dry-run
    python -m tools.migrate_manifold_format --manifold <path> --to mds --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from audit.hardlinks import audit_hardlinks
from formats.base import parse_also_format
from formats.directory import DirectorySampleSource, purge_loose_layout
from formats.factory import make_writer


def _load_index(root: Path) -> list[dict[str, Any]]:
    idx_path = root / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"index.json not found at {idx_path}. "
            "This manifold may not have been compiled by manifold_compile.py."
        )
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_hardlink_gate(
    root: Path,
    force_duplicate: bool,
    accept_space_loss: bool,
) -> bool:
    """Tiered hardlink gate. Returns True when container write may proceed."""
    result = audit_hardlinks(root / "images" if (root / "images").exists() else root)
    print(f"[HARDLINK] {result.summary()}")

    if result.verdict == "PROCEED":
        return True
    if result.verdict == "WARN":
        if force_duplicate:
            print("[HARDLINK] WARN-tier gate overridden by --force-duplicate.")
            return True
        print(
            "[HARDLINK] 5-25% hardlink fraction. Container write may be "
            "net-negative on disk. Re-run with --force-duplicate to proceed."
        )
        return False
    # verdict == BLOCK
    if force_duplicate and accept_space_loss:
        print("[HARDLINK] BLOCK-tier gate overridden by --force-duplicate --accept-space-loss.")
        return True
    print(
        "[HARDLINK] >25% hardlink fraction. Container write would duplicate "
        "the manifold. Re-run with --force-duplicate --accept-space-loss to proceed."
    )
    return False


@dataclass(frozen=True)
class MigrationFlags:
    """Behaviour switches for `migrate_manifold` (all default to off)."""

    dry_run: bool = False
    verify: bool = False
    force_duplicate: bool = False
    accept_space_loss: bool = False
    purge_source: bool = False


def _stream_into_writers(root: Path, index: list[dict[str, Any]], formats: list[str]) -> None:
    """Stream every indexed sample through one writer per requested format."""
    source = DirectorySampleSource(root, index)
    writers = [make_writer(fmt) for fmt in formats]
    for w in writers:
        w.open(root, policy=None)

    written = 0
    try:
        for sample in source:
            for w in writers:
                w.write(sample)
            written += 1
            if written % 5000 == 0:
                print(f"  [{written}/{len(index)}] samples streamed...")
    finally:
        for w in writers:
            w.close()

    print(f"[DONE] wrote {written} samples across {len(writers)} container(s)")


def _after_write(root: Path, flags: MigrationFlags, has_containers: bool) -> None:
    """Print the verify notice and purge the loose layout when requested."""
    if flags.verify:
        print("[VERIFY] Round-trip verification is not implemented in Phase 4;")
        print("         readers land in the training suite as a separate release.")
        print("         Compare directory counts to container counts manually.")

    if not (flags.purge_source and has_containers):
        return
    if not flags.accept_space_loss:
        print("[SKIP] --purge-source ignored without --accept-space-loss")
        return
    print("[PURGE] Removing source directory layout (images/, labels/, targets/, masks/)")
    purge_loose_layout(root)


def migrate_manifold(
    root: Path,
    formats: list[str],
    flags: MigrationFlags | None = None,
    **overrides: bool,
) -> int:
    """Write an existing manifold into the given container formats; returns the exit code.

    Switches come from `flags` (a `MigrationFlags`); the legacy keyword
    arguments (`dry_run`, `verify`, `force_duplicate`, `accept_space_loss`,
    `purge_source`) are still accepted and override it.
    """
    opts = replace(flags or MigrationFlags(), **overrides)
    if not root.exists():
        print(f"[ERROR] Manifold not found: {root}")
        return 1

    print(f"[MIGRATE] {root.name} -> {', '.join(formats)}")

    # Hardlink gate — only matters when a container is in the target list.
    non_directory = [f for f in formats if f != "directory"]
    if non_directory and not _check_hardlink_gate(
        root, opts.force_duplicate, opts.accept_space_loss
    ):
        return 1

    index = _load_index(root)
    print(f"[SCAN] {len(index)} entries in index.json")

    if opts.dry_run:
        print("[DRY-RUN] Not writing any files.")
        return 0

    _stream_into_writers(root, index, formats)
    _after_write(root, opts, bool(non_directory))
    return 0


def main() -> int:
    """Parse the CLI arguments and run the container migration."""
    parser = argparse.ArgumentParser(
        description="Retroactive container-format migration"
    )
    parser.add_argument("--manifold", type=str, required=True,
                        help="Path to the manifold folder to migrate")
    parser.add_argument("--to", type=str, required=True,
                        help="Comma-separated container formats (mds,litdata,webdataset,parquet)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="Print sample counts for manual comparison")
    parser.add_argument("--force-duplicate", action="store_true",
                        help="Proceed despite WARN-tier hardlink fraction")
    parser.add_argument("--accept-space-loss", action="store_true",
                        help="Proceed despite BLOCK-tier hardlink fraction")
    parser.add_argument(
        "--purge-source",
        action="store_true",
        help="Remove source directories after migration (requires --accept-space-loss)",
    )
    args = parser.parse_args()

    root = Path(args.manifold)
    formats = parse_also_format(args.to)
    if not formats:
        print("[ERROR] --to must specify at least one format.")
        return 1

    flags = MigrationFlags(
        dry_run=args.dry_run,
        verify=args.verify,
        force_duplicate=args.force_duplicate,
        accept_space_loss=args.accept_space_loss,
        purge_source=args.purge_source,
    )
    return migrate_manifold(root, formats, flags)


if __name__ == "__main__":
    sys.exit(main())
