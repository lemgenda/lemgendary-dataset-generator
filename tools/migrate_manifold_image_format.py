"""
Retroactive image-format migration.

Phase 3 of the 2026 modernization roadmap.

Walks an existing manifold and rewrites each image, target, and mask to
the configured output format, in place.

Hardlink preservation
---------------------
The migration uses `open(path, "wb")` — which truncates the file but
keeps its inode — rather than creating a new file. All NTFS/POSIX
hardlinks pointing at that inode therefore see the transcoded bytes.

Because the extension cannot change without breaking the hardlink, the
original filename is preserved even when the underlying byte format
changes (e.g. a `.jpg` file may contain WebP bytes after migration).
PIL sniffs the format from the magic number, so downstream loaders are
unaffected. The registry's `img_format` column records the actual byte
format.

Usage:
    python -m tools.migrate_manifold_image_format --manifold <path>
    python -m tools.migrate_manifold_image_format --manifold <path> --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from core.config_schema import ImageFormatPolicy
from formats.transcode import ImageTranscoder

logger = logging.getLogger(__name__)

_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ImageFormatPolicy field -> argparse destination of the CLI flag that sets it.
_POLICY_ARGS: tuple[tuple[str, str], ...] = (
    ("format", "image_format"),
    ("quality", "image_quality"),
    ("target_quality", "target_quality"),
    ("mask_format", "mask_format"),
)

# Transcode kind, matching ImageTranscoder.encode()'s Literal parameter.
_KindLiteral = Literal["image", "target", "mask"]


def _iter_manifold_images(manifold: Path) -> Iterator[Path]:
    """Walk images / targets / masks across every split. Yields file paths."""
    for sub in ("images", "targets", "masks"):
        for split in ("train", "val", "test"):
            d = manifold / sub / split
            if not d.exists():
                continue
            try:
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                        yield p
            except OSError as exc:
                logger.warning("Could not list manifold directory %s: %s", d, exc)
                continue


def _kind_from_path(p: Path, root: Path) -> _KindLiteral:
    """Return the transcode kind for a file based on its top-level folder.

    Returns one of the three literals accepted by ImageTranscoder.encode():
    "mask" for files under masks/, "target" for files under targets/, and
    "image" for anything else (or paths that don't resolve under root).
    """
    try:
        rel = p.relative_to(root)
    except ValueError:
        logger.debug("Path %s is outside manifold root %s; treating as image", p, root)
        return "image"
    top = rel.parts[0].lower() if rel.parts else "images"
    if top == "masks":
        return "mask"
    if top == "targets":
        return "target"
    return "image"


def _unique_by_inode(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield one path per physical inode, deduplicating hardlinked copies.

    Prevents transcoding the same underlying bytes twice when a manifold
    uses hardlinks to duplicate targets back to their sources.
    """
    seen: set[tuple[int, int]] = set()
    for p in paths:
        try:
            st = p.stat()
        except OSError as exc:
            logger.warning("Could not stat %s, skipping: %s", p, exc)
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            continue
        seen.add(key)
        yield p


def _transcode_in_place(
    path: Path,
    transcoder: ImageTranscoder,
    kind: _KindLiteral,
) -> tuple[int, int]:
    """Transcode one file in place. Returns (old_size, new_size).

    In-place write (`open(path, "wb")`) preserves the inode, which means
    every hardlink pointing at that file sees the new bytes without any
    additional work.
    """
    old_size = path.stat().st_size
    with Image.open(path) as img:
        img.load()
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        encoded, _fmt = transcoder.encode(img, kind=kind)
    with open(path, "wb") as f:
        f.write(encoded)
    return old_size, path.stat().st_size


def _update_registry_sizes(
    db_path: Path,
    updates: list[tuple[int, str, str]],
) -> None:
    """Batch-update registry rows with (img_size_bytes, img_format, name)."""
    if not db_path.exists() or not updates:
        return
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.executemany(
            "UPDATE registry SET img_size_bytes = ?, img_format = ? WHERE name = ?",
            updates,
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[WARN] Registry update failed: {e}")


@dataclass
class _MigrationTally:
    """Running totals of an in-place transcode pass."""

    total_old: int = 0
    total_new: int = 0
    processed: int = 0
    failed: int = 0
    registry_updates: list[tuple[int, str, str]] = field(default_factory=list)


def _registry_format(transcoder: ImageTranscoder, kind: _KindLiteral) -> str:
    """Return the registry `img_format` value written for a file of `kind`."""
    fmt = transcoder.policy.mask_format if kind == "mask" else transcoder.policy.format
    return "webp" if fmt == "webp-lossless" else fmt


def _transcode_all(
    manifold: Path, unique_paths: list[Path], transcoder: ImageTranscoder
) -> _MigrationTally:
    """Transcode every path in place, collecting sizes and registry updates."""
    tally = _MigrationTally()
    for path in unique_paths:
        kind = _kind_from_path(path, manifold)
        try:
            old, new = _transcode_in_place(path, transcoder, kind=kind)
            tally.total_old += old
            tally.total_new += new
            tally.processed += 1
            tally.registry_updates.append((new, _registry_format(transcoder, kind), path.stem))
            if tally.processed % 500 == 0:
                print(f"  [{tally.processed}/{len(unique_paths)}] processed...")
        except (OSError, ValueError, RuntimeError) as e:
            tally.failed += 1
            logger.warning("Transcode failed for %s: %s", path, e)
            print(f"  [FAIL] {path.name}: {e}")
    return tally


def _report_tally(tally: _MigrationTally, db_path: Path) -> None:
    """Print the size summary and write the registry updates."""
    print()
    print(f"[DONE] processed: {tally.processed}  failed: {tally.failed}")
    if tally.total_old > 0:
        saved_pct = (1 - tally.total_new / tally.total_old) * 100
        print(
            f"[DONE] old: {tally.total_old / 1024**3:.2f} GB  "
            f"new: {tally.total_new / 1024**3:.2f} GB  "
            f"({saved_pct:.1f}% saved)"
        )

    if tally.registry_updates:
        _update_registry_sizes(db_path, tally.registry_updates)
        print(f"[DONE] registry updated for {len(tally.registry_updates)} rows")


def migrate_manifold(
    manifold: Path,
    policy: ImageFormatPolicy,
    dry_run: bool,
) -> int:
    """Transcode every physical file in the manifold to the target format."""
    transcoder = ImageTranscoder(policy)
    if not transcoder.enabled:
        print("[ABORT] Policy format is 'keep' — nothing to migrate.")
        return 0

    print(f"[MIGRATE] {manifold.name}")
    print(
        f"  policy: {policy.format} q={policy.quality} | "
        f"target q={policy.target_quality} | mask={policy.mask_format}"
    )

    all_paths = list(_iter_manifold_images(manifold))
    unique_paths = list(_unique_by_inode(all_paths))
    print(f"  files (logical):  {len(all_paths)}")
    print(f"  inodes (physical): {len(unique_paths)}")
    print()

    if dry_run:
        print("[DRY-RUN] Not writing any files.")
        return 0

    tally = _transcode_all(manifold, unique_paths, transcoder)
    _report_tally(tally, manifold / "manifold_registry.db")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the image-format migration."""
    parser = argparse.ArgumentParser(
        description="Retroactive image-format migration"
    )
    parser.add_argument(
        "--manifold", type=str, required=True,
        help="Path to the manifold folder to migrate",
    )
    parser.add_argument(
        "--image-format", type=str, default="webp",
        choices=["webp", "jpeg", "png", "keep"],
    )
    parser.add_argument("--image-quality", type=int, default=92)
    parser.add_argument("--target-quality", type=int, default=95)
    parser.add_argument(
        "--mask-format", type=str, default="webp-lossless",
        choices=["webp-lossless", "png"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Transcode a manifold's images to the requested format; returns the exit code."""
    manifold = Path(args.manifold)
    if not manifold.exists():
        print(f"[ERROR] Manifold not found: {manifold}")
        return 1

    return migrate_manifold(manifold, image_policy_from_args(args), args.dry_run)


def image_policy_from_args(args: argparse.Namespace) -> ImageFormatPolicy:
    """Build an ImageFormatPolicy from the --image-format/--*-quality/--mask-format args."""
    return ImageFormatPolicy(
        **{field_name: getattr(args, arg_name) for field_name, arg_name in _POLICY_ARGS}
    )


def main() -> int:
    """Parse the CLI arguments and run the image-format migration."""
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
