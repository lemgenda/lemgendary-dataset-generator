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
    python migrate_manifold_image_format.py --manifold <path>
    python migrate_manifold_image_format.py --manifold <path> --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Iterator, Literal

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_schema import ImageFormatPolicy
from formats.transcode import ImageTranscoder


_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

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
            except OSError:
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
        except OSError:
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

    db_path = manifold / "manifold_registry.db"
    registry_updates: list[tuple[int, str, str]] = []

    total_old = 0
    total_new = 0
    processed = 0
    failed = 0

    for path in unique_paths:
        kind = _kind_from_path(path, manifold)
        try:
            old, new = _transcode_in_place(path, transcoder, kind=kind)
            total_old += old
            total_new += new
            processed += 1
            fmt = (
                transcoder.policy.mask_format
                if kind == "mask"
                else transcoder.policy.format
            )
            if fmt == "webp-lossless":
                fmt = "webp"
            registry_updates.append((new, fmt, path.stem))
            if processed % 500 == 0:
                print(f"  [{processed}/{len(unique_paths)}] processed...")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {path.name}: {e}")

    print()
    print(f"[DONE] processed: {processed}  failed: {failed}")
    if total_old > 0:
        saved_pct = (1 - total_new / total_old) * 100
        print(
            f"[DONE] old: {total_old / 1024**3:.2f} GB  "
            f"new: {total_new / 1024**3:.2f} GB  "
            f"({saved_pct:.1f}% saved)"
        )

    if registry_updates:
        _update_registry_sizes(db_path, registry_updates)
        print(f"[DONE] registry updated for {len(registry_updates)} rows")

    return 0


def build_parser() -> argparse.ArgumentParser:
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
    manifold = Path(args.manifold)
    if not manifold.exists():
        print(f"[ERROR] Manifold not found: {manifold}")
        return 1

    policy = ImageFormatPolicy(
        format=args.image_format,
        quality=args.image_quality,
        target_quality=args.target_quality,
        mask_format=args.mask_format,
    )
    return migrate_manifold(manifold, policy, args.dry_run)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())