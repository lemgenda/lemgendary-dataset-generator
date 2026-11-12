"""The manifold registry during a compile: resumption scans, orphan adoption and inserts.

Every manifold keeps a ``manifold_registry.db`` SQLite file. A compile
resumes from it: names already registered or already on disk are skipped,
and images on disk without a registry row ("orphans") are adopted.
"""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from core.compiler_core import initialize_registry

logger = logging.getLogger(__name__)

REGISTRY_INSERT_SQL = """
    INSERT OR IGNORE INTO registry (
        name, source, task, split, hash,
        nima_score, caption, style_tag, clip_latent, img_bytes, perceptual_hash
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Placeholder hash of orphan rows adopted from disk (never a real digest).
ADOPTED_HASH = "adopted"
_ORPHAN_CHUNK_SIZE = 100000
_SCAN_REPORT_EVERY = 5000


def registry_row(res: dict[str, Any]) -> tuple[Any, ...]:
    """The registry INSERT parameters for one sample record."""
    return (
        res["name"], res["source"], res["task"], res["split"], res["hash"],
        res["nima_score"], res.get("caption"), res.get("style_tag"),
        res.get("clip_latent"), res.get("img_bytes"), res.get("perceptual_hash"),
    )


def insert_rows(conn: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> None:
    """Insert registry rows, ignoring names that are already registered."""
    conn.executemany(REGISTRY_INSERT_SQL, rows)


def open_registry(db_path: Path, pascal_name: str) -> sqlite3.Connection:
    """2026 Phase 1.3: open the registry inside the manifold folder (migrating the legacy one)."""
    _root = (
        Path(__file__).resolve().parent.parent
        if Path(__file__).parent.name == "core"
        else Path(__file__).resolve().parent
    )
    legacy_path = _root / ".cache" / f"registry_{pascal_name}.db"
    return initialize_registry(db_path, migrate_from=legacy_path)


def load_existing_names(conn: sqlite3.Connection, db_path: Path, pascal_name: str) -> set[str]:
    """Names already in the registry (the resumption checkpoint)."""
    existing_names: set[str] = set()
    if db_path.exists():
        print(f"[RESUMPTION] Scanning {pascal_name} registry for existing entries...")
        try:
            rows = conn.execute("SELECT name FROM registry").fetchall()
            existing_names = {r[0] for r in rows}
            if existing_names:
                print(
                    f"[OK] Found {len(existing_names)} existing samples. Resuming from"
                    " checkpoint."
                )
        except sqlite3.Error as e:
            logger.warning("Resumption scan of %s failed: %s", db_path, e)
    return existing_names


def load_registry_hashes(conn: sqlite3.Connection) -> tuple[set[str], list[str]]:
    """Return the exact hashes and ``phash:dhash`` strings already in the registry.

    A resumed compile seeds its dedup state from these so samples registered
    by an earlier run are not compiled a second time.
    """
    exact = {
        row[0]
        for row in conn.execute("SELECT hash FROM registry WHERE hash IS NOT NULL AND hash != ?",
                                (ADOPTED_HASH,))
    }
    perceptual = [
        row[0]
        for row in conn.execute(
            "SELECT perceptual_hash FROM registry WHERE perceptual_hash IS NOT NULL"
        )
    ]
    return exact, perceptual


def _stem_lower(fname: str) -> str:
    """The sample key of a file name (up to its first dot), lower-cased.

    The first dot is deliberate: ``x.target.webp`` and ``x.mask.webp`` are
    companions of sample ``x`` (WebDataset key rule), not samples of their own.
    """
    dot_idx = fname.find('.')
    return fname[:dot_idx].lower() if dot_idx != -1 else fname.lower()


def _scan_split(split_path: Path, found: set[str], count: int) -> int:
    """Add the sample stems of one split folder to ``found``; return the running count."""
    buf: list[str] = []
    try:
        with os.scandir(str(split_path)) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                buf.append(_stem_lower(entry.name))
                count += 1
                if count % _SCAN_REPORT_EVERY == 0:
                    print(f"   -> Indexed {count // 1000}k samples...", flush=True)
                    found.update(buf)
                    buf = []
    except OSError as exc:
        print(f"[WARNING] Directory indexing interrupted: {exc}")
    found.update(buf)
    return count


def scan_physical_index(output_root: Path, pascal_name: str) -> set[str]:
    """2026 Resilience: High-Speed Physical Scan (SOTA v6.2) of the images on disk."""
    existing_on_disk: set[str] = set()
    img_dir = output_root / "images"
    if not img_dir.exists():
        return existing_on_disk
    print(
        f"[RESUMPTION] Surgical scan of {pascal_name} manifold for physical consistency..."
    )
    count = 0
    for split in ["train", "val"]:
        split_path = img_dir / split
        if split_path.exists():
            count = _scan_split(split_path, existing_on_disk, count)

    print(
        f"[OK] Physical discovery complete: {len(existing_on_disk)} samples verified on"
        " disk."
    )
    return existing_on_disk


def _orphan_row(o_key: str, task: str) -> tuple[Any, ...]:
    """Registry row for an image found on disk without a registry entry."""
    parts = o_key.split("_")
    o_source = "_".join(parts[1:-1]) if len(parts) >= 3 else "unknown"
    return (o_key, o_source, task, "train", ADOPTED_HASH, 1.0, None, None, None, None, None)


def adopt_orphans(conn: sqlite3.Connection, existing_names: set[str],
                  existing_on_disk: set[str], task: str) -> None:
    """2026 Orphan Rescue: register images on disk that the registry does not know."""
    lower_registry = {n.lower() for n in existing_names}
    orphans = [k for k in existing_on_disk if k not in lower_registry]
    if not orphans:
        return

    print(f"[REPAIR] Found {len(orphans)} orphans on disk. Commencing batch adoption...")
    total_adopted = 0
    for i in range(0, len(orphans), _ORPHAN_CHUNK_SIZE):
        orphan_entries = [_orphan_row(o_key, task) for o_key in orphans[i:i + _ORPHAN_CHUNK_SIZE]]
        insert_rows(conn, orphan_entries)
        conn.commit()
        total_adopted += len(orphan_entries)
        print(
            f"   -> Adopted {total_adopted // 1000}k / {len(orphans) // 1000}k orphans...",
            flush=True,
        )

    print(f"[OK] [REPAIR] {total_adopted} orphans successfully merged into registry.")
    existing_names.update(orphans)
