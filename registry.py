"""
Manifold registry lifecycle.

Phase 1.5.5 of the 2026 modernization roadmap. Extracted from compiler_core.py.
Registry databases live at ``<manifold>/manifold_registry.db`` (Phase 1.3).
Legacy ``.cache/registry_<name>.db`` files are auto-migrated on first open.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

# Columns added since the pre-1.3 schema. Used by _ensure_registry_schema()
# to ALTER-in missing columns on existing databases.
_REGISTRY_V2_COLUMNS = [
    ("perceptual_hash", "TEXT"),
    ("quality_dist", "BLOB"),
    ("img_format", "TEXT"),
    ("img_size_bytes", "INTEGER"),
    ("target_size_bytes", "INTEGER"),
    ("mask_size_bytes", "INTEGER"),
    ("is_hardlinked", "INTEGER DEFAULT 0"),
    ("reject_code", "TEXT"),
    ("audit_trail", "TEXT"),
    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
]


def ensure_registry_schema(conn: sqlite3.Connection) -> None:
    """Bring an existing registry up to the Phase 1.3 schema. Idempotent."""
    try:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(registry)")}
    except sqlite3.Error:
        existing_cols = set()

    for col_name, col_def in _REGISTRY_V2_COLUMNS:
        if col_name not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE registry ADD COLUMN {col_name} {col_def}")
            except sqlite3.Error:
                if "DEFAULT" in col_def:
                    fallback = col_def.split("DEFAULT")[0].strip()
                    conn.execute(f"ALTER TABLE registry ADD COLUMN {col_name} {fallback}")
                else:
                    raise

    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY,
            sample_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            event_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sample_id) REFERENCES registry(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reject_log (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            sample_name TEXT NOT NULL,
            reject_code TEXT NOT NULL,
            reason TEXT,
            rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_registry_hash ON registry(hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_registry_phash ON registry(perceptual_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_registry_source ON registry(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_registry_task ON registry(task)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reject_code ON reject_log(reject_code)")
    conn.commit()


def initialize_registry(db_path, migrate_from: Path | None = None) -> sqlite3.Connection:
    """Open (or create) a manifold registry database."""
    db_path = Path(db_path)

    if migrate_from is not None and not db_path.exists():
        migrate_from = Path(migrate_from)
        if migrate_from.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(migrate_from), str(db_path))
            print(f"[REGISTRY] Migrated legacy DB: {migrate_from} -> {db_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=100000")
    conn.execute("PRAGMA temp_store=MEMORY")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS registry (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE, source TEXT, task TEXT, split TEXT,
            hash TEXT, nima_score REAL, caption TEXT,
            style_tag TEXT, clip_latent BLOB,
            img_bytes BLOB, cluster_id INTEGER DEFAULT -1,
            perceptual_hash TEXT,
            quality_dist BLOB,
            img_format TEXT,
            img_size_bytes INTEGER,
            target_size_bytes INTEGER,
            mask_size_bytes INTEGER,
            is_hardlinked INTEGER DEFAULT 0,
            reject_code TEXT,
            audit_trail TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    ensure_registry_schema(conn)
    return conn