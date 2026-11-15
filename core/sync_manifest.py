"""Manifest checksums, idempotency ledger, and structured reports for Kaggle sync.

Provides deterministic SHA-256 Merkle manifest generation and verification
for compiled manifolds on push and pull, idempotent push detection, and
machine-readable sync reporting.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.member_paths import matches_prefixes

logger = logging.getLogger(__name__)

SYNC_MANIFEST_FILENAME = "sync_manifest.json"
SYNC_STATE_FILENAME = ".sync_state.json"
_HASH_CHUNK_BYTES = 1 << 20
_IGNORED_NAMES = frozenset({
    SYNC_MANIFEST_FILENAME,
    SYNC_STATE_FILENAME,
    ".DS_Store",
    "Thumbs.db",
})
_IGNORED_SUFFIXES = (".tmp", ".part", ".pyc")


@dataclass(frozen=True)
class ManifestFileEntry:
    """Checksum and size record for a single file inside a compiled manifold."""

    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to a JSON-compatible dictionary."""
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class SyncManifest:
    """Deterministic content manifest for a compiled dataset manifold."""

    schema_version: int
    manifold_name: str
    compiler_version: str
    manifest_sha256: str
    file_count: int
    total_bytes: int
    files: list[ManifestFileEntry]

    @property
    def total_files(self) -> int:
        """Alias of ``file_count``."""
        return self.file_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "manifold_name": self.manifold_name,
            "compiler_version": self.compiler_version,
            "manifest_sha256": self.manifest_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files": [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncManifest:
        """Deserialize a SyncManifest from a parsed JSON mapping."""
        raw_files = data.get("files") or []
        entries = [
            ManifestFileEntry(
                path=str(item["path"]),
                size=int(item["size"]),
                sha256=str(item["sha256"]),
            )
            for item in raw_files
            if isinstance(item, dict) and "path" in item
        ]
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            manifold_name=str(data.get("manifold_name", "")),
            compiler_version=str(data.get("compiler_version", "")),
            manifest_sha256=str(data.get("manifest_sha256", "")),
            file_count=int(data.get("file_count", len(entries))),
            total_bytes=int(data.get("total_bytes", sum(e.size for e in entries))),
            files=entries,
        )


class SyncReport:
    """Machine-readable execution report for a Kaggle push, pull, or metadata operation."""

    def __init__(self, **kwargs: Any) -> None:
        self._data: dict[str, Any] = {
            "action": "",
            "status": "pending",
            "repo_id": "",
            "manifold": "",
            "version": None,
            "pinned_version": None,
            "manifest_sha256": None,
            "checksum_verified": False,
            "idempotent_noop": False,
            "resumed": False,
            "retries": 0,
            "files_count": 0,
            "bytes_transferred": 0,
            "elapsed_seconds": 0.0,
            "errors": [],
        }
        self.update(**kwargs)

    def update(self, **kwargs: Any) -> None:
        """Update multiple report fields in place."""
        for key, value in kwargs.items():
            if key == "errors" and value is not None:
                self._data["errors"] = list(value)
            else:
                self._data[key] = value

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a dictionary."""
        return dict(self._data)

    def to_json(self) -> str:
        """Serialize report to formatted JSON."""
        return json.dumps(self.to_dict(), indent=2)

    def write(self, path: Path) -> None:
        """Write JSON report to ``path``, creating parent directories if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Compute streaming SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _should_ignore(rel_parts: tuple[str, ...], filename: str) -> bool:
    if filename in _IGNORED_NAMES or filename.endswith(_IGNORED_SUFFIXES):
        return True
    return any(part.startswith(".") or part == "__pycache__" for part in rel_parts)


def compute_merkle_root(entries: list[ManifestFileEntry]) -> str:
    """Compute canonical Merkle root SHA-256 across sorted file entries."""
    root_hasher = hashlib.sha256()
    for entry in sorted(entries, key=lambda e: e.path):
        line = f"{entry.path}:{entry.size}:{entry.sha256}\n"
        root_hasher.update(line.encode("utf-8"))
    return root_hasher.hexdigest()


def _scan_manifest_entries(manifold_dir: Path) -> list[ManifestFileEntry]:
    """Walk ``manifold_dir`` and return sorted file entries with SHA-256 digests."""
    entries: list[ManifestFileEntry] = []
    for root, dirs, files in os.walk(manifold_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        root_path = Path(root)
        for fname in sorted(files):
            full_path = root_path / fname
            rel_path = full_path.relative_to(manifold_dir)
            if not _should_ignore(rel_path.parts, fname) and full_path.is_file():
                entries.append(
                    ManifestFileEntry(
                        path=rel_path.as_posix(),
                        size=full_path.stat().st_size,
                        sha256=sha256_file(full_path),
                    )
                )
    entries.sort(key=lambda e: e.path)
    return entries


def compute_sync_manifest(
    manifold_dir: Path,
    compiler_version: str = "",
    write_file: bool = True,
) -> SyncManifest:
    """Scan ``manifold_dir``, compute per-file and Merkle SHA-256 checksums, and optionally save."""
    manifold_dir = Path(manifold_dir).resolve()
    entries = _scan_manifest_entries(manifold_dir)
    manifest = SyncManifest(
        schema_version=1,
        manifold_name=manifold_dir.name,
        compiler_version=compiler_version,
        manifest_sha256=compute_merkle_root(entries),
        file_count=len(entries),
        total_bytes=sum(e.size for e in entries),
        files=entries,
    )
    if write_file and manifold_dir.exists():
        manifest_path = manifold_dir / SYNC_MANIFEST_FILENAME
        rendered = json.dumps(manifest.to_dict(), indent=2)
        existing = (
            manifest_path.read_text(encoding="utf-8")
            if manifest_path.is_file()
            else None
        )
        if existing != rendered:
            manifest_path.write_text(rendered, encoding="utf-8")
    return manifest


def _verify_sidecar_sizes(manifold_dir: Path) -> list[str]:
    """Verify that any .data.index.json sidecar agrees with its shard's byte size on disk."""
    errors: list[str] = []
    shards_dir = manifold_dir / "shards"
    if not shards_dir.exists():
        return errors
    for sidecar in sorted(shards_dir.rglob("*.data.index.json")):
        shard_path = sidecar.with_name(sidecar.name.removesuffix(".index.json"))
        if not shard_path.is_file():
            errors.append(f"Missing shard for sidecar {sidecar.relative_to(manifold_dir)}")
            continue
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            shard_meta = payload.get("shard") if isinstance(payload, dict) else None
            if isinstance(shard_meta, dict):
                expected_size = shard_meta.get("size")
            elif isinstance(payload, dict):
                expected_size = payload.get("size")
            else:
                expected_size = None
            actual_size = shard_path.stat().st_size
            if expected_size is not None and int(expected_size) != actual_size:
                errors.append(
                    f"Sidecar index size mismatch for {shard_path.name}: "
                    f"expected {expected_size}, got {actual_size}"
                )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"Corrupt shard sidecar {sidecar.name}: {exc}")
    return errors


def verify_sync_manifest(
    manifold_dir: Path,
    expected_sha256: str | None = None,
    only: tuple[str, ...] = (),
) -> tuple[bool, str, list[str]]:
    """Verify ``manifold_dir`` against its ``sync_manifest.json`` and shard sidecars.

    Returns ``(passed, manifest_sha256, errors)``.
    """
    manifold_dir = Path(manifold_dir).resolve()
    if not manifold_dir.exists():
        return False, "", [f"Manifold directory does not exist: {manifold_dir}"]

    errors: list[str] = _verify_sidecar_sizes(manifold_dir)
    manifest_path = manifold_dir / SYNC_MANIFEST_FILENAME

    if manifest_path.is_file():
        try:
            saved = SyncManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return False, "", [f"Failed reading {SYNC_MANIFEST_FILENAME}: {exc}"]

        if not only:
            recomputed_merkle = compute_merkle_root(saved.files)
            if saved.manifest_sha256 and recomputed_merkle != saved.manifest_sha256:
                errors.append(
                    f"Manifest Merkle root mismatch: recorded {saved.manifest_sha256}, "
                    f"computed {recomputed_merkle}"
                )

        if expected_sha256 and saved.manifest_sha256 != expected_sha256:
            errors.append(
                f"Manifest SHA-256 {saved.manifest_sha256} does not match "
                f"expected {expected_sha256}"
            )

        selected_entries = [
            entry
            for entry in saved.files
            if not only or matches_prefixes(entry.path, only)
        ]
        for entry in selected_entries:
            target = manifold_dir / entry.path
            if not target.is_file():
                errors.append(f"Missing file from manifest: {entry.path}")
                continue
            actual_size = target.stat().st_size
            if actual_size != entry.size:
                errors.append(
                    f"Size mismatch for {entry.path}: expected {entry.size}, got {actual_size}"
                )
                continue
            actual_sha = sha256_file(target)
            if actual_sha != entry.sha256:
                errors.append(
                    f"SHA-256 mismatch for {entry.path}: "
                    f"expected {entry.sha256[:12]}..., got {actual_sha[:12]}..."
                )
        return len(errors) == 0, saved.manifest_sha256, errors

    computed = compute_sync_manifest(manifold_dir, write_file=False)
    if expected_sha256 and computed.manifest_sha256 != expected_sha256:
        errors.append(
            f"Computed manifest SHA-256 {computed.manifest_sha256} does not match "
            f"expected {expected_sha256}"
        )
    return len(errors) == 0, computed.manifest_sha256, errors


def read_sync_state(manifold_dir: Path) -> dict[str, Any]:
    """Read the local sync state ledger (``.sync_state.json``) if present."""
    state_file = Path(manifold_dir) / SYNC_STATE_FILENAME
    if not state_file.is_file():
        return {}
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read sync state from %s: %s", state_file, exc)
        return {}


def write_sync_state(
    manifold_dir: Path,
    repo_id: str,
    manifest_sha256: str,
    remote_version: int,
) -> None:
    """Record a completed push in ``.sync_state.json`` for idempotency checks."""
    state_file = Path(manifold_dir) / SYNC_STATE_FILENAME
    payload = {
        "repo_id": repo_id,
        "manifest_sha256": manifest_sha256,
        "remote_version": remote_version,
        "synced_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    try:
        state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write sync state to %s: %s", state_file, exc)


def is_push_idempotent_noop(
    manifold_dir: Path,
    clean_repo_id: str,
    manifest_sha256: str,
    remote_version_info: dict[str, Any],
) -> bool:
    """Return True when the remote Kaggle dataset already has this exact ``manifest_sha256``."""
    latest_remote = int(remote_version_info.get("latest_version") or 0)
    if latest_remote <= 0 or not manifest_sha256:
        return False

    remote_manifest_sha = remote_version_info.get("manifest_sha256")
    if isinstance(remote_manifest_sha, str) and remote_manifest_sha == manifest_sha256:
        return True

    state = read_sync_state(manifold_dir)
    if (
        state.get("repo_id") == clean_repo_id
        and state.get("manifest_sha256") == manifest_sha256
        and int(state.get("remote_version") or 0) > 0
    ):
        return True
    return False
