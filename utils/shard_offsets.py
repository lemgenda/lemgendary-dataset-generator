"""Shard offset index (``shard_offsets.json``) of WebDataset shard directories.

One file per shard directory (``shards/<split>/``) records, for every plain tar
shard in it, where each regular member's payload starts::

    {"version": 2, "shards": {"shard-00000.data": {
        "size": 17244160, "mtime_ns": 1762000000000000000,
        "names": [...], "offsets": [...], "sizes": [...]}}}

The training suite reads it (``training/data/containers/shard_tables.py``,
``SHIPPED_INDEX_FILENAME`` and ``STORE_VERSION``) instead of reading every tar
header of a split, which took up to 48 minutes on Kaggle's network storage. A
stored table is used there when the shard's size matches and either its
modification time matches too or the headers of the first, middle and last
member agree with the table, so the file stays valid after a Kaggle upload.
This module is the compiler's single writer and checker of that format; every
step that writes or rewrites shards calls :func:`write_shard_offsets` once it
is done. The file name has no leading dot because Kaggle's upload skips
hidden files.

A compressed shard cannot be read by offset and gets no entry. Writing reuses
the entry of a shard whose size and modification time are unchanged, so a
repair that rewrote a few shards rescans only those.
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from utils.shard_files import SHARD_SUFFIXES

__all__ = [
    "OFFSETS_FILENAME",
    "OFFSETS_VERSION",
    "MemberRef",
    "OffsetsWrite",
    "headers_match",
    "offsets_problems",
    "scan_tar_members",
    "shard_dirs",
    "write_shard_offsets",
]

logger = logging.getLogger(__name__)

OFFSETS_FILENAME = "shard_offsets.json"
"""Index file next to the shards (the training suite's ``SHIPPED_INDEX_FILENAME``)."""

OFFSETS_VERSION = 2
"""Layout version (the training suite's ``STORE_VERSION``); readers ignore other versions."""

_HEADER_SIZE = tarfile.BLOCKSIZE


class MemberRef(NamedTuple):
    """Location of one regular tar member's payload."""

    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class OffsetsWrite:
    """Outcome of indexing one shard directory."""

    shard_dir: Path
    path: Path | None
    shards: int
    scanned: int
    members: int

    @property
    def bytes_written(self) -> int:
        """Size of the written index file (0 when none was written)."""
        return self.path.stat().st_size if self.path is not None else 0


def shard_dirs(shards_root: Path) -> list[Path]:
    """Every directory at or below ``shards_root`` that directly holds shards, sorted."""
    if not shards_root.is_dir():
        return []
    dirs = {p.parent for p in shards_root.rglob("*")
            if p.is_file() and p.suffix in SHARD_SUFFIXES}
    return sorted(dirs)


def _shards_in(shard_dir: Path) -> list[Path]:
    return sorted(p for p in shard_dir.iterdir() if p.is_file() and p.suffix in SHARD_SUFFIXES)


def scan_tar_members(path: Path) -> list[MemberRef] | None:
    """Payload offsets of every regular member of a plain tar, in archive order.

    ``None`` for a shard that cannot be read by offset (compressed, or holding
    sparse members). A damaged archive raises ``tarfile.TarError``/``OSError``.
    """
    try:
        with tarfile.open(path, mode="r:") as tf:
            members = tf.getmembers()
    except tarfile.ReadError as exc:
        logger.debug("Shard %s is not a plain tar (%s); no offset entry.", path, exc)
        return None
    refs: list[MemberRef] = []
    for member in members:
        if not member.isfile():
            continue
        if member.issparse():
            logger.debug("Shard %s holds sparse member %s; no offset entry.", path, member.name)
            return None
        refs.append(MemberRef(member.name, int(member.offset_data), int(member.size)))
    return refs


def _spot_check_refs(refs: list[MemberRef]) -> list[MemberRef]:
    """First, middle and last member of a table (fewer for short tables)."""
    if not refs:
        return []
    return [refs[i] for i in sorted({0, len(refs) // 2, len(refs) - 1})]


def _header_matches(header: bytes, ref: MemberRef) -> bool:
    """Whether the header block before ``ref``'s payload describes ``ref`` (name prefix)."""
    try:
        info = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
    except tarfile.HeaderError as exc:
        logger.debug("No valid tar header before member %s: %s", ref.name, exc)
        return False
    return info.size == ref.size and bool(info.name) and ref.name.startswith(info.name)


def headers_match(path: Path, refs: list[MemberRef]) -> bool:
    """Whether the headers of the first, middle and last member agree with ``refs``.

    The training suite's check of a table whose modification time differs.
    """
    with open(path, "rb") as handle:
        for ref in _spot_check_refs(refs):
            if ref.offset < _HEADER_SIZE:
                return False
            handle.seek(ref.offset - _HEADER_SIZE)
            header = handle.read(_HEADER_SIZE)
            if len(header) != _HEADER_SIZE or not _header_matches(header, ref):
                return False
    return True


def _encode(shard: Path, refs: list[MemberRef]) -> dict[str, Any]:
    stat = shard.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "names": [ref.name for ref in refs],
        "offsets": [ref.offset for ref in refs],
        "sizes": [ref.size for ref in refs],
    }


def _decode(entry: Any, shard_size: int) -> list[MemberRef] | None:
    """Table of a stored entry; ``None`` when malformed or past the shard's end."""
    if not isinstance(entry, dict):
        return None
    names, offsets, sizes = entry.get("names"), entry.get("offsets"), entry.get("sizes")
    if not (isinstance(names, list) and isinstance(offsets, list) and isinstance(sizes, list)):
        return None
    if not len(names) == len(offsets) == len(sizes):
        return None
    refs: list[MemberRef] = []
    for name, offset, size in zip(names, offsets, sizes, strict=True):
        if not (isinstance(name, str) and isinstance(offset, int) and isinstance(size, int)):
            return None
        if offset < 0 or size < 0 or offset + size > shard_size:
            return None
        refs.append(MemberRef(name, offset, size))
    return refs


class _StoredIndex(NamedTuple):
    """A directory's index file as read: its shard entries, or why it cannot be used."""

    entries: dict[str, Any]
    problem: str | None


def _read_index(path: Path) -> _StoredIndex:
    if not path.is_file():
        return _StoredIndex({}, None)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return _StoredIndex({}, f"unreadable ({exc})")
    if not isinstance(data, dict) or data.get("version") != OFFSETS_VERSION:
        return _StoredIndex({}, f"not layout version {OFFSETS_VERSION}")
    shards = data.get("shards")
    if not isinstance(shards, dict):
        return _StoredIndex({}, "no shards mapping")
    return _StoredIndex(shards, None)


def _reusable(entry: Any, shard: Path) -> list[MemberRef] | None:
    """Table of ``entry`` when it was written for ``shard`` as it is now (size and mtime)."""
    if not isinstance(entry, dict):
        return None
    stat = shard.stat()
    if entry.get("size") != stat.st_size or entry.get("mtime_ns") != stat.st_mtime_ns:
        return None
    return _decode(entry, stat.st_size)


def _write_atomic(path: Path, payload: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            logger.warning("Removing leftover temp index %s", tmp)
            tmp.unlink()


def _index_dir(shard_dir: Path) -> OffsetsWrite:
    """Write (or refresh) the index of one shard directory."""
    path = shard_dir / OFFSETS_FILENAME
    shards = _shards_in(shard_dir)
    old = _read_index(path).entries
    entries: dict[str, Any] = {}
    scanned = members = 0
    for shard in shards:
        previous = old.get(shard.name)
        kept = _reusable(previous, shard)
        if kept is not None:
            entries[shard.name] = previous
            members += len(kept)
            continue
        refs = scan_tar_members(shard)
        scanned += 1
        if refs is not None:
            entries[shard.name] = _encode(shard, refs)
            members += len(refs)
    if not entries:
        if path.is_file():
            path.unlink()
        return OffsetsWrite(shard_dir, None, len(shards), scanned, 0)
    payload = json.dumps({"version": OFFSETS_VERSION, "shards": entries}, separators=(",", ":"))
    _write_atomic(path, payload)
    return OffsetsWrite(shard_dir, path, len(shards), scanned, members)


def write_shard_offsets(shards_root: Path) -> list[OffsetsWrite]:
    """Write ``shard_offsets.json`` in every shard directory at or below ``shards_root``.

    Called by every step that writes or rewrites shards, after its last shard
    is in place. A directory whose shards are all compressed, or that holds no
    shards any more, gets no file (a stale one is removed).
    """
    dirs = shard_dirs(shards_root)
    results = [_index_dir(shard_dir) for shard_dir in dirs]
    if shards_root.is_dir():
        for orphan in sorted(shards_root.rglob(OFFSETS_FILENAME)):
            if orphan.parent not in dirs:
                logger.info("Removing shard index %s: its folder holds no shards.", orphan)
                orphan.unlink()
    for result in results:
        logger.info("Shard index %s: %d shards, %d rescanned, %d members.",
                    result.shard_dir, result.shards, result.scanned, result.members)
    return results


def _is_plain_tar(shard: Path) -> bool:
    try:
        with tarfile.open(shard, mode="r:") as tf:
            tf.next()
    except tarfile.ReadError:
        return False
    return True


def _entry_problem(entry: Any, shard: Path, members: int | None) -> str | None:
    """Why ``entry`` would not be accepted for ``shard`` by the training suite, if it would not."""
    stat = shard.stat()
    if entry is None:
        return None if not _is_plain_tar(shard) else "no entry"
    if not isinstance(entry, dict) or entry.get("size") != stat.st_size:
        return "size differs from the shard"
    refs = _decode(entry, stat.st_size)
    if refs is None:
        return "malformed entry"
    if members is not None and len(refs) != members:
        return f"{len(refs)} members indexed, {members} in the shard"
    if entry.get("mtime_ns") != stat.st_mtime_ns and not headers_match(shard, refs):
        return "headers do not match the entry"
    return None


def _dir_problems(shard_dir: Path, members: Mapping[Path, int]) -> list[str]:
    path = shard_dir / OFFSETS_FILENAME
    stored = _read_index(path)
    label = f"{shard_dir.name}/{OFFSETS_FILENAME}"
    if stored.problem is not None:
        return [f"{label}: {stored.problem}"]
    shards = _shards_in(shard_dir)
    problems = []
    for shard in shards:
        try:
            problem = _entry_problem(stored.entries.get(shard.name), shard, members.get(shard))
        except (OSError, tarfile.TarError) as exc:
            problem = f"cannot be checked ({exc})"
        if problem is not None:
            problems.append(f"{label}: {shard.name}: {problem}")
    names = {shard.name for shard in shards}
    extra = sorted(name for name in stored.entries if name not in names)
    if extra:
        problems.append(f"{label}: {len(extra)} entries without a shard (first: {extra[0]})")
    return problems


def offsets_problems(shards_root: Path, members: Mapping[Path, int] | None = None) -> list[str]:
    """Defects of every ``shard_offsets.json`` present at or below ``shards_root``.

    A directory without the file is not checked. A present file is stale when
    the training suite would reject an entry (size, malformed table, headers),
    when a plain tar shard has no entry, when an entry names no shard, or when
    ``members`` (regular members per shard, from an audit) disagrees with it.
    """
    counts = members or {}
    problems: list[str] = []
    for shard_dir in shard_dirs(shards_root):
        if (shard_dir / OFFSETS_FILENAME).is_file():
            problems.extend(_dir_problems(shard_dir, counts))
    return problems
