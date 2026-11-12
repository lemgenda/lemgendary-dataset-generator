"""Extract selected members of a remote zip through HTTP byte ranges.

Only the central directory and the byte ranges of the wanted members are
read, so a few folders (for example ``labels/`` and ``classes.txt``) come
out of a 90 GB archive for the cost of their own size. ZIP64 archives are
supported; stored and deflated members are decoded, and every member's
CRC-32 and size are checked before it is written (temporary file, then an
atomic rename). A member already on disk with the right size and CRC is
skipped, so an interrupted run resumes.

The archive is read through a :class:`RangeReader` (``size()`` and an
inclusive ``read(start, end)``); :mod:`core.kaggle_http` provides one over
a signed Kaggle download URL.
"""

from __future__ import annotations

import os
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from utils.member_paths import (
    common_wrapper,
    local_path,
    matches_prefixes,
    relative_to_wrapper,
)

__all__ = [
    "ExtractStats",
    "RangeReader",
    "SelectedMember",
    "ZipEntry",
    "ZipRangeError",
    "extract_members",
    "read_central_directory",
    "select_members",
]

_EOCD_SIG = b"PK\x05\x06"
_ZIP64_EOCD_SIG = b"PK\x06\x06"
_CENTRAL_SIG = b"PK\x01\x02"
_LOCAL_SIG = b"PK\x03\x04"
_TAIL_BYTES = 65536 + 22
_LOCAL_HEADER = 30
_ZIP64_EXTRA_ID = 1
_U32_MAX = 0xFFFFFFFF
_U16_MAX = 0xFFFF
_STORED, _DEFLATED = 0, 8
# Members closer than this are fetched in one request; one request stays below the cap.
_GAP_BYTES = 256 * 1024
_GROUP_CAP = 64 * 1024 * 1024
# Slack after the last member of a group for its local header's name and extra field.
_LOCAL_SLACK = 64 * 1024
_TMP_SUFFIX = ".part"


class ZipRangeError(OSError):
    """The remote archive is not a readable zip, or a member failed its check."""


class RangeReader(Protocol):
    """Random access to the bytes of a remote file."""

    def size(self) -> int:
        """Total size of the file in bytes."""
        raise NotImplementedError

    def read(self, start: int, end: int) -> bytes:
        """Bytes ``start`` to ``end`` inclusive."""
        raise NotImplementedError


@dataclass(frozen=True)
class ZipEntry:
    """One central directory record."""

    name: str
    offset: int
    compressed: int
    size: int
    method: int
    crc: int


@dataclass(frozen=True)
class SelectedMember:
    """A wanted member and its path relative to the archive's wrapper folder."""

    entry: ZipEntry
    rel: str


@dataclass
class ExtractStats:
    """What an extraction wrote, skipped and fetched."""

    written: int = 0
    skipped: int = 0
    bytes_read: int = 0
    failed: list[str] = field(default_factory=list)


def _end_record(tail: bytes) -> tuple[int, int, int]:
    """(entries, central directory size, central directory offset) from the archive tail."""
    at = tail.rfind(_EOCD_SIG)
    if at < 0:
        raise ZipRangeError("no end of central directory record: not a zip archive")
    entries = struct.unpack("<H", tail[at + 10:at + 12])[0]
    cd_size, cd_offset = struct.unpack("<II", tail[at + 12:at + 20])
    if _U32_MAX not in (cd_size, cd_offset) and entries != _U16_MAX:
        return entries, cd_size, cd_offset
    at64 = tail.rfind(_ZIP64_EOCD_SIG)
    if at64 < 0:
        raise ZipRangeError("ZIP64 archive without a ZIP64 end of central directory record")
    entries, cd_size, cd_offset = struct.unpack("<QQQ", tail[at64 + 32:at64 + 56])
    return entries, cd_size, cd_offset


def _zip64_values(extra: bytes, sizes: list[int]) -> None:
    """Replace the 0xFFFFFFFF placeholders of ``sizes`` (size, compressed, offset) in place."""
    pos = 0
    while pos + 4 <= len(extra):
        header_id, length = struct.unpack("<HH", extra[pos:pos + 4])
        if header_id == _ZIP64_EXTRA_ID:
            values = extra[pos + 4:pos + 4 + length]
            read = 0
            for i, value in enumerate(sizes):
                if value == _U32_MAX:
                    sizes[i] = struct.unpack("<Q", values[read:read + 8])[0]
                    read += 8
            return
        pos += 4 + length


def _parse_central_directory(data: bytes) -> list[ZipEntry]:
    entries: list[ZipEntry] = []
    pos = 0
    while pos < len(data):
        if data[pos:pos + 4] != _CENTRAL_SIG:
            raise ZipRangeError(f"bad central directory record at byte {pos}")
        method = struct.unpack("<H", data[pos + 10:pos + 12])[0]
        crc, compressed, size = struct.unpack("<III", data[pos + 16:pos + 28])
        name_len, extra_len, comment_len = struct.unpack("<HHH", data[pos + 28:pos + 34])
        offset = struct.unpack("<I", data[pos + 42:pos + 46])[0]
        name = data[pos + 46:pos + 46 + name_len].decode("utf-8")
        values = [size, compressed, offset]
        _zip64_values(data[pos + 46 + name_len:pos + 46 + name_len + extra_len], values)
        entries.append(ZipEntry(name.replace("\\", "/"), values[2], values[1], values[0],
                                method, crc))
        pos += 46 + name_len + extra_len + comment_len
    return entries


def read_central_directory(reader: RangeReader) -> list[ZipEntry]:
    """Every member record of the remote zip (two range requests)."""
    total = reader.size()
    tail = reader.read(max(0, total - _TAIL_BYTES), total - 1)
    count, cd_size, cd_offset = _end_record(tail)
    entries = _parse_central_directory(reader.read(cd_offset, cd_offset + cd_size - 1))
    if len(entries) != count:
        raise ZipRangeError(f"central directory lists {len(entries)} members, expected {count}")
    return entries


def select_members(entries: list[ZipEntry], prefixes: tuple[str, ...]) -> list[SelectedMember]:
    """The file members under ``prefixes`` (relative to the wrapper folder), in archive order."""
    wrapper = common_wrapper(e.name for e in entries)
    chosen = []
    for entry in entries:
        if entry.name.endswith("/"):
            continue
        rel = relative_to_wrapper(entry.name, wrapper)
        if matches_prefixes(rel, prefixes):
            chosen.append(SelectedMember(entry, rel))
    return sorted(chosen, key=lambda m: m.entry.offset)


def _groups(members: list[SelectedMember]) -> list[list[SelectedMember]]:
    """Members split into runs that one range request can fetch."""
    groups: list[list[SelectedMember]] = []
    current: list[SelectedMember] = []
    for member in members:
        if current:
            last = current[-1].entry
            gap = member.entry.offset - (last.offset + _LOCAL_HEADER + last.compressed)
            span = member.entry.offset + member.entry.compressed - current[0].entry.offset
            if gap > _GAP_BYTES or span > _GROUP_CAP:
                groups.append(current)
                current = []
        current.append(member)
    if current:
        groups.append(current)
    return groups


def _decode(entry: ZipEntry, raw: bytes) -> bytes:
    if entry.method == _STORED:
        body = raw
    elif entry.method == _DEFLATED:
        body = zlib.decompress(raw, -15)
    else:
        raise ZipRangeError(f"{entry.name}: unsupported compression method {entry.method}")
    if len(body) != entry.size or zlib.crc32(body) & _U32_MAX != entry.crc:
        raise ZipRangeError(f"{entry.name}: size or CRC-32 check failed")
    return body


def _member_body(entry: ZipEntry, data: bytes, base: int) -> bytes:
    at = entry.offset - base
    if data[at:at + 4] != _LOCAL_SIG:
        raise ZipRangeError(f"{entry.name}: no local file header at byte {entry.offset}")
    name_len, extra_len = struct.unpack("<HH", data[at + 26:at + 30])
    start = at + _LOCAL_HEADER + name_len + extra_len
    raw = data[start:start + entry.compressed]
    if len(raw) != entry.compressed:
        raise ZipRangeError(f"{entry.name}: member data truncated")
    return _decode(entry, raw)


def _is_complete(path: Path, entry: ZipEntry) -> bool:
    if not path.is_file() or path.stat().st_size != entry.size:
        return False
    return zlib.crc32(path.read_bytes()) & _U32_MAX == entry.crc


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    tmp.write_bytes(body)
    os.replace(tmp, path)


def _fetch_group(reader: RangeReader, group: list[SelectedMember], total: int,
                 stats: ExtractStats) -> bytes:
    start = group[0].entry.offset
    last = group[-1].entry
    end = min(total - 1, last.offset + _LOCAL_HEADER + last.compressed + _LOCAL_SLACK)
    data = reader.read(start, end)
    stats.bytes_read += len(data)
    return data


def extract_members(reader: RangeReader, members: list[SelectedMember], dest: Path,
                    progress: Callable[[ExtractStats], None] | None = None) -> ExtractStats:
    """Write ``members`` under ``dest`` (their wrapper-relative paths); verified, resumable.

    A member that fails its size/CRC check is listed in ``failed`` and not
    written; the caller decides whether that fails the download.
    """
    stats = ExtractStats()
    total = reader.size()
    targets = {m.entry.name: local_path(dest, m.rel) for m in members}
    pending = []
    for member in members:
        if _is_complete(targets[member.entry.name], member.entry):
            stats.skipped += 1
        else:
            pending.append(member)
    for group in _groups(pending):
        data = _fetch_group(reader, group, total, stats)
        for member in group:
            try:
                body = _member_body(member.entry, data, group[0].entry.offset)
            except (ZipRangeError, zlib.error):
                stats.failed.append(member.rel)
                continue
            _write(targets[member.entry.name], body)
            stats.written += 1
        if progress is not None:
            progress(stats)
    return stats
