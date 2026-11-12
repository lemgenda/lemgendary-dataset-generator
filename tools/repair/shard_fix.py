"""Rewrite a damaged WebDataset shard: no empty samples, contiguous groups, WebP only.

A shard needs a fix when it holds zero-byte members (the whole sample they
belong to is dropped, so no image is left without its pair), when a
sample's members are scattered (they are regrouped in first-seen order), or
when it still holds PNG/JPEG members (re-encoded to WebP with the
compiler's settings, see :mod:`tools.repair.member_webp`).

The fixed shard is written to a temporary file next to the original,
re-audited, and only then swapped in with ``os.replace``, which removes the
original in the same atomic step. No copy of the original is kept; a crash
before the swap leaves the original untouched and at most a stray
``.repair-tmp`` file, which the next run overwrites.
"""

from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from tools.repair.member_webp import convertible_members, to_webp
from utils.image_payload import CorruptImageError
from utils.shard_audit import ShardAudit, audit_shard, split_member_name

__all__ = ["ShardFix", "ShardFixError", "apply_shard_fix", "plan_shard_fix", "rewrite_shard"]

_TMP_SUFFIX = ".repair-tmp"


class ShardFixError(RuntimeError):
    """The rewritten shard failed its verification; the original is untouched."""


@dataclass(frozen=True)
class ShardFix:
    """What a repair would change in one shard."""

    shard: Path
    drop_keys: frozenset[str]
    regroup: bool
    samples_before: int
    convert: frozenset[str] = field(default_factory=frozenset)

    @property
    def samples_after(self) -> int:
        """Samples left once the dropped ones are gone."""
        return self.samples_before - len(self.drop_keys)


def plan_shard_fix(audit: ShardAudit) -> ShardFix | None:
    """The fix ``audit`` calls for, or None when the shard is sound (or unreadable)."""
    if audit.error:
        return None
    drop_keys = frozenset(audit.zero_byte_keys)
    convert = frozenset(name for name in convertible_members(audit)
                        if split_member_name(name)[0] not in drop_keys)
    if not audit.needs_rewrite and not convert:
        return None
    return ShardFix(
        shard=audit.path,
        drop_keys=drop_keys,
        regroup=bool(audit.scattered),
        samples_before=audit.samples,
        convert=convert,
    )


def _copy_member(src: tarfile.TarFile, dst: tarfile.TarFile, member: tarfile.TarInfo,
                 convert: frozenset[str]) -> None:
    """Copy ``member`` into ``dst``, re-encoded as WebP when it is in ``convert``."""
    handle = src.extractfile(member)
    if member.name not in convert:
        dst.addfile(member, handle)
        return
    if handle is None:
        raise ShardFixError(f"{member.name}: member has no payload")
    try:
        name, payload = to_webp(member.name, handle.read())
    except CorruptImageError as exc:
        raise ShardFixError(f"{member.name}: cannot convert to WebP ({exc})") from exc
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(payload), member.mtime, member.mode
    dst.addfile(info, io.BytesIO(payload))


def rewrite_shard(fix: ShardFix, out_path: Path) -> int:
    """Write the fixed shard to ``out_path``; return the samples written.

    Members are read by header offset, so only one payload is in memory at a
    time however far apart a sample's members were. Members named in
    ``fix.convert`` are re-encoded to WebP on the way.
    """
    written = 0
    with tarfile.open(fix.shard, "r:") as src:
        groups: dict[str, list[tarfile.TarInfo]] = {}
        for member in src.getmembers():
            if member.isfile():
                groups.setdefault(split_member_name(member.name)[0], []).append(member)
        with tarfile.open(out_path, "w") as dst:
            for key, members in groups.items():
                if key in fix.drop_keys:
                    continue
                for member in members:
                    _copy_member(src, dst, member, fix.convert)
                written += 1
    return written


def _verify(fix: ShardFix, out_path: Path) -> None:
    audit = audit_shard(out_path)
    leftover = convertible_members(audit) if not audit.error else frozenset()
    if (audit.error or audit.needs_rewrite or leftover
            or audit.samples != fix.samples_after):
        raise ShardFixError(
            f"{fix.shard.name}: rewritten shard failed verification (samples {audit.samples}"
            f" of {fix.samples_after}, zero-byte {len(audit.zero_byte)},"
            f" scattered {len(audit.scattered)}, png/jpg {len(leftover)},"
            f" error {audit.error})"
        )


def apply_shard_fix(fix: ShardFix) -> None:
    """Rewrite to a temporary file, verify it, then atomically replace the original."""
    tmp = fix.shard.with_name(fix.shard.name + _TMP_SUFFIX)
    try:
        rewrite_shard(fix, tmp)
        _verify(fix, tmp)
    except (OSError, tarfile.TarError, ShardFixError):
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, fix.shard)
