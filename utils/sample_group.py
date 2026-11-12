"""One WebDataset sample: its raw members, their encoding and contiguous writing.

A sample groups every member that shares a key: the primary image, an
optional restoration target, an optional segmentation mask and an optional
label sidecar. Converters build a :class:`RawSample`, encode it with
:func:`encode_sample` (which rejects corrupt images) and write the result
with :func:`write_sample`, which always places the whole group contiguously
inside the tar shard so sequential WebDataset readers never seek.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from typing import Any

from utils.container_stream import (
    IMAGE_WEBP_QUALITY,
    TARGET_WEBP_QUALITY,
    loose_mds_record,
)
from utils.image_payload import (
    CorruptImageError,
    encode_image_payload,
    encode_mask_payload,
)

__all__ = [
    "LABEL_EXTS",
    "MASK_SUFFIX",
    "TARGET_SUFFIX",
    "EncodedSample",
    "RawMember",
    "RawSample",
    "SampleOutcome",
    "add_tar_member",
    "encode_sample",
    "encode_sample_or_reject",
    "sample_member_names",
    "to_mds_record",
    "write_sample",
]

# Label sidecar extensions packed verbatim next to the image (YOLO txt, JSON).
LABEL_EXTS: tuple[str, ...] = (".txt", ".json")
TARGET_SUFFIX = ".target"
MASK_SUFFIX = ".mask"


@dataclass(frozen=True)
class RawMember:
    """Raw bytes of one member and its original lower-case extension."""

    payload: bytes
    ext: str


@dataclass(frozen=True)
class RawSample:
    """Unencoded members of one sample; ``primary`` is the image (or lone target)."""

    key: str
    primary: RawMember
    target: RawMember | None = None
    mask: RawMember | None = None
    label: RawMember | None = None
    primary_quality: int = IMAGE_WEBP_QUALITY


@dataclass(frozen=True)
class EncodedSample:
    """Validated, encoded members of one sample, in tar write order."""

    key: str
    members: tuple[tuple[str, bytes], ...]

    def payload(self, suffix: str) -> bytes:
        """Return the payload of the member whose name is ``key + suffix + ext``, or b''."""
        prefix = f"{self.key}{suffix}."
        for name, data in self.members:
            if name.startswith(prefix) and "." not in name[len(prefix):]:
                return data
        return b""

    def member(self, name: str) -> bytes:
        """Return the payload of the member called exactly ``name``, or b''."""
        for member_name, data in self.members:
            if member_name == name:
                return data
        return b""


@dataclass(frozen=True)
class SampleOutcome:
    """Result of encoding one sample: the encoded group or a rejection reason."""

    key: str
    sample: EncodedSample | None
    reason: str = ""


def encode_sample(raw: RawSample, transcode_webp: bool) -> EncodedSample:
    """Encode every member of ``raw``; raise :class:`CorruptImageError` on any bad image."""
    data, ext = encode_image_payload(
        raw.primary.payload, raw.primary.ext, raw.key, raw.primary_quality, transcode_webp
    )
    members: list[tuple[str, bytes]] = [(f"{raw.key}{ext}", data)]
    if raw.target is not None:
        t_data, t_ext = encode_image_payload(
            raw.target.payload,
            raw.target.ext,
            f"{raw.key}{TARGET_SUFFIX}",
            TARGET_WEBP_QUALITY,
            transcode_webp,
        )
        members.append((f"{raw.key}{TARGET_SUFFIX}{t_ext}", t_data))
    if raw.mask is not None:
        m_data, m_ext = encode_mask_payload(raw.mask.payload, f"{raw.key}{MASK_SUFFIX}")
        members.append((f"{raw.key}{MASK_SUFFIX}{m_ext}", m_data))
    if raw.label is not None:
        if not raw.label.payload:
            raise CorruptImageError(f"{raw.key}{raw.label.ext}: zero-byte label sidecar")
        try:
            raw.label.payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CorruptImageError(f"{raw.key}{raw.label.ext}: label is not UTF-8") from exc
        members.append((f"{raw.key}{raw.label.ext}", raw.label.payload))
    return EncodedSample(key=raw.key, members=tuple(members))


def encode_sample_or_reject(raw: RawSample, transcode_webp: bool) -> SampleOutcome:
    """Encode ``raw``; a corrupt member yields an outcome carrying the reason instead."""
    try:
        return SampleOutcome(raw.key, encode_sample(raw, transcode_webp))
    except CorruptImageError as exc:
        return SampleOutcome(raw.key, None, str(exc))


def sample_member_names(sample: EncodedSample) -> list[str]:
    """Return the tar member names of ``sample`` in write order."""
    return [name for name, _ in sample.members]


def add_tar_member(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    """Append one regular-file member ``name`` holding ``payload`` to ``tf``."""
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def write_sample(tf: tarfile.TarFile, sample: EncodedSample) -> None:
    """Write every member of ``sample`` back to back (key distance 1)."""
    for name, data in sample.members:
        add_tar_member(tf, name, data)


def to_mds_record(split: str, sample: EncodedSample) -> dict[str, Any]:
    """Build the loose MDS row of ``sample`` (image, target, mask, label text)."""
    record = loose_mds_record(
        split, sample.key, sample.members[0][1], sample.payload(TARGET_SUFFIX)
    )
    record["mask"] = sample.payload(MASK_SUFFIX)
    for ext in LABEL_EXTS:
        label = sample.member(f"{sample.key}{ext}")
        if label:
            record["label"] = label.decode("utf-8")
            break
    return record
