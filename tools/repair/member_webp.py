"""Convert leftover PNG/JPEG members of a WebDataset shard to WebP.

Older compiles left a few raw ``.png``/``.jpg`` members in otherwise WebP
shards. A repair re-encodes them with the compiler's own payload encoders
(:mod:`utils.image_payload`) and qualities (:mod:`utils.container_stream`),
so a converted member is what a fresh compile would write:

- ``<key>.mask.<ext>``   -> lossless WebP, class values preserved;
- ``<key>.target.<ext>`` -> WebP at ``TARGET_WEBP_QUALITY``;
- any other image        -> WebP at ``IMAGE_WEBP_QUALITY``.

A member whose WebP name is already taken in its sample is left alone, so
no tar ever holds two members with the same name.
"""

from __future__ import annotations

from utils.container_stream import IMAGE_WEBP_QUALITY, TARGET_WEBP_QUALITY
from utils.image_payload import encode_image_payload, encode_mask_payload
from utils.sample_group import MASK_SUFFIX, TARGET_SUFFIX
from utils.shard_audit import ShardAudit

__all__ = ["LEGACY_IMAGE_EXTS", "WEBP_EXT", "convertible_members", "to_webp", "webp_name"]

LEGACY_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg")
WEBP_EXT = ".webp"


def _ext(name: str) -> str:
    base = name.rpartition("/")[2]
    dot = base.rfind(".")
    return base[dot:].lower() if dot > 0 else ""


def webp_name(name: str) -> str:
    """``"a.target.webp"`` for ``"a.target.png"``: the last extension becomes ``.webp``."""
    ext = _ext(name)
    return f"{name[:len(name) - len(ext)]}{WEBP_EXT}" if ext else name


def convertible_members(audit: ShardAudit) -> frozenset[str]:
    """PNG/JPEG member names of ``audit`` whose WebP name is free within their sample."""
    out: set[str] = set()
    for key, suffixes in audit.suffixes.items():
        names = {f"{key}{suffix}" for suffix in suffixes}
        for name in names:
            if _ext(name) in LEGACY_IMAGE_EXTS and webp_name(name) not in names:
                out.add(name)
    return frozenset(out)


def to_webp(name: str, payload: bytes) -> tuple[str, bytes]:
    """Re-encode member ``name`` as WebP; return the new name and payload.

    Raises :class:`utils.image_payload.CorruptImageError` when the payload is
    empty or undecodable.
    """
    base = name.rpartition("/")[2]
    if f"{MASK_SUFFIX}." in base:
        data, _ = encode_mask_payload(payload, name)
    else:
        quality = TARGET_WEBP_QUALITY if f"{TARGET_SUFFIX}." in base else IMAGE_WEBP_QUALITY
        data, _ = encode_image_payload(payload, _ext(name), name, quality, transcode_webp=True)
    return webp_name(name), data
