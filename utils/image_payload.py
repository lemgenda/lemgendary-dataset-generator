"""Validated image, target and mask payload encoding for shard writers.

Single source of truth for turning raw image bytes into WebDataset member
payloads. Every converter (loose directory, zip stream, MDS) goes through
:func:`encode_image_payload` or :func:`encode_mask_payload`, so an empty or
undecodable file is rejected with :class:`CorruptImageError` instead of being
written into a shard as a zero-byte or broken member.
"""

from __future__ import annotations

import io

from PIL import Image

from utils.container_stream import transcode_to_webp

__all__ = [
    "CorruptImageError",
    "encode_image_payload",
    "encode_mask_payload",
    "validate_image_bytes",
]

# Pillow modes whose pixel values are already class indices in 0..255.
_INDEX_MODES = ("L", "1")
# Pillow modes kept as colour-coded masks (lossless, every channel preserved).
_COLOUR_MODES = ("RGB", "RGBA")


class CorruptImageError(ValueError):
    """Raised when image bytes are empty, undecodable or not a usable mask."""


def validate_image_bytes(raw: bytes, name: str) -> None:
    """Raise :class:`CorruptImageError` unless ``raw`` is a non-empty, decodable image."""
    if not raw:
        raise CorruptImageError(f"{name}: zero-byte image")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise CorruptImageError(f"{name}: undecodable image ({exc})") from exc


def encode_image_payload(
    raw: bytes, ext: str, name: str, quality: int, transcode_webp: bool
) -> tuple[bytes, str]:
    """Validate ``raw`` and return ``(payload, extension)``, WebP-transcoded on request."""
    validate_image_bytes(raw, name)
    if not transcode_webp:
        return raw, ext.lower()
    return transcode_to_webp(raw, ext, quality=quality)


def encode_mask_payload(raw: bytes, name: str) -> tuple[bytes, str]:
    """Validate ``raw`` and return it as a lossless WebP mask preserving its values.

    Palette (``P``) masks keep their palette indices as grey levels, so class
    ids survive the round trip; ``L``/``1`` masks keep their grey levels and
    ``RGB``/``RGBA`` colour-coded masks keep every channel. Masks with more
    than 8 bits per value cannot be stored losslessly in WebP and are rejected.
    """
    validate_image_bytes(raw, name)
    with Image.open(io.BytesIO(raw)) as img:
        if img.mode == "P":
            out = Image.frombytes("L", img.size, img.tobytes())
        elif img.mode in _INDEX_MODES:
            out = img.convert("L")
        elif img.mode in _COLOUR_MODES:
            out = img.copy()
        else:
            raise CorruptImageError(
                f"{name}: mask mode {img.mode} cannot be stored losslessly as WebP"
            )
    buf = io.BytesIO()
    out.save(buf, format="WEBP", lossless=True)
    return buf.getvalue(), ".webp"
