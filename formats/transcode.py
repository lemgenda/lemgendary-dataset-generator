"""
Image transcoding layer.

Phase 3 of the 2026 modernization roadmap.

Single responsibility: encode an in-memory PIL image into the byte format
dictated by an ImageFormatPolicy, and report the resulting format name so
the caller can pick a matching file extension.

The policy distinguishes three kinds:

    image   -> policy.format  (default: webp q=92)
    target  -> policy.format  (default: webp q=95)
    mask    -> policy.mask_format (default: webp-lossless)

When policy.format == "keep", the transcoder signals via KeepFormatError
that the caller must preserve the original bytes and skip the encode.
"""

from __future__ import annotations

import io
from typing import Any, Literal

from PIL import Image

from core.config_schema import ImageFormatPolicy


_FORMAT_EXT: dict[str, str] = {
    "webp": ".webp",
    "jpeg": ".jpg",
    "png": ".png",
}


class KeepFormatError(Exception):
    """Raised when the policy format is 'keep' — caller should copy the
    original bytes instead of encoding."""


class ImageTranscoder:
    """Stateless transcoder. Construct once per worker and reuse."""

    def __init__(self, policy: ImageFormatPolicy | None = None) -> None:
        self.policy = policy or ImageFormatPolicy()

    @property
    def enabled(self) -> bool:
        """True when the policy requests transcoding (anything but 'keep')."""
        return self.policy.format != "keep"

    def extension_for(self, fmt: str) -> str:
        """Map a format name to its canonical file extension."""
        return _FORMAT_EXT.get(fmt, ".bin")

    # ── Public encode entry ─────────────────────────────────────────────────
    def encode(
        self,
        img: Image.Image,
        kind: Literal["image", "target", "mask"] = "image",
    ) -> tuple[bytes, str]:
        """Encode an image. Returns (bytes, format_name).

        Raises KeepFormatError when the policy is 'keep' — the caller
        should preserve the original bytes.
        """
        if kind == "mask":
            return self._encode_mask(img)
        if kind == "target":
            return self._encode_with_quality(img, self.policy.target_quality)
        return self._encode_with_quality(img, self.policy.quality)

    # ── Internal encoders ───────────────────────────────────────────────────
    def _encode_with_quality(self, img: Image.Image, quality: int) -> tuple[bytes, str]:
        fmt = self.policy.format
        if fmt == "keep":
            raise KeepFormatError()
        if fmt == "webp":
            return self._encode_webp(img, quality=quality, lossless=False)
        if fmt == "jpeg":
            return self._encode_jpeg(img, quality=quality)
        if fmt == "png":
            return self._encode_png(img)
        # Unknown format — safe fallback to WebP.
        return self._encode_webp(img, quality=quality, lossless=False)

    def _encode_mask(self, img: Image.Image) -> tuple[bytes, str]:
        if self.policy.mask_format == "webp-lossless":
            return self._encode_webp(img, quality=100, lossless=True)
        return self._encode_png(img)

    @staticmethod
    def _encode_webp(img: Image.Image, quality: int, lossless: bool) -> tuple[bytes, str]:
        buf = io.BytesIO()
        kwargs: dict[str, Any] = {"format": "WEBP", "method": 4}
        if lossless:
            kwargs["lossless"] = True
        else:
            kwargs["quality"] = quality
        img.save(buf, **kwargs)
        return buf.getvalue(), "webp"

    def _encode_jpeg(self, img: Image.Image, quality: int) -> tuple[bytes, str]:
        buf = io.BytesIO()
        if img.mode in ("RGBA", "LA", "P"):
            img = self._flatten_alpha(img)
        else:
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), "jpeg"

    @staticmethod
    def _encode_png(img: Image.Image) -> tuple[bytes, str]:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "png"

    @staticmethod
    def _flatten_alpha(img: Image.Image) -> Image.Image:
        """Composite an alpha-carrying image onto white. Returns RGB."""
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode == "LA":
            img = img.convert("RGBA")
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            return background
        return img.convert("RGB")