"""Image-mode utilities. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

import numpy as np
from PIL import Image


def ensure_srgb(img: Image.Image) -> Image.Image:
    """Coerce any PIL image to RGB.

    Handles alpha (RGBA), palette (P, with or without transparency), and
    grayscale-with-alpha (LA) source modes. Returns a new PIL.Image.Image
    in RGB mode. The function does not signal whether a conversion happened;
    callers that need to know compare ``result.mode`` themselves.
    """
    if img.mode != "RGB":
        if img.mode in ("RGBA", "P", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
        img = img.convert("RGB")
    return img


def is_black_image(img: Image.Image, threshold: float = 0.1) -> bool:
    """Detect near-black frames.

    A frame is 'black' when the ratio of pixels with luminance < 10 exceeds
    ``1 - threshold``. Default threshold 0.1 → at least 90% of pixels must
    be near-black.
    """
    img_thumb = img.resize((64, 64), Image.Resampling.NEAREST) if img.size[0] > 64 else img
    grayscale = img_thumb.convert("L")
    stat = np.array(grayscale)
    black_ratio = float(np.sum(stat < 10) / stat.size)
    return (black_ratio > (1.0 - threshold))