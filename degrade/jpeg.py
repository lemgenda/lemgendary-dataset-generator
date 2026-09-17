"""
JPEG compression degradation kernel.

Phase 6 of the 2026 modernization roadmap.

Simulates discrete cosine transform (DCT) 8x8 block quantization artifacts
and chroma subsampling via in-memory encoding buffers.
"""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from PIL import Image


class JPEGCompression:
    """Simulates realistic lossy JPEG compression artifacts."""

    name = "jpeg_compression"

    def __init__(self, quality: int = 40) -> None:
        self.quality = max(5, min(95, int(quality)))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Convert float32 HWC in [0, 1] to uint8 PIL Image
        clipped_u8 = np.clip(img * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        pil_img = Image.fromarray(clipped_u8, mode="RGB")

        # Encode to JPEG buffer and decode back
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self.quality, subsampling=2)
        buf.seek(0)
        decoded = Image.open(buf)
        out_arr = np.asarray(decoded, dtype=np.float32) / 255.0

        params = {
            "quality": self.quality,
            "subsampling": "4:2:0",
        }
        return out_arr, params
