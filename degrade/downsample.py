"""
Downsampling and resolution degradation kernels.

Phase 6 of the 2026 modernization roadmap.

Simulates optical resolution loss and downsampling interpolation:
  - Bicubic downsampling & upsampling
  - Bilinear downsampling
  - Lanczos-3 downsampling
  - Nearest neighbor pixelation
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image


_PIL_RESAMPLE_MODES = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}


class Downsample:
    """Simulates spatial resolution loss via downscale-upscale cycle."""

    name = "downsample"

    def __init__(
        self,
        scale_factor: float = 0.5,
        mode: str = "bicubic",
        keep_low_res: bool = False,
    ) -> None:
        self.scale_factor = max(0.05, min(1.0, float(scale_factor)))
        self.mode = mode.lower()
        if self.mode not in _PIL_RESAMPLE_MODES:
            self.mode = "bicubic"
        self.keep_low_res = keep_low_res

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = img.shape[:2]
        small_h = max(8, int(round(h * self.scale_factor)))
        small_w = max(8, int(round(w * self.scale_factor)))

        clipped_u8 = np.clip(img * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        pil_img = Image.fromarray(clipped_u8, mode="RGB")
        resample_mode = _PIL_RESAMPLE_MODES[self.mode]

        # Downsample
        down = pil_img.resize((small_w, small_h), resample=resample_mode)

        # Upscale back to original shape unless keep_low_res is requested
        if not self.keep_low_res:
            up = down.resize((w, h), resample=resample_mode)
            out_arr = np.asarray(up, dtype=np.float32) / 255.0
        else:
            out_arr = np.asarray(down, dtype=np.float32) / 255.0

        params = {
            "scale_factor": round(self.scale_factor, 3),
            "mode": self.mode,
            "downsampled_size": [small_h, small_w],
        }
        return out_arr, params
