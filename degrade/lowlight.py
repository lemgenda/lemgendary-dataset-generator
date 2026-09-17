"""
Low-light and extreme underexposure degradation kernel.

Phase 6 of the 2026 modernization roadmap.

Simulates extreme sensor photon starvation:
  - Non-linear gamma curve darkening
  - Readout noise in deep shadows
  - Sensor color temperature shift / chroma desaturation
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


class LowLight:
    """Simulates realistic low-light / underexposed camera capture."""

    name = "low_light"

    def __init__(
        self,
        gamma: float = 2.2,
        color_shift: tuple[float, float, float] = (0.92, 0.95, 1.05),
        noise_scale: float = 0.02,
    ) -> None:
        self.gamma = max(1.0, float(gamma))
        self.color_shift = np.array(color_shift, dtype=np.float32)
        self.noise_scale = max(0.0, float(noise_scale))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        seed_val = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        # 1. Non-linear gamma curve darkening (photon starvation)
        darkened = np.power(np.maximum(img, 0.0), self.gamma)

        # 2. Color temperature tint in dark environment
        tinted = darkened * self.color_shift

        # 3. Readout noise visible in shadow regions
        if self.noise_scale > 0:
            shadow_mask = 1.0 - np.mean(tinted, axis=2, keepdims=True)
            readout_noise = np_rng.normal(0.0, self.noise_scale, size=img.shape).astype(np.float32)
            tinted += readout_noise * shadow_mask

        out = np.clip(tinted, 0.0, 1.0)

        params = {
            "gamma": round(self.gamma, 3),
            "color_shift": [round(float(x), 3) for x in self.color_shift],
            "noise_scale": round(self.noise_scale, 4),
        }
        return out, params
