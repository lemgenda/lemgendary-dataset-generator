"""
Rain and precipitation degradation kernels.

Phase 6 of the 2026 modernization roadmap.

Pure NumPy / SciPy implementation for:
  - Linear directional rain streaks
  - Foggy rain mist / droplet scattering
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from scipy.ndimage import convolve

from .blur import create_motion_kernel


class RainStreaks:

    """Simulates physical rain streaks falling at a specified wind angle."""

    name = "rain_streaks"

    def __init__(
        self,
        density: float = 0.015,
        length: int = 25,
        angle_deg: float = 75.0,
        opacity: float = 0.65,
    ) -> None:
        self.density = max(0.001, min(0.2, float(density)))
        self.length = max(5, int(length))
        self.angle_deg = float(angle_deg)
        self.opacity = max(0.1, min(1.0, float(opacity)))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = img.shape[:2]
        seed_val = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        # Generate sparse rain drop impulse noise
        rain_noise = (np_rng.uniform(0.0, 1.0, size=(h, w)) < self.density).astype(np.float32)
        rain_brightness = np_rng.uniform(0.7, 1.0, size=(h, w)).astype(np.float32)
        streak_seeds = rain_noise * rain_brightness

        # Smear impulses into elongated streaks via directional motion kernel
        kernel = create_motion_kernel(self.length, self.angle_deg)
        streak_layer = convolve(streak_seeds, kernel, mode="wrap")

        # Normalize and enhance streak visibility
        max_v = np.max(streak_layer)
        if max_v > 0:
            streak_layer = (streak_layer / max_v) * self.opacity

        streak_layer = streak_layer[:, :, np.newaxis]

        # Additive blending with slight scene attenuation
        out = img * (1.0 - streak_layer * 0.3) + streak_layer
        out = np.clip(out, 0.0, 1.0)

        params = {
            "density": round(self.density, 4),
            "length": self.length,
            "angle_deg": round(self.angle_deg, 2),
            "opacity": round(self.opacity, 3),
        }
        return out, params


class RainMist:
    """Simulates fine mist / droplet scattering during heavy rain."""

    name = "rain_mist"

    def __init__(self, intensity: float = 0.3) -> None:
        self.intensity = max(0.05, min(1.0, float(intensity)))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        seed_val = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        mist_light = np_rng.uniform(0.85, 0.95)
        mist_map = self.intensity * 0.5
        out = img * (1.0 - mist_map) + mist_light * mist_map
        out = np.clip(out, 0.0, 1.0)

        params = {
            "intensity": round(self.intensity, 3),
        }
        return out, params
