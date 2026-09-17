"""
Atmospheric haze degradation kernel.

Phase 6 of the 2026 modernization roadmap.

Simulates atmospheric scattering based on the physical dark channel model:
    I(x) = J(x) * t(x) + A * (1 - t(x))
where J(x) is clean scene radiance, t(x) is the medium transmission map,
and A is atmospheric global airlight.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


class AtmosphericHaze:
    """Simulates realistic atmospheric haze and depth-dependent scattering."""

    name = "atmospheric_haze"

    def __init__(
        self,
        beta: float = 0.6,
        atmospheric_light: tuple[float, float, float] = (0.88, 0.90, 0.92),
    ) -> None:
        self.beta = max(0.05, float(beta))
        self.airlight = np.array(atmospheric_light, dtype=np.float32)

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = img.shape[:2]
        seed_val = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        # Generate synthetic depth gradient: top (sky/distant) to bottom (foreground)
        # with smooth low-frequency spatial variation
        grid = np.mgrid[0:h, 0:w]
        yy = grid[0].astype(np.float32)
        xx = grid[1].astype(np.float32)
        base_depth = 1.0 - (yy / max(1.0, float(h))) * 0.7  # distant at top (1.0), near at bottom (0.3)

        # Subtle spatial perturbation to depth map
        noise_phase_x = np_rng.uniform(0.0, 2.0 * np.pi)
        noise_phase_y = np_rng.uniform(0.0, 2.0 * np.pi)
        depth_perturbation = 0.15 * np.sin(xx / max(1.0, float(w)) * 3.0 + noise_phase_x) * \
                             np.cos(yy / max(1.0, float(h)) * 3.0 + noise_phase_y)
        depth = np.clip(base_depth + depth_perturbation, 0.1, 1.5)

        # Transmission map: t(x) = exp(-beta * depth)
        tx = np.exp(-self.beta * depth)[:, :, np.newaxis]

        # Atmospheric airlight blending
        hazed = img * tx + self.airlight * (1.0 - tx)
        out = np.clip(hazed, 0.0, 1.0)

        params = {
            "beta": round(self.beta, 4),
            "airlight_mean": round(float(np.mean(self.airlight)), 3),
        }
        return out, params
