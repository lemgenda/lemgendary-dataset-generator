"""
Noise degradation kernels.

Phase 6 of the 2026 modernization roadmap.

Pure NumPy implementations for:
  - Additive Gaussian noise
  - Poisson (photon shot) noise
  - Salt-and-pepper impulse noise
  - ISO-calibrated camera sensor noise (Poisson photon + Gaussian readout)
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


class GaussianNoise:
    """Applies additive zero-mean Gaussian noise."""

    name = "gaussian_noise"

    def __init__(self, sigma: float = 0.05, mean: float = 0.0) -> None:
        self.sigma = float(sigma)
        self.mean = float(mean)

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Use seeded NumPy generator from rng integer seed
        seed_val = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        noise = np_rng.normal(self.mean, self.sigma, size=img.shape).astype(np.float32)
        out = np.clip(img + noise, 0.0, 1.0)

        params = {
            "sigma": round(self.sigma, 4),
            "mean": round(self.mean, 4),
        }
        return out, params


class PoissonNoise:
    """Simulates photon shot noise via Poisson scaling."""

    name = "poisson_noise"

    def __init__(self, scale: float = 255.0) -> None:
        self.scale = max(1.0, float(scale))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        seed_val = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        # Poisson distribution operates on photon counts
        scaled = np.clip(img * self.scale, 0.0, None)
        noisy = np_rng.poisson(scaled).astype(np.float32) / self.scale
        out = np.clip(noisy, 0.0, 1.0)

        params = {
            "scale": round(self.scale, 2),
        }
        return out, params


class SaltPepperNoise:
    """Applies salt-and-pepper impulse noise."""

    name = "salt_pepper_noise"

    def __init__(self, amount: float = 0.02, salt_vs_pepper: float = 0.5) -> None:
        self.amount = max(0.0, min(1.0, float(amount)))
        self.salt_vs_pepper = max(0.0, min(1.0, float(salt_vs_pepper)))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        seed_val = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        out = img.copy()
        h, w = img.shape[:2]
        num_pixels = h * w

        # Salt (white pixels)
        num_salt = int(num_pixels * self.amount * self.salt_vs_pepper)
        if num_salt > 0:
            coords_y = np_rng.integers(0, h, size=num_salt)
            coords_x = np_rng.integers(0, w, size=num_salt)
            out[coords_y, coords_x, :] = 1.0

        # Pepper (black pixels)
        num_pepper = int(num_pixels * self.amount * (1.0 - self.salt_vs_pepper))
        if num_pepper > 0:
            coords_y = np_rng.integers(0, h, size=num_pepper)
            coords_x = np_rng.integers(0, w, size=num_pepper)
            out[coords_y, coords_x, :] = 0.0

        params = {
            "amount": round(self.amount, 4),
            "salt_vs_pepper": round(self.salt_vs_pepper, 2),
        }
        return out, params


class ISOCalibratedNoise:
    """Simulates calibrated camera sensor noise model (heteroscedastic Gaussian + Poisson)."""

    name = "iso_noise"

    def __init__(self, iso_level: int = 1600) -> None:
        self.iso_level = max(100, int(iso_level))
        # ISO scaling factors for read noise (sigma_read) and shot noise (sigma_shot)
        iso_norm = self.iso_level / 100.0
        self.sigma_read = 0.002 * (iso_norm ** 0.6)
        self.sigma_shot = 0.008 * (iso_norm ** 0.5)

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        seed_val = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed_val)

        # Signal-dependent variance: Var(I) = sigma_shot^2 * I + sigma_read^2
        variance = (self.sigma_shot**2) * np.maximum(img, 0.0) + (self.sigma_read**2)
        std_dev = np.sqrt(variance).astype(np.float32)

        noise = np_rng.normal(0.0, 1.0, size=img.shape).astype(np.float32) * std_dev
        out = np.clip(img + noise, 0.0, 1.0)

        params = {
            "iso_level": self.iso_level,
            "sigma_read": round(float(self.sigma_read), 5),
            "sigma_shot": round(float(self.sigma_shot), 5),
        }
        return out, params
