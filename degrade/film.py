"""
Vintage film, physical artifact, and archive degradation kernels.

Phase 6 of the 2026 modernization roadmap.

Simulates historical analog film stock artifacts:
  - Non-linear film grain
  - Vertical physical scratches
  - Dust particles / specks
  - Color fading and contrast degradation
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def _numpy_rng(rng: random.Random) -> np.random.Generator:
    """Derive a NumPy generator from the per-sample Python RNG (consumes one draw)."""
    return np.random.default_rng(rng.randint(0, 2**31 - 1))


def _draw_scratch(out: np.ndarray, np_rng: np.random.Generator, opacity: float) -> int:
    """Draw one wobbling vertical scratch into ``out`` in place; return its start column."""
    h, w = out.shape[:2]
    x_start = np_rng.integers(0, w)
    scratch_color = 1.0 if np_rng.random() > 0.3 else 0.0  # White or dark scratch
    width = np_rng.choice([1, 2])
    wobble = np_rng.integers(-1, 2, size=h)
    x_track = np.clip(x_start + np.cumsum(wobble) // 10, 0, w - width)

    for y in range(h):
        x = x_track[y]
        out[y, x:x + width, :] = (
            out[y, x:x + width, :] * (1.0 - opacity) + scratch_color * opacity
        )
    return int(x_start)


def _stamp_speck(out: np.ndarray, np_rng: np.random.Generator) -> None:
    """Stamp one round dust speck into ``out`` in place."""
    h, w = out.shape[:2]
    cy = np_rng.integers(0, h)
    cx = np_rng.integers(0, w)
    radius = np_rng.integers(1, 4)
    dust_color = 0.1 if np_rng.random() > 0.4 else 0.9

    y_min, y_max = max(0, cy - radius), min(h, cy + radius + 1)
    x_min, x_max = max(0, cx - radius), min(w, cx + radius + 1)

    yy, xx = np.ogrid[y_min - cy:y_max - cy, x_min - cx:x_max - cx]
    mask = (xx**2 + yy**2) <= (radius**2)

    out[y_min:y_max, x_min:x_max][mask] = dust_color


class FilmGrain:
    """Simulates analog photographic film grain."""

    name = "film_grain"

    def __init__(self, intensity: float = 0.05, monochrome: bool = True) -> None:
        self.intensity = max(0.005, min(0.5, float(intensity)))
        self.monochrome = monochrome

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = img.shape[:2]
        np_rng = _numpy_rng(rng)

        # Grain is strongest in midtones, weaker in deep shadows/highlights
        midtones_weight = 4.0 * img * (1.0 - img)

        if self.monochrome:
            grain = np_rng.normal(0.0, self.intensity, size=(h, w, 1)).astype(np.float32)
        else:
            grain = np_rng.normal(0.0, self.intensity, size=(h, w, 3)).astype(np.float32)

        out = np.clip(img + grain * midtones_weight, 0.0, 1.0)

        params = {
            "intensity": round(self.intensity, 4),
            "monochrome": self.monochrome,
        }
        return out, params


class FilmScratches:
    """Simulates vertical physical film scratches from transport mechanisms."""

    name = "film_scratches"

    def __init__(self, num_scratches: int = 3, opacity: float = 0.6) -> None:
        self.num_scratches = max(1, int(num_scratches))
        self.opacity = max(0.1, min(1.0, float(opacity)))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        np_rng = _numpy_rng(rng)

        out = img.copy()
        scratch_coords = [
            _draw_scratch(out, np_rng, self.opacity) for _ in range(self.num_scratches)
        ]

        params = {
            "num_scratches": self.num_scratches,
            "opacity": round(self.opacity, 2),
            "scratch_x": scratch_coords,
        }
        return out, params


class FilmDust:
    """Simulates physical dust particles and fiber specks."""

    name = "film_dust"

    def __init__(self, num_specks: int = 15) -> None:
        self.num_specks = max(1, int(num_specks))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        np_rng = _numpy_rng(rng)

        out = img.copy()
        for _ in range(self.num_specks):
            _stamp_speck(out, np_rng)

        params = {
            "num_specks": self.num_specks,
        }
        return out, params


class ColorFade:
    """Simulates photochemical dye color fading and contrast compression."""

    name = "color_fade"

    def __init__(self, saturation: float = 0.7, contrast: float = 0.85) -> None:
        self.saturation = max(0.0, min(1.5, float(saturation)))
        self.contrast = max(0.2, min(1.5, float(contrast)))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Grayscale luminance
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        gray_3ch = gray[:, :, np.newaxis]

        # Saturation adjustment
        desat = gray_3ch + self.saturation * (img - gray_3ch)

        # Contrast adjustment around midtone 0.5
        faded = 0.5 + self.contrast * (desat - 0.5)

        out = np.clip(faded, 0.0, 1.0)
        params = {
            "saturation": round(self.saturation, 3),
            "contrast": round(self.contrast, 3),
        }
        return out, params
