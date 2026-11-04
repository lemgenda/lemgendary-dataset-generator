"""
Blur degradation kernels.

Phase 6 of the 2026 modernization roadmap.

Pure NumPy / SciPy implementations for:
  - Gaussian blur
  - Linear motion blur (directional)
  - Defocus blur (disk aperture)
  - Box blur
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import convolve

from .base import DegradationFilter


def _create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generate 2D normalized Gaussian kernel."""
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def create_motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    """Generate 2D normalized linear motion blur kernel."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    for i in range(size):
        offset = i - center
        x = round(center + offset * cos_a)
        y = round(center + offset * sin_a)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    ksum = np.sum(kernel)
    if ksum == 0:
        kernel[center, center] = 1.0
        return kernel
    return kernel / ksum


_create_motion_kernel = create_motion_kernel


def _create_disk_kernel(radius: int) -> np.ndarray:
    """Generate 2D normalized circular disk defocus kernel."""
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    mask = (xx**2 + yy**2) <= (radius**2)
    kernel = mask.astype(np.float32)
    return kernel / np.sum(kernel)


def _gaussian_kernel_size(sigma: float, kernel_size: int | None) -> int:
    """Odd kernel size: derived from ``sigma`` when unset, else rounded up to odd."""
    if kernel_size is None:
        return max(3, math.ceil(sigma * 3.0) * 2 + 1)
    return kernel_size + 1 - kernel_size % 2


@dataclass
class GaussianBlur(DegradationFilter):
    """Applies isotropic Gaussian blur."""

    name = "gaussian_blur"

    sigma: float = 1.8
    kernel_size: int | None = None

    def __post_init__(self) -> None:
        self.sigma = max(0.1, float(self.sigma))
        self.kernel_size = _gaussian_kernel_size(self.sigma, self.kernel_size)

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        size = _gaussian_kernel_size(self.sigma, self.kernel_size)
        kernel = _create_gaussian_kernel(size, self.sigma)
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = convolve(img[:, :, c], kernel, mode="reflect")

        params = {
            "sigma": round(self.sigma, 4),
            "kernel_size": size,
        }
        return np.clip(out, 0.0, 1.0), params


@dataclass
class MotionBlur(DegradationFilter):
    """Applies directional linear motion blur."""

    name = "motion_blur"

    kernel_size: int = 15
    angle: float | None = None

    def __post_init__(self) -> None:
        self.kernel_size = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1
        self.angle = self.angle

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        angle = self.angle if self.angle is not None else rng.uniform(0.0, 360.0)
        kernel = _create_motion_kernel(self.kernel_size, angle)

        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = convolve(img[:, :, c], kernel, mode="reflect")

        params = {
            "kernel_size": self.kernel_size,
            "angle_deg": round(angle, 2),
        }
        return np.clip(out, 0.0, 1.0), params


@dataclass
class DefocusBlur(DegradationFilter):
    """Applies circular disk defocus blur."""

    name = "defocus_blur"

    radius: int = 3

    def __post_init__(self) -> None:
        self.radius = max(1, int(self.radius))

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        kernel = _create_disk_kernel(self.radius)
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = convolve(img[:, :, c], kernel, mode="reflect")

        params = {
            "radius": self.radius,
            "kernel_size": self.radius * 2 + 1,
        }
        return np.clip(out, 0.0, 1.0), params


@dataclass
class BoxBlur(DegradationFilter):
    """Applies uniform box blur."""

    name = "box_blur"

    kernel_size: int = 5

    def __post_init__(self) -> None:
        self.kernel_size = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        k = np.ones((self.kernel_size, self.kernel_size), dtype=np.float32) / (self.kernel_size**2)
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = convolve(img[:, :, c], k, mode="reflect")

        params = {
            "kernel_size": self.kernel_size,
        }
        return np.clip(out, 0.0, 1.0), params
