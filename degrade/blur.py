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
from typing import Any

import numpy as np
from scipy.ndimage import convolve


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
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
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
    size = radius * 2 + 1
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    mask = (xx**2 + yy**2) <= (radius**2)
    kernel = mask.astype(np.float32)
    return kernel / np.sum(kernel)


class GaussianBlur:
    """Applies isotropic Gaussian blur."""

    name = "gaussian_blur"

    def __init__(self, sigma: float = 1.8, kernel_size: int | None = None) -> None:
        self.sigma = max(0.1, float(sigma))
        if kernel_size is None:
            k = int(math.ceil(self.sigma * 3.0)) * 2 + 1
            self.kernel_size = max(3, k)
        else:
            self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        kernel = _create_gaussian_kernel(self.kernel_size, self.sigma)
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = convolve(img[:, :, c], kernel, mode="reflect")

        params = {
            "sigma": round(self.sigma, 4),
            "kernel_size": self.kernel_size,
        }
        return np.clip(out, 0.0, 1.0), params


class MotionBlur:
    """Applies directional linear motion blur."""

    name = "motion_blur"

    def __init__(self, kernel_size: int = 15, angle: float | None = None) -> None:
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.angle = angle

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


class DefocusBlur:
    """Applies circular disk defocus blur."""

    name = "defocus_blur"

    def __init__(self, radius: int = 3) -> None:
        self.radius = max(1, int(radius))

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


class BoxBlur:
    """Applies uniform box blur."""

    name = "box_blur"

    def __init__(self, kernel_size: int = 5) -> None:
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

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
