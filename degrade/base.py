"""
Degradation profile protocol and composition engine.

Phase 6 of the 2026 modernization roadmap.

Provides the structural interface for degradation kernels, sequential composite
chaining, intensity-based preset configuration, and parameter logging for
downstream training supervision.

Array convention:
    Images are represented as float32 ndarrays with shape (H, W, C) in [0.0, 1.0].
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Protocol, overload, runtime_checkable

import numpy as np
from PIL import Image


@dataclass
class DegradationStepResult:
    """Outcome of a single degradation stage."""
    image: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DegradationProfile(Protocol):
    """Structural interface for a single degradation kernel.

    Implementations are stateless. Determinism is ensured by the caller
    passing a seeded random.Random instance via the `rng` kwarg.
    """

    name: str

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, Any]]: ...

    def config(self) -> dict[str, Any]:
        """The kernel's configured (not per-sample) parameters, for reproducibility."""
        ...


@dataclass
class DegradationFilter:
    """Base of the concrete kernels: the constructor arguments are dataclass fields.

    ``config()`` reports those fields (after clamping in ``__post_init__``);
    attributes derived from them are declared ``init=False`` and left out.
    """

    name: ClassVar[str] = "filter"

    def config(self) -> dict[str, Any]:
        """The configured parameters, keyed by constructor argument name."""
        return {
            f.name: list(value) if isinstance(value := getattr(self, f.name), tuple) else value
            for f in fields(self) if f.init
        }


# ─── Format Conversion Utilities ───────────────────────────────────────────
def pil_to_float_array(img: Image.Image) -> np.ndarray:
    """Convert PIL image to float32 ndarray with shape (H, W, C) in [0.0, 1.0]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def float_array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert float32 ndarray in [0.0, 1.0] to uint8 PIL RGB image."""
    clipped = np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


# ─── Composite Profile ──────────────────────────────────────────────────────
@dataclass
class CompositeProfile:
    """Chains multiple DegradationProfile instances sequentially."""

    profiles: Sequence[DegradationProfile]
    name: str = "composite"

    def __post_init__(self) -> None:
        self.profiles = list(self.profiles)

    def config(self) -> dict[str, Any]:
        """The chain's name and each kernel's type and configured parameters, in order."""
        return {
            "name": self.name,
            "steps": [{"type": p.name, "params": p.config()} for p in self.profiles],
        }

    def __call__(
        self,
        img: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        current = img
        steps: list[dict[str, Any]] = []

        for p in self.profiles:
            current, params = p(current, rng)
            steps.append({
                "type": p.name,
                "params": params,
            })

        return current, steps


def composite(*profiles: DegradationProfile, name: str = "composite") -> CompositeProfile:
    """Factory creating a sequential CompositeProfile."""
    return CompositeProfile(profiles=profiles, name=name)


# ─── Dynamic In-Memory Degrader ─────────────────────────────────────────────
class DynamicDegrader:
    """High-speed in-memory degradation runner for both PIL and NumPy inputs.

    Usable both during offline dataset compilation and online dataloader transforms.
    """

    def __init__(
        self,
        profile: DegradationProfile | CompositeProfile,
        seed: int | None = None,
    ) -> None:
        self.profile = profile
        self.seed = seed
        self._rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        """Reseed the degrader so later samples are reproducible."""
        self.seed = seed
        self._rng = random.Random(seed)

    def degrade_pil(
        self,
        img: Image.Image,
        sample_seed: int | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Apply degradation directly to PIL RGB image and return (PIL, meta)."""
        rng = random.Random(sample_seed) if sample_seed is not None else self._rng
        arr = pil_to_float_array(img)
        degraded_arr, step_info = self.profile(arr, rng)
        degradations = (
            [{"type": self.profile.name, "params": step_info}]
            if isinstance(step_info, dict)
            else step_info
        )
        meta = {
            "profile": self.profile.name,
            "seed": sample_seed if sample_seed is not None else self.seed,
            "degradations": degradations,
        }
        return float_array_to_pil(degraded_arr), meta

    def degrade_array(
        self,
        img: np.ndarray,
        sample_seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply degradation directly to float32 HWC array and return (ndarray, meta)."""
        rng = random.Random(sample_seed) if sample_seed is not None else self._rng
        arr = img.astype(np.float32)
        degraded_arr, step_info = self.profile(arr, rng)
        degradations = (
            [{"type": self.profile.name, "params": step_info}]
            if isinstance(step_info, dict)
            else step_info
        )
        meta = {
            "profile": self.profile.name,
            "seed": sample_seed if sample_seed is not None else self.seed,
            "degradations": degradations,
        }
        return degraded_arr, meta

    @overload
    def __call__(
        self,
        img: Image.Image,
        sample_seed: int | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        ...

    @overload
    def __call__(
        self,
        img: np.ndarray,
        sample_seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ...

    def __call__(
        self,
        img: Image.Image | np.ndarray,
        sample_seed: int | None = None,
    ) -> tuple[Image.Image | np.ndarray, dict[str, Any]]:
        """Degrades the input and returns (degraded_output, metadata_dict)."""
        if isinstance(img, Image.Image):
            return self.degrade_pil(img, sample_seed=sample_seed)
        return self.degrade_array(img, sample_seed=sample_seed)
