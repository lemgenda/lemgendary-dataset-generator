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
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, overload, runtime_checkable

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
class CompositeProfile:
    """Chains multiple DegradationProfile instances sequentially."""

    def __init__(
        self,
        profiles: Sequence[DegradationProfile],
        name: str = "composite",
    ) -> None:
        self.profiles = list(profiles)
        self.name = name

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


# ─── Profile Parser & Presets ───────────────────────────────────────────────
_INTENSITY_SCALES: dict[str, float] = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.75,
}


def _build_preset(canonical: str, scale: float) -> CompositeProfile | None:
    """Construct preset profile by canonical alias name."""
    from .blur import DefocusBlur, GaussianBlur, MotionBlur
    from .downsample import Downsample
    from .film import ColorFade, FilmGrain, FilmScratches
    from .haze import AtmosphericHaze
    from .jpeg import JPEGCompression
    from .lowlight import LowLight
    from .noise import ISOCalibratedNoise
    from .rain import RainStreaks

    k_blur = int(round(15 * scale))
    k_blur = k_blur + 1 if k_blur % 2 == 0 else k_blur
    k_gauss = int(round(9 * scale))
    k_gauss = k_gauss + 1 if k_gauss % 2 == 0 else k_gauss

    presets: dict[str, CompositeProfile] = {
        "motion-blur+iso-noise": composite(
            MotionBlur(kernel_size=max(3, k_blur), angle=45.0),
            ISOCalibratedNoise(iso_level=max(100, int(round(1600 * scale)))),
            name="motion-blur+iso-noise",
        ),
        "lowlight-noise": composite(
            LowLight(gamma=1.0 + (1.2 * scale), noise_scale=0.03 * scale),
            ISOCalibratedNoise(iso_level=int(round(3200 * scale))),
            name="lowlight-noise",
        ),
        "rainy-haze": composite(
            AtmosphericHaze(beta=0.6 * scale),
            RainStreaks(density=0.015 * scale, length=int(round(25 * scale))),
            name="rainy-haze",
        ),
        "vintage-film": composite(
            FilmGrain(intensity=0.06 * scale),
            FilmScratches(num_scratches=int(round(4 * scale))),
            ColorFade(saturation=max(0.2, 1.0 - 0.4 * scale)),
            name="vintage-film",
        ),
        "compression-artifacts": composite(
            JPEGCompression(quality=max(10, min(95, int(round(90 - 45 * scale))))),
            name="compression-artifacts",
        ),
        "super-resolution-x4": composite(
            Downsample(scale_factor=0.25, mode="bicubic"),
            name="super-resolution-x4",
        ),
        "full-spectrum-restoration": composite(
            GaussianBlur(sigma=1.5 * scale, kernel_size=max(3, k_gauss)),
            ISOCalibratedNoise(iso_level=int(round(1200 * scale))),
            JPEGCompression(quality=max(15, min(90, int(round(85 - 35 * scale))))),
            name="full-spectrum-restoration",
        ),
    }
    return presets.get(canonical)



def parse_profile(expr: str, intensity: str = "medium") -> CompositeProfile:
    """Parse a profile expression or preset alias into a CompositeProfile.

    Supported Preset Aliases:
        - 'motion-blur+iso-noise'
        - 'lowlight-noise'
        - 'rainy-haze'
        - 'vintage-film'
        - 'compression-artifacts'
        - 'super-resolution-x4'
        - 'full-spectrum-restoration'
    """
    from .blur import DefocusBlur, GaussianBlur, MotionBlur
    from .downsample import Downsample
    from .film import FilmGrain
    from .haze import AtmosphericHaze
    from .jpeg import JPEGCompression
    from .lowlight import LowLight
    from .noise import GaussianNoise, ISOCalibratedNoise
    from .rain import RainStreaks

    scale = _INTENSITY_SCALES.get(intensity.lower(), 1.0)
    canonical = expr.strip().lower().replace(" ", "")

    preset = _build_preset(canonical, scale)
    if preset is not None:
        return preset

    # Component-wise parsing fallback e.g. "blur.gauss+noise.gauss+jpeg"
    profiles: list[DegradationProfile] = []
    for token in canonical.split("+"):
        if token.startswith("blur.gauss") or token == "blur":
            profiles.append(GaussianBlur(sigma=1.8 * scale))
        elif token.startswith("blur.motion"):
            profiles.append(MotionBlur(kernel_size=max(3, int(round(13 * scale)))))
        elif token.startswith("blur.defocus"):
            profiles.append(DefocusBlur(radius=max(1, int(round(3 * scale)))))
        elif token.startswith("noise.gauss") or token == "noise":
            profiles.append(GaussianNoise(sigma=0.04 * scale))
        elif token.startswith("noise.iso"):
            profiles.append(ISOCalibratedNoise(iso_level=int(round(1600 * scale))))
        elif token.startswith("jpeg"):
            q = max(10, min(95, int(round(90 - 40 * scale))))
            profiles.append(JPEGCompression(quality=q))
        elif token.startswith("haze"):
            profiles.append(AtmosphericHaze(beta=0.5 * scale))
        elif token.startswith("rain"):
            profiles.append(RainStreaks(density=0.015 * scale))
        elif token.startswith("lowlight"):
            profiles.append(LowLight(gamma=1.0 + 1.0 * scale))
        elif token.startswith("downsample"):
            profiles.append(Downsample(scale_factor=0.5))
        elif token.startswith("film"):
            profiles.append(FilmGrain(intensity=0.05 * scale))
        else:
            raise ValueError(f"Unknown degradation profile token or preset: {token!r}")

    return CompositeProfile(profiles=profiles, name=expr)
