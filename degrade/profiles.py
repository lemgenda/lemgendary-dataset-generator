"""
Degradation profile expression parser and named presets.

Turns a preset alias (``"rainy-haze"``) or a ``+``-joined token expression
(``"blur.gauss+noise.gauss+jpeg"``) into a CompositeProfile built from the
concrete kernels. Kept apart from ``degrade.base`` because the kernels
import the base protocol.
"""

from __future__ import annotations

from .base import CompositeProfile, DegradationProfile, composite
from .blur import DefocusBlur, GaussianBlur, MotionBlur
from .downsample import Downsample
from .film import ColorFade, FilmGrain, FilmScratches
from .haze import AtmosphericHaze
from .jpeg import JPEGCompression
from .lowlight import LowLight
from .noise import GaussianNoise, ISOCalibratedNoise
from .rain import RainStreaks


# ─── Profile Parser & Presets ───────────────────────────────────────────────
_INTENSITY_SCALES: dict[str, float] = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.75,
}


def _build_preset(canonical: str, scale: float) -> CompositeProfile | None:
    """Construct preset profile by canonical alias name."""
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
