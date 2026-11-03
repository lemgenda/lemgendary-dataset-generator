"""
LemGendary Dataset Compiler — Degradation Engine.

Phase 6 of the 2026 modernization roadmap.

Provides pure NumPy / SciPy / PIL degradation profiles for compiler-time
derived manifold synthesis and training-time on-the-fly augmentation.
"""

from __future__ import annotations

from .base import (
    CompositeProfile,
    DegradationProfile,
    DegradationStepResult,
    DynamicDegrader,
    composite,
    float_array_to_pil,
    pil_to_float_array,
)
from .blur import BoxBlur, DefocusBlur, GaussianBlur, MotionBlur
from .downsample import Downsample
from .film import ColorFade, FilmDust, FilmGrain, FilmScratches
from .haze import AtmosphericHaze
from .jpeg import JPEGCompression
from .lowlight import LowLight
from .noise import GaussianNoise, ISOCalibratedNoise, PoissonNoise, SaltPepperNoise
from .profiles import parse_profile
from .rain import RainMist, RainStreaks

__all__ = [
    # Haze & Rain
    "AtmosphericHaze",
    "BoxBlur",
    "ColorFade",
    "CompositeProfile",
    "DefocusBlur",
    # Base Engine & Types
    "DegradationProfile",
    "DegradationStepResult",
    # Resolution & Film
    "Downsample",
    "DynamicDegrader",
    "FilmDust",
    "FilmGrain",
    "FilmScratches",
    # Blur
    "GaussianBlur",
    # Noise
    "GaussianNoise",
    "ISOCalibratedNoise",
    # Compression & Lighting
    "JPEGCompression",
    "LowLight",
    "MotionBlur",
    "PoissonNoise",
    "RainMist",
    "RainStreaks",
    "SaltPepperNoise",
    "composite",
    "float_array_to_pil",
    "parse_profile",
    "pil_to_float_array",
]
