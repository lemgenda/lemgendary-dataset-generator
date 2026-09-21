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
    parse_profile,
    pil_to_float_array,
)
from .blur import BoxBlur, DefocusBlur, GaussianBlur, MotionBlur
from .downsample import Downsample
from .film import ColorFade, FilmDust, FilmGrain, FilmScratches
from .haze import AtmosphericHaze
from .jpeg import JPEGCompression
from .lowlight import LowLight
from .noise import GaussianNoise, ISOCalibratedNoise, PoissonNoise, SaltPepperNoise
from .rain import RainMist, RainStreaks

__all__ = [
    # Base Engine & Types
    "DegradationProfile",
    "DegradationStepResult",
    "CompositeProfile",
    "DynamicDegrader",
    "composite",
    "parse_profile",
    "pil_to_float_array",
    "float_array_to_pil",
    # Blur
    "GaussianBlur",
    "MotionBlur",
    "DefocusBlur",
    "BoxBlur",
    # Noise
    "GaussianNoise",
    "PoissonNoise",
    "SaltPepperNoise",
    "ISOCalibratedNoise",
    # Haze & Rain
    "AtmosphericHaze",
    "RainStreaks",
    "RainMist",
    # Compression & Lighting
    "JPEGCompression",
    "LowLight",
    # Resolution & Film
    "Downsample",
    "FilmGrain",
    "FilmScratches",
    "FilmDust",
    "ColorFade",
]
