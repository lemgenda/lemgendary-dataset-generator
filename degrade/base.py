"""
Degradation profile protocol.

Defines the shape that every degradation kernel implements.
Composition: profiles are chained via the `Composite` helper (Phase 6)
so callers can request combinations like
    blur.gauss(sigma=2) + noise.gauss(sigma=0.05) + jpeg(quality=40)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DegradationProfile(Protocol):
    """Structural interface for a single degradation kernel.

    Implementations are stateless. Determinism is provided by the caller
    passing a seeded `random.Random` instance via the `rng` kwarg.
    """

    def __call__(self, img: np.ndarray, rng: Any) -> np.ndarray:
        """Apply the degradation to a float32 HWC array in [0, 1].

        Returns a new array; inputs are not mutated.
        """
        ...


def composite(*profiles: DegradationProfile) -> DegradationProfile:
    """Chain multiple degradation profiles into one. Phase 6 implementation."""
    raise NotImplementedError("Populated in Phase 6 of the modernization roadmap.")