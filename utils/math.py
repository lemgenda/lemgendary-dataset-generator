"""Distribution helpers. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

import numpy as np


def get_gaussian_probs(mean_score: float, sigma: float = 1.0) -> list[float]:
    """Convert a scalar quality score (1-10) to a 10-bin distribution."""
    x = np.arange(1, 11)
    probs = np.exp(-0.5 * ((x - mean_score) / sigma) ** 2)
    probs /= probs.sum()
    return probs.tolist()