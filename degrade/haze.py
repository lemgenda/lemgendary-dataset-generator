"""
Haze / atmospheric scatter kernels.

Single responsibility: synthesize haze by applying the atmospheric
scatter model I(x) = J(x)*t(x) + A*(1 - t(x)) with configurable beta
(scattering coefficient) and A (atmospheric light).

Phase 6.
"""

from __future__ import annotations