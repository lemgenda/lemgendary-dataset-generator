"""
Low-light degradation.

Single responsibility: apply gamma + Poisson noise + readout noise to
simulate underexposed capture. Not a simple brightness scalar — the
photon-starvation model is the point.

Phase 6.
"""

from __future__ import annotations