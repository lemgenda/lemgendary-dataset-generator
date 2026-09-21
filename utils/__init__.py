"""
LemGendary Dataset Compiler — Pure utility helpers.

Phase 1.5.5 of the 2026 modernization roadmap. Submodules:

    utils.fs         — filesystem helpers
    utils.geometry   — bbox / keypoint math
    utils.hashing    — content hashing
    utils.image      — image-mode utilities
    utils.math       — distribution helpers
    utils.naming     — slug / category mapping
    utils.net        — HTTP helpers
"""
from __future__ import annotations

from . import archive, fs, geometry, hashing, image, math, naming, net

__all__ = ["archive", "fs", "geometry", "hashing", "image", "math", "naming", "net"]