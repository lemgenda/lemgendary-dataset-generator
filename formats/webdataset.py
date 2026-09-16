"""
WebDataset TarWriter.

Single responsibility: write Samples to WebDataset `.tar` shards for
diffusion manifolds.

The existing `ShardWriter` class in `compiler_core.py` is the pre-Phase-4
implementation; Phase 4 moves it here.
"""

from __future__ import annotations

# Phase 4 will move ShardWriter from compiler_core into this module