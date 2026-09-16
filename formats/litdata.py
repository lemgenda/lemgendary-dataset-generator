"""
PyTorch LitData writer.

Single responsibility: serialize Samples into LitData's on-disk format,
optimized for variable-shape inputs (detection bboxes, pose keypoints).

Requires: litdata

Phase 4. Populated only when `--also-format litdata` is passed.
"""

from __future__ import annotations

# Phase 4 will define: class LitDataWriter(Writer)