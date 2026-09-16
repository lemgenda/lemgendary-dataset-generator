"""
Mask generation strategies.

Single responsibility: produce binary or multi-class segmentation masks
from images when the source manifold has no mask pairs.

Strategies (planned):
    - ParseNet (face parsing)      (reuses models.detection.AutoLabeler in seg mode)
    - SAM v2 (generic objects)     (new dependency, opt-in)
    - MODNet (portrait matting)    (new dependency, opt-in)

Phase 5. Populated when a segmentation manifold config declares
`mask_strategy: <name>`.
"""

from __future__ import annotations