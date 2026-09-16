"""
Label generation strategies.

Single responsibility: produce label artifacts (captions, YOLO boxes,
classification tokens, quality distributions) from raw images when no
upstream labels exist.

Strategies (planned):
    - BLIP captioning       (reuses models.diffusion.CaptionSentry)
    - YOLO auto-labeling    (reuses models.detection.AutoLabeler)
    - CLIP zero-shot class  (uses models.encoder.CLIPManifold)
    - NIMA quality score    (reuses models.quality_scorer.QualitySentry)

Phase 5. Populated when a manifold config declares `labeling: true` for a
source that ships no annotations.
"""

from __future__ import annotations