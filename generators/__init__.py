"""
LemGendary Dataset Compiler — Smart Generation Layer.

Phase 5 of the 2026 modernization roadmap.

Public surface:
    GenerationResult      — result dataclass
    Generator             — Protocol
    LabelGenerator        — labels (BLIP / CLIP / YOLO / ParseNet / NIMA)
    PromptGenerator       — structured diffusion prompts
    MaskGenerator         — segmentation masks (ParseNet / SAM / MODNet)

The compiler instantiates one generator per worker based on CONFIG task
inference. All three modules compose the existing worker-global model
wrappers — no new model loading.
"""

from __future__ import annotations

__all__ = [
    "GenerationResult",
    "Generator",
    "LabelGenerator",
    "PromptGenerator",
    "MaskGenerator",
]

from .base import GenerationResult, Generator
from .labels import LabelGenerator
from .prompts import PromptGenerator
from .masks import MaskGenerator