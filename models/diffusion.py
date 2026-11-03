"""BLIP caption generation. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CaptionSentry:
    """Generate natural-language captions for diffusion manifolds using BLIP.

    The checkpoint, token budget and fallback caption are fields; the
    processor and model are loaded from ``model_id`` after construction. They
    are typed as ``Any`` because transformers 5.17's stubs for
    PreTrainedModel / ProcessorMixin do not declare full runtime signatures.
    """

    device: str = field(default_factory=_default_device)
    model_id: str = "Salesforce/blip-image-captioning-base"
    max_new_tokens: int = 50
    fallback_caption: str = "a high quality image"
    processor: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        blip_cls: Any = BlipForConditionalGeneration
        self.model = blip_cls.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, img_pil: Image.Image) -> str:
        """Generate a caption for the given image. Falls back to a default on failure."""
        try:
            inputs = self.processor(img_pil, return_tensors="pt").to(self.device)
            # 2026 Resilience: We use greedy search for speed
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("BLIP caption generation failed; using fallback caption: %s", exc)
            return self.fallback_caption
