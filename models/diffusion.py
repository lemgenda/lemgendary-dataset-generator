"""BLIP caption generation. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

from typing import Any
import logging

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)


class CaptionSentry:
    """Generate natural-language captions for diffusion manifolds using BLIP.

    The processor and model are typed as ``Any`` because transformers 5.17's
    stubs for PreTrainedModel / ProcessorMixin do not declare full runtime
    signatures without per-call suppressions.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.device = device
        self.processor: Any = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_cls: Any = BlipForConditionalGeneration
        self.model: Any = blip_cls.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, img_pil: Image.Image) -> str:
        """Generate a caption for the given image. Falls back to a default on failure."""
        try:
            inputs = self.processor(img_pil, return_tensors="pt").to(self.device)
            # 2026 Resilience: We use greedy search for speed
            out = self.model.generate(**inputs, max_new_tokens=50)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as exc:
            logger.debug("BLIP caption generation fallback invoked: %s", exc)
            return "a high quality image"  # Fallback safety descriptor