"""BLIP caption generation. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class CaptionSentry:
    """Generate natural-language captions for diffusion manifolds using BLIP.

    The processor is typed as ``Any`` because transformers 5.17's stub for
    ``ProcessorMixin.__call__`` does not declare ``return_tensors`` (or
    ``padding``) even though runtime accepts both. Declaring as ``Any`` lets
    the call sites pass those kwargs without per-call suppressions.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.device = device
        self.processor: Any = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
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
        except Exception:
            return "a high quality image"  # Fallback safety descriptor