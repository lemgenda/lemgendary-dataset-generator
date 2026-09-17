"""CLIP style manifold. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

from typing import Any, cast

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPManifold:
    """Zero-shot style tagging and latent extraction via CLIP.

    The processor and model are typed as ``Any`` because transformers 5.17's
    stubs for CLIPModel / ProcessorMixin do not declare full runtime signatures
    without per-call suppressions.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.device = device
        self.model_id = "openai/clip-vit-base-patch32"
        self.processor: Any = CLIPProcessor.from_pretrained(self.model_id)
        clip_cls: Any = CLIPModel
        self.model: Any = clip_cls.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

        # Standard Style Manifold for Zero-Shot Tagging
        self.styles = [
            "photo", "anime", "cg-art", "sketch", "oil-painting",
            "vector-art", "minimalist",
        ]

    @torch.no_grad()
    def extract_features(self, img_pil: Image.Image) -> torch.Tensor:
        """Extract latent vector for style clustering."""
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        raw_features = self.model.get_image_features(**inputs)
        image_features: torch.Tensor = cast(torch.Tensor, raw_features)
        # Normalize for cosine similarity / clustering stability
        return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def tag_style(self, img_pil: Image.Image) -> str:
        """Zero-shot style classification."""
        inputs = self.processor(
            text=self.styles, images=img_pil, return_tensors="pt", padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()
        return self.styles[best_idx]