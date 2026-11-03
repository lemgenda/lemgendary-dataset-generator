"""
Prompt generation for diffusion manifolds.

Phase 5 of the 2026 modernization roadmap.

Composes the image's BLIP caption and CLIP style tag into a structured
prompt string. Templates are additive — new templates can be registered
in `_TEMPLATES` without touching the class.

The generator is designed for standalone CLI use (`generate_cli.py
--kind prompt`) rather than in-compile generation, since diffusion
manifolds typically build captions during compile and refine prompts
later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any

from PIL import Image

from .base import GenerationResult, Generator
from .model_handles import ModelHandles


_TEMPLATES: dict[str, str] = {
    "diffusers-v1": "{subject}, {style}, {lighting}, {camera}, {quality_tokens}",
    "sd-v1": "{subject}, {style}, {quality_tokens}",
    "flux-v1": (
        "{subject}. Style: {style}. Lighting: {lighting}. Camera: {camera}. {quality_tokens}"
    ),
    "minimal": "{subject}",
}

_DEFAULT_QUALITY_TOKENS = "masterpiece, best quality, highly detailed, 8k"
_DEFAULT_LIGHTING = "natural lighting"
_DEFAULT_CAMERA = "medium shot"


@dataclass(frozen=True)
class PromptStyle:
    """Fixed prompt fragments; None/empty falls back to the module defaults."""

    quality_tokens: str | None = None
    lighting: str | None = None
    camera: str | None = None


_STYLE_KEYS: tuple[str, ...] = tuple(f.name for f in fields(PromptStyle))
_MODEL_KEYS: tuple[str, ...] = ("captioner", "clip")


class PromptGenerator(Generator):
    """Composes captions + style tags into structured prompt strings."""

    def __init__(
        self,
        template: str = "diffusers-v1",
        **options: Any,
    ) -> None:
        """Select a template and bind optional models / style fragments.

        ``options`` accepts the model keywords ``captioner`` and ``clip`` and
        the style keywords ``quality_tokens``, ``lighting`` and ``camera``
        (see ``PromptStyle``); any other keyword raises ``TypeError``.
        """
        if template not in _TEMPLATES:
            raise ValueError(
                f"Unknown prompt template: {template!r}. "
                f"Valid: {sorted(_TEMPLATES)}"
            )
        style = PromptStyle(**{k: options.pop(k) for k in _STYLE_KEYS if k in options})
        handles = ModelHandles.from_kwargs(type(self).__name__, options, _MODEL_KEYS)
        self.strategy = f"prompt:{template}"
        self.template = _TEMPLATES[template]
        self._captioner = handles.captioner
        self._clip = handles.clip
        self._quality_tokens = style.quality_tokens or _DEFAULT_QUALITY_TOKENS
        self._lighting = style.lighting or _DEFAULT_LIGHTING
        self._camera = style.camera or _DEFAULT_CAMERA

    def generate(
        self,
        img: Image.Image,
        context: dict[str, Any] | None = None,
    ) -> GenerationResult:
        ctx = context or {}

        # Prefer an existing caption from the context over a fresh BLIP run.
        subject = ctx.get("caption")
        if not subject and self._captioner is not None:
            subject = self._captioner.generate(img)
        if not subject:
            subject = "an image"

        style = ctx.get("style_tag")
        if not style and self._clip is not None:
            style = self._clip.tag_style(img)
        if not style:
            style = "photo"

        subject = _strip_trailing_punct(subject)

        prompt = self.template.format(
            subject=subject,
            style=style,
            lighting=self._lighting,
            camera=self._camera,
            quality_tokens=self._quality_tokens,
        )
        prompt = re.sub(r"\s*,\s*", ", ", prompt).strip(" ,")

        return GenerationResult(
            kind="prompt",
            value=prompt,
            strategy=self.strategy,
            metadata={"template": self.template},
        )


def _strip_trailing_punct(s: str) -> str:
    return s.strip().rstrip(".,;:").strip()
