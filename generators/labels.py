"""
Label generation strategies.

Phase 5 of the 2026 modernization roadmap.

Composes the existing worker-global model wrappers into label artifacts.
No new model loading — the caller passes in the already-constructed
CaptionSentry / CLIPManifold / AutoLabeler / QualitySentry instances.

Strategies:
    blip_caption          — natural-language caption (BLIP)
    clip_zeroshot         — zero-shot class prediction (CLIP)
    yolo_detection        — bounding boxes + landmarks (AutoLabeler)
    parsenet_segmentation — semantic segmentation polygons (AutoLabeler in seg mode)
    nima_quality          — 10-bin quality distribution (QualitySentry)
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from .base import GenerationResult


_VALID_STRATEGIES: frozenset[str] = frozenset({
    "blip_caption",
    "clip_zeroshot",
    "yolo_detection",
    "parsenet_segmentation",
    "nima_quality",
})


class LabelGenerator:
    """Dispatches a task to the appropriate loaded model wrapper."""

    def __init__(
        self,
        strategy: str,
        *,
        captioner: Any = None,
        clip: Any = None,
        labeler: Any = None,
        sentry: Any = None,
        classes: list[str] | None = None,
    ) -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown label strategy: {strategy!r}. "
                f"Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self.strategy = strategy
        self._captioner = captioner
        self._clip = clip
        self._labeler = labeler
        self._sentry = sentry
        self._classes = classes or []

    def generate(
        self,
        img: Image.Image,
        context: dict[str, Any] | None = None,
    ) -> GenerationResult:
        ctx = context or {}
        if self.strategy == "blip_caption":
            return self._blip(img)
        if self.strategy == "clip_zeroshot":
            return self._clip_zero(img)
        if self.strategy in ("yolo_detection", "parsenet_segmentation"):
            return self._detector(img, ctx)
        if self.strategy == "nima_quality":
            return self._nima(img)
        return GenerationResult(
            kind="label",
            value={},
            strategy=self.strategy,
        )

    # ── Strategies ──────────────────────────────────────────────────────────
    def _blip(self, img: Image.Image) -> GenerationResult:
        if self._captioner is None:
            return GenerationResult(kind="label", value={}, strategy=self.strategy)
        caption = self._captioner.generate(img)
        return GenerationResult(
            kind="label",
            value={"caption": caption},
            strategy=self.strategy,
        )

    def _clip_zero(self, img: Image.Image) -> GenerationResult:
        if self._clip is None or not self._classes:
            return GenerationResult(kind="label", value={}, strategy=self.strategy)
        # CLIPManifold.tag_style uses its own internal label list; caller
        # can override self._classes to drive a custom vocabulary.
        tag = self._clip.tag_style(img)
        class_label = 0
        if tag in self._classes:
            class_label = self._classes.index(tag)
        return GenerationResult(
            kind="label",
            value={"class_label": class_label, "tag": tag},
            strategy=self.strategy,
        )

    def _detector(
        self,
        img: Image.Image,
        ctx: dict[str, Any],
    ) -> GenerationResult:
        if self._labeler is None:
            return GenerationResult(kind="label", value={}, strategy=self.strategy)
        annotations = self._labeler.predict(img) or []
        return GenerationResult(
            kind="label",
            value={"annotations": annotations},
            strategy=self.strategy,
            metadata={"task": ctx.get("task"), "slug": ctx.get("slug")},
        )

    def _nima(self, img: Image.Image) -> GenerationResult:
        if self._sentry is None:
            return GenerationResult(kind="label", value={}, strategy=self.strategy)
        score, probs = self._sentry.score(img, return_probs=True)
        return GenerationResult(
            kind="label",
            value={"nima_probs": probs, "nima_score": score},
            strategy=self.strategy,
            confidence=float(score) if isinstance(score, (int, float)) else None,
        )