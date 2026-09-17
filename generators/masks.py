"""
Mask generation strategies.

Phase 5 of the 2026 modernization roadmap.

Strategies:
    parsenet  — face parsing via the existing AutoLabeler in segmentation mode
                (bitmap output derived from the polygon prediction)
    sam       — Segment Anything v2 (optional; install `segment-anything`)
    modnet    — portrait matting (optional; install `modnet`)

The optional strategies raise ImportError with install instructions on
first use. Phase 5 core ships `parsenet` only.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .base import GenerationResult


_VALID_STRATEGIES: frozenset[str] = frozenset({"parsenet", "sam", "modnet"})


class MaskGenerator:
    """Dispatch a mask generation call to the configured strategy."""

    def __init__(
        self,
        strategy: str = "parsenet",
        *,
        labeler: Any = None,
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown mask strategy: {strategy!r}. "
                f"Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self.strategy = strategy
        self._labeler = labeler
        self._size = image_size
        self._sam_predictor: Any = None
        self._modnet: Any = None

    def generate(
        self,
        img: Image.Image,
        context: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if self.strategy == "parsenet":
            return self._parsenet(img)
        if self.strategy == "sam":
            return self._sam(img)
        if self.strategy == "modnet":
            return self._modnet_impl(img)
        return GenerationResult(kind="mask", value=None, strategy=self.strategy)

    # ── ParseNet ────────────────────────────────────────────────────────────
    def _parsenet(self, img: Image.Image) -> GenerationResult:
        """Render the labeler's polygon output as a single-channel bitmap.

        The AutoLabeler returns annotations in the same shape used by
        process_image. For segmentation mode it produces a list of dicts
        with 'type': 'segmentation' and 'data' being a flat polygon list.
        Polygons are drawn onto a blank L-mode image; class indices map to
        pixel values (0 = background, 1..N = classes).
        """
        if self._labeler is None:
            return GenerationResult(kind="mask", value=None, strategy=self.strategy)

        annotations = self._labeler.predict(img) or []
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for ann in annotations:
            if ann.get("type") != "segmentation":
                continue
            cls = int(ann.get("cls", 1))
            data = ann.get("data") or []
            points = _pairs(data)
            if len(points) >= 3:
                draw.polygon(points, fill=cls)

        if self._size and (w, h) != self._size:
            mask = mask.resize(self._size, Image.Resampling.NEAREST)

        return GenerationResult(
            kind="mask",
            value=mask,
            strategy=self.strategy,
            metadata={"polygon_count": len(annotations)},
        )

    # ── SAM (optional) ──────────────────────────────────────────────────────
    def _sam(self, img: Image.Image) -> GenerationResult:
        try:
            importlib.import_module("segment_anything")
        except ImportError as e:
            raise ImportError(
                "segment-anything is required for --mask-strategy sam. "
                "Install from: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e
        # SAM requires a checkpoint path and prompts; this strategy is a
        # placeholder until the deployment-specific wiring lands.
        return GenerationResult(kind="mask", value=None, strategy=self.strategy)

    # ── MODNet (optional) ───────────────────────────────────────────────────
    def _modnet_impl(self, img: Image.Image) -> GenerationResult:
        try:
            importlib.import_module("modnet")
        except ImportError as e:
            raise ImportError(
                "modnet is required for --mask-strategy modnet. "
                "Install from: pip install git+https://github.com/ZHKKKe/MODNet.git"
            ) from e
        return GenerationResult(kind="mask", value=None, strategy=self.strategy)


def _pairs(flat: list[Any]) -> list[tuple[float, float]]:
    """Convert a flat [x0,y0,x1,y1,...] list into point tuples.

    Silently drops a trailing odd element if present.
    """
    out: list[tuple[float, float]] = []
    n = len(flat) - (len(flat) % 2)
    for i in range(0, n, 2):
        try:
            out.append((float(flat[i]), float(flat[i + 1])))
        except (TypeError, ValueError):
            continue
    return out


# Silence "unused import" for numpy — retained for future SAM/MODNet strategies
_ = np