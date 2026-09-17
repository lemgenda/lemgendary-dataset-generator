"""
Generator protocol and shared result type.

Phase 5 of the 2026 modernization roadmap.

Every generator (labels, prompts, masks) returns a `GenerationResult`.
The consumer (`process_image` or `generate_cli.py`) dispatches on `.kind`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@dataclass
class GenerationResult:
    """Structured output of a single generation call.

    kind is one of:
        'label'   — value is a dict with optional keys:
                        nima_probs:    list[float] (10-bin distribution)
                        class_label:   int
                        annotations:   list[dict] (bbox / pose / segmentation)
        'prompt'  — value is a str
        'mask'    — value is a PIL.Image in mode 'L' or '1'
    """

    kind: str
    value: Any
    strategy: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Generator(Protocol):
    """Structural interface for every generator."""

    strategy: str

    def generate(
        self,
        img: Image.Image,
        context: dict[str, Any] | None = None,
    ) -> GenerationResult: ...