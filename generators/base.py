"""
Generator protocol.

Defines the shape that every generator (labels, prompts, masks) implements.
Nothing here is called by the current compiler — Phase 5 wires these into
`process_image` when the config requests a generation strategy.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class Generator(Protocol):
    """Structural interface for a label/prompt/mask generator.

    Implementations are stateful (model-loaded) and thread-safe within a
    single worker process. Each worker instantiates its own copy.
    """

    def __init__(self, device: str = "cuda") -> None: ...

    def generate(self, img: Image.Image, context: dict[str, Any] | None = None) -> Any:
        """Produce the derived artifact for a single image.

        Return type depends on the generator:
            labels.py    -> dict | list
            prompts.py   -> str
            masks.py     -> PIL.Image (mode "L" or "1")
        """
        ...