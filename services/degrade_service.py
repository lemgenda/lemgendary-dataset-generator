"""
DegradeService — In-process degradation synthesis service.

Single Responsibility: Coordinate and invoke synthetic restoration manifold
generation programmatically without subprocess spawning.
"""

from __future__ import annotations

import logging
from typing import TypedDict, Unpack, cast

from rich.console import Console

from tools import generate_degrade

from ._errors import SERVICE_ERRORS

console = Console()
logger = logging.getLogger(__name__)


class DegradeOptions(TypedDict, total=False):
    """Optional settings for :meth:`DegradeService.run_degrade`."""

    profile: str
    intensity: str
    pairs: int | None
    val_split: float
    seed: int
    image_format: str
    workers: int | None
    dry_run: bool


class DegradeService:
    """Encapsulates degradation manifold synthesis."""

    @staticmethod
    def run_degrade(
        source: str,
        output: str,
        **options: Unpack[DegradeOptions],
    ) -> int:
        """Execute synthetic degradation manifold generation in-process.

        ``options`` defaults: ``profile="motion-blur+iso-noise"``,
        ``intensity="medium"``, ``pairs=None``, ``val_split=0.12``, ``seed=42``,
        ``image_format="webp"``, ``workers=None`` and ``dry_run=False``.
        """

        fmt = cast(
            generate_degrade.ImageFormat,
            options.get("image_format", generate_degrade.DEFAULT_IMAGE_FORMAT),
        )
        try:
            return generate_degrade.synthesize_manifold(
                source=source,
                output=output,
                profile_expr=options.get("profile", generate_degrade.DEFAULT_PROFILE),
                intensity=options.get("intensity", generate_degrade.DEFAULT_INTENSITY),
                pairs=options.get("pairs"),
                val_split=options.get("val_split", generate_degrade.DEFAULT_VAL_SPLIT),
                seed=options.get("seed", generate_degrade.DEFAULT_SEED),
                image_format=fmt,
                workers=options.get("workers"),
                dry_run=options.get("dry_run", False),
            )
        except SERVICE_ERRORS as exc:
            logger.exception("Degradation synthesis error")
            console.print(f"[bold red]Degradation synthesis error:[/bold red] {exc}")
            return 1
