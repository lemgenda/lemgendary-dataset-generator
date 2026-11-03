"""
DegradeService — degradation synthesis service and its process entry point.

Single Responsibility: Coordinate and invoke synthetic restoration manifold
generation programmatically. The CLI calls :func:`run_degrade` in-process;
an API job runs ``python -m services.degrade_service`` with the arguments of
:func:`degrade_invocation`, which :func:`main` parses with the same table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TypedDict, Unpack, cast

from rich.console import Console

from tools import generate_degrade

from ._errors import SERVICE_ERRORS
from .invocation import (
    Flag,
    FlagTable,
    ServiceInvocation,
    add_flags,
    flag_args,
    parsed_options,
)

console = Console()
logger = logging.getLogger(__name__)


class DegradeOptions(TypedDict, total=False):
    """Optional settings for :func:`run_degrade`."""

    profile: str
    intensity: str
    pairs: int | None
    val_split: float
    seed: int
    image_format: str
    workers: int | None
    dry_run: bool


DEGRADE_FLAGS: FlagTable = (
    Flag("profile", "--profile", "text"),
    Flag("intensity", "--intensity", "text"),
    Flag("pairs", "--pairs", "int"),
    Flag("val_split", "--val-split", "float"),
    Flag("seed", "--seed", "int"),
    Flag("image_format", "--image-format", "text"),
    Flag("workers", "--workers", "int"),
    Flag("dry_run", "--dry-run", "switch"),
)


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


def degrade_invocation(source: str, output: str, options: DegradeOptions) -> ServiceInvocation:
    """The isolated-process form of ``run_degrade(source, output, **options)``."""
    args = ["--source", source, "--output", output, *flag_args(options, DEGRADE_FLAGS)]
    return ServiceInvocation(__name__, tuple(args))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    return add_flags(parser, DEGRADE_FLAGS)


def main(argv: list[str] | None = None) -> int:
    """Process entry point: ``python -m services.degrade_service --source S --output O ...``."""
    ns = _parser().parse_args(argv)
    options = cast(DegradeOptions, parsed_options(ns, DEGRADE_FLAGS))
    return run_degrade(ns.source, ns.output, **options)


if __name__ == "__main__":
    sys.exit(main())
