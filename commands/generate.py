"""Top-level smart-generation and degradation commands (label, prompt, mask, degrade)."""

from __future__ import annotations

import dataclasses
import logging
from typing import Annotated

import typer

from commands.common import (
    _is_server_available,
    route_through_sidecar,
)
from commands.options import (
    DeviceOpt,
    DryRunOpt,
    NoServerOpt,
    SampleOpt,
    TargetOptions,
    flat_values,
    options_command,
    require_target,
)
from services import (
    DegradeService,
    GenerationService,
)

logger = logging.getLogger(__name__)


# ─── smart generation (Phase 5) ─────────────────────────────────────────────
@dataclasses.dataclass(frozen=True, kw_only=True)
class GenerationRunOptions:
    """Execution options shared by ``label`` / ``prompt`` / ``mask``."""

    device: DeviceOpt = "cpu"
    sample: SampleOpt = None
    dry_run: DryRunOpt = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class LabelOptions:
    """Options of the ``label`` command."""

    target: TargetOptions
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help=(
                "blip_caption | clip_zeroshot | yolo_detection | parsenet_segmentation"
                " | nima_quality"
            ),
        ),
    ] = "blip_caption"
    run: GenerationRunOptions


@dataclasses.dataclass(frozen=True, kw_only=True)
class PromptOptions:
    """Options of the ``prompt`` command."""

    target: TargetOptions
    template: Annotated[
        str, typer.Option("--template", help="diffusers-v1 | sd-v1 | flux-v1 | minimal")
    ] = "diffusers-v1"
    run: GenerationRunOptions


@dataclasses.dataclass(frozen=True, kw_only=True)
class MaskOptions:
    """Options of the ``mask`` command."""

    target: TargetOptions
    strategy: Annotated[str, typer.Option("--strategy", help="parsenet | sam | modnet")] = (
        "parsenet"
    )
    run: GenerationRunOptions


def _generate(
    target: TargetOptions, kind: str, strategy: str, run: GenerationRunOptions, **extra: str
) -> None:
    """Resolve the manifold, run one GenerationService pass and exit with its code."""
    manifold = require_target(target)
    code = GenerationService.generate(
        manifold=manifold,
        kind=kind,
        strategy=strategy,
        **extra,
        device=run.device,
        sample=run.sample,
        dry_run=run.dry_run,
    )
    raise typer.Exit(code=code)


@options_command(LabelOptions)
def label(opts: LabelOptions) -> None:
    """Generate labels for an existing manifold."""
    _generate(opts.target, "label", opts.strategy, opts.run)


@options_command(PromptOptions)
def prompt(opts: PromptOptions) -> None:
    """Generate structured prompts for diffusion manifolds."""
    _generate(opts.target, "prompt", opts.template, opts.run, template=opts.template)


@options_command(MaskOptions)
def mask(opts: MaskOptions) -> None:
    """Generate masks for a segmentation manifold."""
    _generate(opts.target, "mask", opts.strategy, opts.run)


# ─── degradation engine (Phase 6) ───────────────────────────────────────────
@dataclasses.dataclass(frozen=True, kw_only=True)
class DegradeSpec:
    """What to degrade: source, output manifold and degradation profile."""

    source: Annotated[
        str,
        typer.Option(
            "--source", "-s", help="Source clean dataset directory or manifold name"
        ),
    ]
    output: Annotated[
        str, typer.Option("--output", "-o", help="Target synthetic manifold name")
    ]
    profile: Annotated[
        str,
        typer.Option(
            "--profile", "-p", help="Degradation profile expression or preset alias"
        ),
    ] = "motion-blur+iso-noise"
    intensity: Annotated[str, typer.Option("--intensity", help="low | medium | high")] = (
        "medium"
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class DegradeSampling:
    """How many pairs to synthesize, the split, RNG seed, output format and workers."""

    pairs: Annotated[
        int | None, typer.Option("--pairs", help="Max sample pairs to synthesize")
    ] = None
    val_split: Annotated[float, typer.Option("--val-split", help="Validation split ratio")] = (
        0.12
    )
    seed: Annotated[int, typer.Option("--seed", help="Deterministic RNG seed")] = 42
    image_format: Annotated[
        str, typer.Option("--image-format", help="webp | jpeg | png")
    ] = "webp"
    workers: Annotated[
        int | None, typer.Option("--workers", "-w", help="Worker thread count")
    ] = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class DegradeOptions:
    """Options of the ``degrade`` command (flattened, also the sidecar job payload)."""

    spec: DegradeSpec
    sampling: DegradeSampling
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview plan without writing")
    ] = False
    no_server: NoServerOpt = False


@options_command(DegradeOptions)
def degrade(opts: DegradeOptions) -> None:
    """Compiler-time degradation synthesis (blur, noise, haze, rain, jpeg, film)."""
    payload = flat_values(opts)
    payload.pop("no_server")
    if not opts.no_server and _is_server_available():
        route_through_sidecar("degrade", payload)

    code = DegradeService.run_degrade(**payload)
    raise typer.Exit(code=code)
