"""Top-level compile, transcode, audit, version and test commands."""

from __future__ import annotations

import dataclasses
import logging
import unittest
from typing import Annotated

import typer

from commands.common import (
    PROJECT_ROOT,
    _is_server_available,
    console,
    route_through_sidecar,
)
from commands.options import (
    NoServerOpt,
    TargetOptions,
    flat_values,
    options_command,
    require_target,
)
from core.cli_args import PROJECT_NAME, __version__
from services import (
    MigrationService,
    audit_manifold,
    run_compile,
)

logger = logging.getLogger(__name__)


# ─── Top-level commands ─────────────────────────────────────────────────────
def version() -> None:
    """Print the current version of the Dataset Compiler Suite."""
    console.print(f"[bold cyan]LemGendary Dataset Compiler Suite[/bold cyan] v{__version__}")
    console.print(f"[dim]Project: {PROJECT_NAME}[/dim]")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompileSelection:
    """Which model to compile, with preset, size cap, name suffix and workers."""

    model: Annotated[str | None, typer.Option("--model", "-m", help="Model key to compile")] = (
        None
    )
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset", "-p", help="Compiler preset profile name (from presets.yaml)"
        ),
    ] = None
    max_gb: Annotated[float | None, typer.Option("--max-gb", help="Override max_size_gb")] = None
    suffix: Annotated[
        str | None, typer.Option("--suffix", help="Manifold name suffix override")
    ] = None
    workers: Annotated[int | None, typer.Option("--workers", help="Parallel worker count")] = (
        None
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompileGates:
    """Switches that disable the vetting, labeling and dedup-hash stages."""

    no_vetting: Annotated[
        bool, typer.Option("--no-vetting", help="Disable NIMA quality gate")
    ] = False
    no_labeling: Annotated[
        bool, typer.Option("--no-labeling", help="Disable YOLO auto-labeling")
    ] = False
    no_hash: Annotated[bool, typer.Option("--no-hash", help="Disable deduplication hash")] = (
        False
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompileImageOptions:
    """Image / target / mask encoding overrides (``None`` = preset or default)."""

    image_format: Annotated[
        str | None, typer.Option("--image-format", help="webp | jpeg | png | keep")
    ] = None
    image_quality: Annotated[int | None, typer.Option("--image-quality")] = None
    target_quality: Annotated[int | None, typer.Option("--target-quality")] = None
    mask_format: Annotated[
        str | None, typer.Option("--mask-format", help="webp-lossless | png")
    ] = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompileContainerOptions:
    """Extra container formats and the hardlink-fraction overrides."""

    also_format: Annotated[
        str | None,
        typer.Option(
            "--also-format", help="Comma-separated container formats (mds,litdata,...)"
        ),
    ] = None
    force_duplicate: Annotated[bool, typer.Option("--force-duplicate")] = False
    accept_space_loss: Annotated[bool, typer.Option("--accept-space-loss")] = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompileStrategies:
    """Smart-generation strategy overrides."""

    label_strategy: Annotated[str | None, typer.Option("--label-strategy")] = None
    prompt_strategy: Annotated[str | None, typer.Option("--prompt-strategy")] = None
    mask_strategy: Annotated[str | None, typer.Option("--mask-strategy")] = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompileOptions:
    """Options of the ``compile`` command."""

    selection: CompileSelection
    gates: CompileGates
    images: CompileImageOptions
    containers: CompileContainerOptions
    strategies: CompileStrategies
    shard_ext: Annotated[
        str,
        typer.Option(
            "--shard-ext",
            help="WebDataset shard file extension without dot (default: 'data')",
        ),
    ] = "data"
    no_server: NoServerOpt = False


@options_command(CompileOptions)
def compile_manifold(opts: CompileOptions) -> None:
    """Compile a manifold from raw sources (delegates to services.run_compile or the API)."""
    values = flat_values(opts)
    values.pop("no_server")
    if not opts.no_server and _is_server_available():
        route_through_sidecar("compile", values)  # unset options stay None (preset fills them)

    code = run_compile(**values)
    raise typer.Exit(code=code)


# ─── transcode (Phase 3) ────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True, kw_only=True)
class TranscodeOptions:
    """Options of the ``transcode`` command."""

    target: TargetOptions
    image_format: Annotated[
        str, typer.Option("--image-format", help="Output format (webp/jpeg/png/keep)")
    ] = "webp"
    image_quality: Annotated[
        int, typer.Option("--image-quality", help="Quality for images (1-100)")
    ] = 92
    target_quality: Annotated[
        int, typer.Option("--target-quality", help="Quality for targets (1-100)")
    ] = 95
    mask_format: Annotated[
        str, typer.Option("--mask-format", help="Mask format (webp-lossless/png)")
    ] = "webp-lossless"
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without writing")] = False


@options_command(TranscodeOptions)
def transcode(opts: TranscodeOptions) -> None:
    """Retroactively transcode an existing manifold's images, in place."""
    code = MigrationService.transcode_images(
        manifold_path=require_target(opts.target),
        image_format=opts.image_format,
        image_quality=opts.image_quality,
        target_quality=opts.target_quality,
        mask_format=opts.mask_format,
        dry_run=opts.dry_run,
    )
    raise typer.Exit(code=code)


# ─── audit (Phase 2) ────────────────────────────────────────────────────────
def audit(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key to audit"),
    manifold: str | None = typer.Option(
        None, "--manifold", help="Direct path to a manifold folder"
    ),
    sample: int | None = typer.Option(None, "--sample", help="Cap image scan for a quick preview"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Full image audit pass (resolution, black-frame, dedup, hardlinks)."""
    code = audit_manifold(
        manifold_path=require_target(TargetOptions(model=model, manifold=manifold)),
        sample=sample,
        as_json=as_json,
    )
    raise typer.Exit(code=code)


# ─── Entry point ────────────────────────────────────────────────────────────

def test_datasets(
    pattern: str = typer.Option(
        "test_*.py", "--pattern", "-p", help="Test file pattern to execute"
    ),
) -> None:
    """Run full test suite for the Dataset Compiler Suite in-process using unittest."""
    project_root = PROJECT_ROOT
    tests_dir = project_root / "tests"
    console.print(
        f"[bold cyan]Running LemGendary Dataset Compiler tests in {tests_dir}...[/bold cyan]"
    )
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        console.print(
            f"[bold red][FAIL] Tests failed ({len(result.failures)} failures, {len(result.errors)}"
            " errors)[/bold red]"
        )
        raise typer.Exit(1)
    console.print(
        f"[bold green][PASS] All {result.testsRun} tests passed successfully![/bold green]"
    )
