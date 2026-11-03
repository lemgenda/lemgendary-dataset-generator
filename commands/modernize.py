"""`modernize` sub-commands: modernization queue, transcode and reduction."""

from __future__ import annotations

import dataclasses
import logging
from typing import Annotated

import typer
from rich.table import Table

from commands.common import (
    PROJECT_ROOT,
    console,
)
from commands.options import flat_values, options_command
from services.doc_service import DocService
from services.migration_service import MigrationService
from tools.modernization_queue import load_queue_manifest

logger = logging.getLogger(__name__)
modernize_app = typer.Typer(help="Dataset modernization queue, transcode, and scale reduction.")


# ─── modernize manifolds (Phase 1.6 / Modernization) ────────────────────────
# Registered on the `modernize` sub-app: a top-level `modernize` command would be
# shadowed by the sub-app of the same name and never be reachable.
@dataclasses.dataclass(frozen=True, kw_only=True)
class ModernizeSelection:
    """Which manifolds to modernize, confirmation and Kaggle handling."""

    all_: Annotated[bool, typer.Option("--all", help="Select all eligible manifolds")] = False
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation gate")] = False
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show plan, do not modify anything")
    ] = False
    datasets: Annotated[
        str | None,
        typer.Option("--datasets", help="Comma-separated folder names or dataset keys"),
    ] = None
    skip_kaggle: Annotated[
        bool, typer.Option("--skip-kaggle", help="Skip Kaggle re-upload")
    ] = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModernizeConversion:
    """Transcode and container-conversion settings for ``modernize manifolds``."""

    image_format: Annotated[
        str,
        typer.Option(
            "--image-format", help="Transcode images during modernize (webp/jpeg/png/keep)"
        ),
    ] = "webp"
    image_quality: Annotated[
        int, typer.Option("--image-quality", help="Quality for images (1-100)")
    ] = 92
    also_format: Annotated[
        str | None,
        typer.Option(
            "--also-format",
            help=(
                "Optional container format to convert dataset into (e.g., webdataset, mds,"
                " litdata, parquet)"
            ),
        ),
    ] = None
    skip_transcode: Annotated[
        bool, typer.Option("--skip-transcode", help="Skip WebP transcoding")
    ] = False
    skip_container: Annotated[
        bool, typer.Option("--skip-container", help="Skip container format conversion")
    ] = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModernizeManifoldsOptions:
    """Options of ``modernize manifolds``."""

    selection: ModernizeSelection
    conversion: ModernizeConversion


@modernize_app.command("manifolds")
@options_command(ModernizeManifoldsOptions)
def modernize_manifolds(opts: ModernizeManifoldsOptions) -> None:
    """Modernize manifolds: retire the legacy suffix, transcode to WebP, write containers."""
    code = MigrationService.modernize_manifolds(**flat_values(opts))
    raise typer.Exit(code=code)


# ─── modernize sub-app ──────────────────────────────────────────────────────
@modernize_app.command("queue")
def modernize_queue() -> None:
    """Display modernization queue status and pending conversions."""
    datasets_root = PROJECT_ROOT.parent / "LemGendaryDatasets"
    registry_yaml = PROJECT_ROOT / "unified_data.yaml"
    items = load_queue_manifest(datasets_root.resolve(), registry_yaml.resolve())

    table = Table(title="LemGendary Dataset Modernization Queue")
    table.add_column("Dataset Key", style="cyan")
    table.add_column("Folder", style="green")
    table.add_column("Format", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Reason", style="dim")

    for item in items:
        if item.is_containerized:
            status_label, reason = (
                "CONTAINERIZED",
                f"{item.container_bytes / (1024**3):.1f} GB ready",
            )
        elif item.zip_path:
            status_label, reason = "PENDING STREAM", item.zip_path.name
        elif item.has_loose_files:
            status_label, reason = "PENDING PACK", "Loose images"
        elif item.legacy_dir:
            status_label, reason = "PENDING LEGACY", item.legacy_dir.name
        else:
            status_label, reason = "MISSING SOURCE", "-"
        table.add_row(
            item.key,
            item.modernized_folder,
            item.canonical_format,
            status_label,
            reason,
        )
    console.print(table)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModernizeRunOptions:
    """Options of ``modernize run`` (mirrors POST /api/modernize/run)."""

    dataset: Annotated[
        str | None, typer.Option("--dataset", "-d", help="Specific dataset key to modernize")
    ] = None
    workers: Annotated[
        int | None, typer.Option("--workers", "-w", help="Worker threads for WebP transcoding")
    ] = None
    push_kaggle: Annotated[
        bool, typer.Option("--push-kaggle", help="Push to Kaggle after conversion")
    ] = False
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview plan without executing conversions")
    ] = False
    force: Annotated[bool, typer.Option("--force", help="Force re-conversion")] = False
    purge_source: Annotated[
        bool,
        typer.Option(
            "--purge-source",
            help="Delete zip, legacy folder and loose images after a successful conversion",
        ),
    ] = False


@modernize_app.command("run")
@options_command(ModernizeRunOptions)
def modernize_run(opts: ModernizeRunOptions) -> None:
    """Run batch modernization queue conversion (same service as POST /api/modernize/run)."""
    code = MigrationService.run_modernization_queue(**flat_values(opts))
    raise typer.Exit(code=code)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModernizeReduceOptions:
    """Options of ``modernize reduce`` / top-level ``reduce``."""

    dataset: Annotated[
        str | None, typer.Option("--dataset", "-d", help="Manifold folder name to reduce")
    ] = None
    fraction: Annotated[
        float | None, typer.Option("--fraction", help="Share of samples to keep, in (0, 1]")
    ] = None
    max_gb: Annotated[
        float | None, typer.Option("--max-gb", help="Target max size for reduced variant")
    ] = None
    suffix: Annotated[
        str | None,
        typer.Option("--suffix", help="Suffix for the reduced manifold (default: Reduced)"),
    ] = None
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Do not prompt; requires --dataset")
    ] = False
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report the plan only; write nothing")
    ] = False


@modernize_app.command("reduce")
@options_command(ModernizeReduceOptions)
def modernize_reduce(opts: ModernizeReduceOptions) -> None:
    """Create a downsampled manifold variant (interactive unless --yes)."""
    code = MigrationService.reduce_manifolds(**flat_values(opts))
    raise typer.Exit(code=code)


@modernize_app.command("regen")
def modernize_regen(
    check: bool = typer.Option(False, "--check", help="Dry-run: print summary only"),
) -> None:
    """Regenerate manifolds.md registry index."""
    code = DocService.rebuild_manifolds_md(check_only=check)
    raise typer.Exit(code=code)
