"""`format` sub-commands: container-format writes and migrations."""

from __future__ import annotations

import dataclasses
import logging
from typing import Annotated

import typer

from commands.options import (
    DryRunOpt,
    TargetOptions,
    options_command,
    require_target,
)
from services.migration_service import MigrationService

logger = logging.getLogger(__name__)
format_app = typer.Typer(help="Container-format writes (MDS / LitData / WebDataset / Parquet).")


# ─── format sub-app (Phase 4) ───────────────────────────────────────────────
@dataclasses.dataclass(frozen=True, kw_only=True)
class FormatWriteOptions:
    """Options of ``format write``."""

    target: TargetOptions
    to: Annotated[
        str,
        typer.Option(
            "--to", help="Comma-separated container formats (mds,litdata,webdataset,parquet)"
        ),
    ]
    force_duplicate: Annotated[
        bool,
        typer.Option("--force-duplicate", help="Proceed despite WARN-tier hardlink fraction"),
    ] = False
    accept_space_loss: Annotated[
        bool,
        typer.Option("--accept-space-loss", help="Proceed despite BLOCK-tier hardlink fraction"),
    ] = False
    dry_run: DryRunOpt = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class FormatMigrateOptions:
    """Options of ``format migrate``."""

    target: TargetOptions
    to: Annotated[str, typer.Option("--to", help="Comma-separated container formats")]
    force_duplicate: Annotated[bool, typer.Option("--force-duplicate")] = False
    accept_space_loss: Annotated[bool, typer.Option("--accept-space-loss")] = False
    purge_source: Annotated[
        bool,
        typer.Option("--purge-source", help="Remove the directory layout after migration"),
    ] = False
    verify: Annotated[bool, typer.Option("--verify")] = False
    dry_run: DryRunOpt = False


@format_app.command("write")
@options_command(FormatWriteOptions)
def format_write(opts: FormatWriteOptions) -> None:
    """Write additional container formats for an existing manifold."""
    code = MigrationService.migrate_containers(
        manifold_path=require_target(opts.target),
        to_formats=opts.to,
        force_duplicate=opts.force_duplicate,
        accept_space_loss=opts.accept_space_loss,
        dry_run=opts.dry_run,
    )
    raise typer.Exit(code=code)


@format_app.command("migrate")
@options_command(FormatMigrateOptions)
def format_migrate(opts: FormatMigrateOptions) -> None:
    """Retroactively write container formats for an existing manifold."""
    code = MigrationService.migrate_containers(
        manifold_path=require_target(opts.target),
        to_formats=opts.to,
        force_duplicate=opts.force_duplicate,
        accept_space_loss=opts.accept_space_loss,
        purge_source=opts.purge_source,
        verify=opts.verify,
        dry_run=opts.dry_run,
    )
    raise typer.Exit(code=code)
