"""`format` sub-commands: container-format writes and migrations."""

from __future__ import annotations

import logging

import typer

from commands.common import (
    _resolve_target,
)
from services import (
    MigrationService,
)

logger = logging.getLogger(__name__)
format_app = typer.Typer(help="Container-format writes (MDS / LitData / WebDataset / Parquet).")


# ─── format sub-app (Phase 4) ───────────────────────────────────────────────
@format_app.command("write")
def format_write(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(
        None, "--manifold", help="Direct path to a manifold folder"
    ),
    to: str = typer.Option(
        ..., "--to", help="Comma-separated container formats (mds,litdata,webdataset,parquet)"
    ),
    force_duplicate: bool = typer.Option(
        False, "--force-duplicate", help="Proceed despite WARN-tier hardlink fraction"
    ),
    accept_space_loss: bool = typer.Option(
        False, "--accept-space-loss", help="Proceed despite BLOCK-tier hardlink fraction"
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Write additional container formats for an existing manifold."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = MigrationService.migrate_containers(
        manifold_path=target,
        to_formats=to,
        force_duplicate=force_duplicate,
        accept_space_loss=accept_space_loss,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


@format_app.command("migrate")
def format_migrate(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(
        None, "--manifold", help="Direct path to a manifold folder"
    ),
    to: str = typer.Option(..., "--to", help="Comma-separated container formats"),
    force_duplicate: bool = typer.Option(False, "--force-duplicate"),
    accept_space_loss: bool = typer.Option(False, "--accept-space-loss"),
    purge_source: bool = typer.Option(
        False, "--purge-source", help="Remove the directory layout after migration"
    ),
    verify: bool = typer.Option(False, "--verify"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Retroactively write container formats for an existing manifold."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = MigrationService.migrate_containers(
        manifold_path=target,
        to_formats=to,
        force_duplicate=force_duplicate,
        accept_space_loss=accept_space_loss,
        purge_source=purge_source,
        verify=verify,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)
