"""`sync` sub-commands: Kaggle push, pull and metadata."""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from commands.common import (
    console,
)
from services.sync_service import SyncService

logger = logging.getLogger(__name__)
sync_app = typer.Typer(help="Kaggle dataset sync (push / pull / metadata).")

OnlyOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--only",
        help="Fetch only this folder (ending in /) or file, relative to the dataset's top"
             " folder; repeatable (e.g. --only labels/ --only classes.txt)",
    ),
]


# ─── sync sub-app ───────────────────────────────────────────────────────────
@sync_app.command("push")
def sync_push(
    model: str = typer.Option(..., "--model", "-m", help="Local manifold name or registry key"),
    url: str | None = typer.Option(None, "--url", help="Kaggle slug override"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Skip server-side extraction monitoring"),
) -> None:
    """Push a compiled manifold to Kaggle."""
    code = SyncService.push(model=model, url=url, no_wait=no_wait)
    raise typer.Exit(code=code)


@sync_app.command("pull")
def sync_pull(
    url: str = typer.Option(..., "--url", help="Kaggle dataset slug (user/name)"),
    only: OnlyOpt = None,
    output: str | None = typer.Option(
        None, "--output", help="Local folder name under the datasets root (default: registry)"
    ),
) -> None:
    """Pull a compiled manifold from Kaggle (all of it, or the --only members)."""
    code = SyncService.pull(url=url, output_name=output, only=tuple(only or ()))
    raise typer.Exit(code=code)


@sync_app.command("metadata")
def sync_metadata(
    model: str | None = typer.Option(
        None, "--model", "-m", help="Specific dataset or manifold key"
    ),
    all_datasets: bool = typer.Option(
        False, "--all", "-a", help="Update metadata for all registered datasets"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview metadata push without mutating Kaggle"
    ),
) -> None:
    """Push Kaggle dataset metadata without re-uploading file archives."""
    if not model and not all_datasets:
        console.print("[red]Error: specify either --model <key> or --all[/red]")
        raise typer.Exit(code=1)

    code = SyncService.push_metadata(model=model, all_datasets=all_datasets, dry_run=dry_run)
    raise typer.Exit(code=code)
