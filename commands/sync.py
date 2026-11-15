"""`sync` sub-commands: Kaggle push, pull and metadata."""

from __future__ import annotations

import dataclasses
import logging
from typing import Annotated

import typer

from commands.common import (
    console,
)
from commands.options import options_command
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class SyncPushOptions:
    """CLI options for ``sync push``."""

    model: Annotated[
        str, typer.Option(..., "--model", "-m", help="Local manifold name or registry key")
    ]
    url: Annotated[str | None, typer.Option("--url", help="Kaggle slug override")] = None
    no_wait: Annotated[
        bool, typer.Option("--no-wait", help="Skip server-side extraction monitoring")
    ] = False
    notes: Annotated[
        str | None,
        typer.Option(
            "--notes",
            help="What changed in this version; added to the generated version notes",
        ),
    ] = None
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Bypass idempotency check and force re-upload"),
    ] = False
    report: Annotated[
        str | None,
        typer.Option("--report", help="Write machine-readable JSON sync report to path"),
    ] = None
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON sync report to stdout"),
    ] = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class SyncPullOptions:
    """CLI options for ``sync pull``."""

    url: Annotated[
        str, typer.Option(..., "--url", help="Kaggle dataset slug (user/name[@version])")
    ]
    only: OnlyOpt = None
    output: Annotated[
        str | None,
        typer.Option(
            "--output", help="Local folder name under the datasets root (default: registry)"
        ),
    ] = None
    version: Annotated[
        str | None,
        typer.Option("--version", "-v", help="Pin dataset version (e.g. 2, v2, or latest)"),
    ] = None
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip interactive confirmation prompts")
    ] = False
    report: Annotated[
        str | None,
        typer.Option("--report", help="Write machine-readable JSON sync report to path"),
    ] = None
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON sync report to stdout"),
    ] = False


# ─── sync sub-app ───────────────────────────────────────────────────────────
@sync_app.command("push")
@options_command(SyncPushOptions)
def sync_push(opts: SyncPushOptions) -> None:
    """Push a compiled manifold to Kaggle."""
    code = SyncService.push(
        model=opts.model,
        url=opts.url,
        no_wait=opts.no_wait,
        notes=opts.notes,
        force=opts.force,
        report_path=opts.report,
        json_output=opts.json_output,
    )
    raise typer.Exit(code=code)


@sync_app.command("pull")
@options_command(SyncPullOptions)
def sync_pull(opts: SyncPullOptions) -> None:
    """Pull a compiled manifold from Kaggle (all of it, or the --only members)."""
    code = SyncService.pull(
        url=opts.url,
        output_name=opts.output,
        only=tuple(opts.only or ()),
        version=opts.version,
        force=opts.force,
        report_path=opts.report,
        json_output=opts.json_output,
    )
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
