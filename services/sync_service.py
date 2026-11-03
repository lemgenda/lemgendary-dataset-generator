"""
SyncService — In-process dataset synchronization service.

Single Responsibility: Coordinate and execute push and pull operations
between local compiled manifolds and remote registries (Kaggle).
"""

from __future__ import annotations

import logging

from rich.console import Console

from tools import manifold_sync, sync_kaggle_metadata

from ._errors import SERVICE_ERRORS

console = Console()
logger = logging.getLogger(__name__)


class SyncService:
    """Encapsulates Kaggle dataset synchronization workflows."""

    @staticmethod
    def push(
        model: str,
        url: str | None = None,
        no_wait: bool = False,
    ) -> int:

        try:
            manifold_sync.action_sync(
                manifold_name=model,
                repo_id=url,
                no_wait=no_wait,
            )
            return 0
        except SystemExit as exc:
            logger.debug("Delegated command exited via SystemExit(%r)", exc.code)
            return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except SERVICE_ERRORS as exc:
            logger.exception("Sync push failed")
            console.print(f"[bold red]Sync push failed:[/bold red] {exc}")
            return 1

    @staticmethod
    def pull(
        url: str,
        output_name: str | None = None,
    ) -> int:
        """Pull remote manifold from registry."""

        try:
            manifold_sync.action_get(
                repo_id=url,
                output_name=output_name,
            )
            return 0
        except SystemExit as exc:
            logger.debug("Delegated command exited via SystemExit(%r)", exc.code)
            return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except SERVICE_ERRORS as exc:
            logger.exception("Sync pull failed")
            console.print(f"[bold red]Sync pull failed:[/bold red] {exc}")
            return 1

    @staticmethod
    def push_metadata(
        model: str | None = None,
        all_datasets: bool = False,
        url: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Push Kaggle dataset metadata without re-uploading file archives."""

        try:
            return sync_kaggle_metadata.sync_metadata(
                repo_id=url,
                metadata_dir=model,
                all_datasets=all_datasets,
                dry_run=dry_run,
            )
        except SystemExit as exc:
            logger.debug("Delegated command exited via SystemExit(%r)", exc.code)
            return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except SERVICE_ERRORS as exc:
            logger.exception("Sync metadata failed")
            console.print(f"[bold red]Sync metadata failed:[/bold red] {exc}")
            return 1
