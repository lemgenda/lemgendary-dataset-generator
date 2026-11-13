"""
SyncService — In-process dataset synchronization service.

Single Responsibility: Coordinate and execute push and pull operations
between local compiled manifolds and remote registries (Kaggle).

The CLI calls :class:`SyncService` in-process. API sync jobs run
``python -m services.sync_service kaggle|manifold ...`` (see
:func:`kaggle_invocation`, :func:`manifold_sync_invocation`), which
:func:`main` dispatches to the same methods.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from typing import Any

from rich.console import Console

from sources import kaggle as kaggle_source
from tools import manifold_sync, sync_kaggle_metadata

from ._errors import SERVICE_ERRORS
from .invocation import (
    Flag,
    FlagTable,
    ServiceInvocation,
    add_flags,
    dispatch,
    flag_args,
    parsed_options,
)

console = Console()
logger = logging.getLogger(__name__)


KAGGLE_FLAGS: FlagTable = (
    Flag("action", "--action", "text"),
    Flag("repo_id", "--repo_id", "text"),
    Flag("output_dir", "--output_dir", "text"),
    Flag("is_competition", "--is_competition", "switch"),
    Flag("all_datasets", "--all-datasets", "switch"),
    Flag("no_wait", "--no-wait", "switch"),
)

MANIFOLD_SYNC_FLAGS: FlagTable = (
    Flag("action", "--action", "text"),
    Flag("model", "--model", "text"),
    Flag("repo_id", "--url", "text"),
    Flag("no_wait", "--no-wait", "switch"),
)


def _exit_code(exc: SystemExit) -> int:
    """The exit code a delegated command passed to ``sys.exit``."""
    logger.debug("Delegated command exited via SystemExit(%r)", exc.code)
    return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)


class SyncService:
    """Encapsulates Kaggle dataset synchronization workflows."""

    @staticmethod
    def push(
        model: str,
        url: str | None = None,
        no_wait: bool = False,
        notes: str | None = None,
    ) -> int:
        """Upload a local manifold to Kaggle; returns the exit code.

        ``notes`` describes the new version (added to the generated version notes).
        """
        try:
            manifold_sync.action_sync(
                manifold_name=model,
                repo_id=url,
                no_wait=no_wait,
                notes=notes,
            )
            return 0
        except SystemExit as exc:
            return _exit_code(exc)
        except SERVICE_ERRORS as exc:
            logger.exception("Sync push failed")
            console.print(f"[bold red]Sync push failed:[/bold red] {exc}")
            return 1

    @staticmethod
    def pull(
        url: str,
        output_name: str | None = None,
        only: tuple[str, ...] = (),
    ) -> int:
        """Pull remote manifold from registry (``only``: just these folders or files)."""

        try:
            manifold_sync.action_get(
                repo_id=url,
                output_name=output_name,
                only=only,
            )
            return 0
        except SystemExit as exc:
            return _exit_code(exc)
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
            return _exit_code(exc)
        except SERVICE_ERRORS as exc:
            logger.exception("Sync metadata failed")
            console.print(f"[bold red]Sync metadata failed:[/bold red] {exc}")
            return 1

    @staticmethod
    def kaggle_transfer(**options: Any) -> int:
        """Run one Kaggle download, upload, status check or metadata push.

        ``options`` are the fields of :class:`sources.kaggle.KaggleRequest`.
        """
        request = kaggle_source.KaggleRequest(**options)
        try:
            return kaggle_source.run_kaggle_action(request)
        except SystemExit as exc:
            return _exit_code(exc)
        except SERVICE_ERRORS as exc:
            logger.exception("Kaggle %s failed", request.action)
            console.print(f"[bold red]Kaggle {request.action} failed:[/bold red] {exc}")
            return 1

    @staticmethod
    def sync_manifold(
        action: str, model: str | None = None, repo_id: str | None = None, no_wait: bool = False
    ) -> int:
        """Push (``sync``) or pull (``get``) a manifold, resolving names from the registry."""
        try:
            return manifold_sync.run_sync_action(action, model, repo_id, no_wait=no_wait)
        except SystemExit as exc:
            return _exit_code(exc)
        except SERVICE_ERRORS as exc:
            logger.exception("Manifold %s failed", action)
            console.print(f"[bold red]Manifold {action} failed:[/bold red] {exc}")
            return 1


def kaggle_invocation(**options: Any) -> ServiceInvocation:
    """The isolated-process form of ``SyncService.kaggle_transfer(**options)``."""
    return ServiceInvocation(__name__, ("kaggle", *flag_args(options, KAGGLE_FLAGS)))


def manifold_sync_invocation(**options: Any) -> ServiceInvocation:
    """The isolated-process form of ``SyncService.sync_manifold(**options)``."""
    return ServiceInvocation(__name__, ("manifold", *flag_args(options, MANIFOLD_SYNC_FLAGS)))


def _subcommand(table: FlagTable, run: Callable[..., int]) -> Callable[[list[str]], int]:
    def _main(argv: list[str]) -> int:
        parser = add_flags(argparse.ArgumentParser(), table)
        return run(**parsed_options(parser.parse_args(argv), table))

    return _main


_SUBCOMMANDS: dict[str, Callable[[list[str]], int]] = {
    "kaggle": _subcommand(KAGGLE_FLAGS, SyncService.kaggle_transfer),
    "manifold": _subcommand(MANIFOLD_SYNC_FLAGS, SyncService.sync_manifold),
}


def main(argv: list[str] | None = None) -> int:
    """Process entry point: ``python -m services.sync_service kaggle|manifold [args]``."""
    return dispatch(_SUBCOMMANDS, argv)


if __name__ == "__main__":
    sys.exit(main())
