"""`docs` sub-commands: documentation regeneration."""

from __future__ import annotations

import logging

import typer

from services.doc_service import DocService

logger = logging.getLogger(__name__)
docs_app = typer.Typer(help="Documentation regeneration.")


# ─── docs sub-app ───────────────────────────────────────────────────────────
@docs_app.command("regen")
def docs_regen() -> None:
    """Regenerate per-manifold README / dataset_info / category / classes."""
    code = DocService.regenerate_all_docs()
    raise typer.Exit(code=code)


@docs_app.command("manifolds")
def docs_manifolds(
    check: bool = typer.Option(False, "--check", help="Dry-run: print summary only"),
) -> None:
    """Rebuild the top-level manifolds.md from live registry DBs."""
    code = DocService.rebuild_manifolds_md(check_only=check)
    raise typer.Exit(code=code)
