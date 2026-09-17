"""
LemGendary Dataset Compiler — Unified CLI.

Phase 1.5 of the 2026 modernization roadmap.

Thin Typer shell over the existing entry points. Every command either
delegates to a current script (`manifold_compile.py`, `modernize_manifold.py`,
etc.) or points the user at the phase that will implement it.

Nothing here changes runtime behavior — this is a structural layer so
Phases 2-7 can add commands one at a time.

Usage:
    python cli.py --help
    python cli.py compile --model nima_technical
    python cli.py env validate
    python cli.py version
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cli_args import PROJECT_NAME, __version__, resolve_lem_env, venv_python


app = typer.Typer(
    name="lemgendary",
    help="LemGendary Dataset Compiler Suite — unified CLI.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# Sub-apps
env_app = typer.Typer(help="Delegate code/infra operations to lem-env (Environment Manager).")
sync_app = typer.Typer(help="Kaggle dataset sync (push / pull).")
docs_app = typer.Typer(help="Documentation regeneration.")
config_app = typer.Typer(help="Configuration inspection and validation.")


# ─── Helpers ────────────────────────────────────────────────────────────────
def _run(cmd: list[str]) -> int:
    """Run a subprocess, print the command, and return its exit code."""
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError as e:
        console.print(f"[red]Command not found:[/red] {e}")
        return 127


def _stub(phase: int, description: str) -> None:
    """Print a phase pointer and exit non-zero."""
    console.print(f"[yellow]Not yet implemented.[/yellow] {description}")
    console.print(f"[dim]Scheduled for Phase {phase} of the modernization roadmap.[/dim]")
    raise typer.Exit(code=1)


# ─── Top-level commands ─────────────────────────────────────────────────────
@app.command()
def version() -> None:
    """Print the current version of the Dataset Compiler Suite."""
    console.print(f"[bold cyan]LemGendary Dataset Compiler Suite[/bold cyan] v{__version__}")
    console.print(f"[dim]Project: {PROJECT_NAME}[/dim]")


@app.command()
def compile(
    model: str | None = typer.Option(None, "--model", "-m", help="Model key to compile"),
    max_gb: float | None = typer.Option(None, "--max-gb", help="Override max_size_gb"),
    suffix: str | None = typer.Option(None, "--suffix", help="Manifold name suffix override"),
    workers: int | None = typer.Option(None, "--workers", help="Parallel worker count"),
    no_vetting: bool = typer.Option(False, "--no-vetting", help="Disable NIMA quality gate"),
    no_labeling: bool = typer.Option(False, "--no-labeling", help="Disable YOLO auto-labeling"),
    no_hash: bool = typer.Option(False, "--no-hash", help="Disable deduplication hash"),
) -> None:
    """Compile a manifold from raw sources (delegates to manifold_compile.py)."""
    cmd = [venv_python(), "manifold_compile.py"]
    if model:
        cmd += ["--model", model]
    if max_gb is not None:
        cmd += ["--max_gb", str(max_gb)]
    if suffix:
        cmd += ["--suffix", suffix]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    if no_vetting:
        cmd += ["--no-vetting"]
    if no_labeling:
        cmd += ["--no-labeling"]
    if no_hash:
        cmd += ["--no-hash"]
    raise typer.Exit(code=_run(cmd))


@app.command()
def reduce(
    max_gb: float | None = typer.Option(None, "--max-gb", help="Target max size for reduced variant"),
) -> None:
    """Create a downsampled manifold variant (delegates to manifold_reduce.py)."""
    cmd = [venv_python(), "manifold_reduce.py", "--reduce"]
    if max_gb is not None:
        cmd += ["--max_gb", str(max_gb)]
    raise typer.Exit(code=_run(cmd))


@app.command()
def modernize(
    all_: bool = typer.Option(False, "--all", help="Migrate every eligible manifold"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation gate"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without modifying"),
    datasets: str | None = typer.Option(None, "--datasets", help="Comma-separated manifold list"),
    skip_kaggle: bool = typer.Option(False, "--skip-kaggle", help="Rename locally, no re-upload"),
) -> None:
    """Retire the legacy `Large` suffix (delegates to modernize_manifold.py)."""
    cmd = [venv_python(), "modernize_manifold.py"]
    if all_:
        cmd += ["--all"]
    if yes:
        cmd += ["--yes"]
    if dry_run:
        cmd += ["--dry-run"]
    if datasets:
        cmd += ["--datasets", datasets]
    if skip_kaggle:
        cmd += ["--skip-kaggle"]
    raise typer.Exit(code=_run(cmd))


# ─── env sub-app (SSOT for env-manager delegation) ──────────────────────────
@env_app.command("validate")
def env_validate() -> None:
    """Delegate code validation to lem-env (py_compile, lint, YAML, JSON, WCAG)."""
    try:
        lem_env = resolve_lem_env()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3)
    raise typer.Exit(code=_run([lem_env, "validate", "--project", PROJECT_NAME]))


@env_app.command("status")
def env_status() -> None:
    """Delegate ecosystem health audit to lem-env (fast mode)."""
    try:
        lem_env = resolve_lem_env()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3)
    raise typer.Exit(code=_run([lem_env, "audit", "--fast"]))


@env_app.command("install")
def env_install(
    no_clean: bool = typer.Option(False, "--no-clean", help="Do not purge .venv first"),
) -> None:
    """Delegate environment provisioning to lem-env."""
    try:
        lem_env = resolve_lem_env()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3)
    cmd = [lem_env, "install", "--project", PROJECT_NAME]
    if no_clean:
        cmd += ["--no-clean"]
    raise typer.Exit(code=_run(cmd))


app.add_typer(env_app, name="env")


# ─── sync sub-app ───────────────────────────────────────────────────────────
@sync_app.command("push")
def sync_push(
    model: str = typer.Option(..., "--model", "-m", help="Local manifold name or registry key"),
    url: str | None = typer.Option(None, "--url", help="Kaggle slug override"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Skip server-side extraction monitoring"),
) -> None:
    """Push a compiled manifold to Kaggle."""
    cmd = [venv_python(), "manifold_sync.py", "--action", "sync", "--model", model]
    if url:
        cmd += ["--url", url]
    if no_wait:
        cmd += ["--no-wait"]
    raise typer.Exit(code=_run(cmd))


@sync_app.command("pull")
def sync_pull(
    url: str = typer.Option(..., "--url", help="Kaggle dataset slug (user/name)"),
) -> None:
    """Pull a compiled manifold from Kaggle."""
    raise typer.Exit(code=_run([
        venv_python(), "manifold_sync.py", "--action", "get", "--url", url,
    ]))


app.add_typer(sync_app, name="sync")


# ─── docs sub-app ───────────────────────────────────────────────────────────
@docs_app.command("regen")
def docs_regen() -> None:
    """Regenerate per-manifold README / dataset_info / category / classes."""
    raise typer.Exit(code=_run([venv_python(), "doc_generator.py", "--all"]))


@docs_app.command("manifolds")
def docs_manifolds(
    check: bool = typer.Option(False, "--check", help="Dry-run: print summary only"),
) -> None:
    """Rebuild the top-level manifolds.md from live registry DBs."""
    cmd = [venv_python(), "regenerate_manifolds_md.py"]
    if check:
        cmd += ["--check"]
    raise typer.Exit(code=_run(cmd))


app.add_typer(docs_app, name="docs")


# ─── config sub-app ─────────────────────────────────────────────────────────
@config_app.command("validate")
def config_validate() -> None:
    """Validate unified_data.yaml against the Pydantic schema."""
    raise typer.Exit(code=_run([venv_python(), "config_schema.py"]))


@config_app.command("show")
def config_show() -> None:
    """Print a summary of the current registry configuration."""
    import yaml
    from pathlib import Path
    p = Path("./unified_data.yaml")
    if not p.exists():
        console.print("[red]unified_data.yaml not found[/red]")
        raise typer.Exit(code=3)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    rm = data.get("_registry_metadata", {})
    table = Table(title="Registry Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("version", str(rm.get("version", "")))
    table.add_row("name_prefix", str(rm.get("name_prefix", "")))
    table.add_row("name_suffix", repr(rm.get("name_suffix", "")))
    table.add_row("output_folder_name", str(rm.get("output_folder_name", "")))
    gc = rm.get("global_constraints", {})
    table.add_row("min_size_gb", str(gc.get("min_size_gb", "")))
    table.add_row("max_size_gb", str(gc.get("max_size_gb", "")))
    table.add_row("datasets", str(len(data.get("datasets", {}))))
    console.print(table)


app.add_typer(config_app, name="config")


# ─── Transcode (Phase 3) ────────────────────────────────────────────────────
def _resolve_manifold_path(model: str) -> Path | None:
    """Resolve a registry key to its manifold folder on disk."""
    import yaml as _yaml
    p = Path("./unified_data.yaml")
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        reg = _yaml.safe_load(f) or {}
    meta = reg.get("_registry_metadata", {})
    entry = reg.get("datasets", {}).get(model)
    if entry is None:
        return None
    name = entry.get("name", model)
    prefix = meta.get("name_prefix", "LemGendized")
    suffix = meta.get("name_suffix", "")
    out = Path(meta.get("output_folder_name", "../LemGendaryDatasets"))
    return out / f"{prefix}{name}{suffix}"


@app.command()
def transcode(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(None, "--manifold", help="Direct path to a manifold folder"),
    image_format: str = typer.Option("webp", "--image-format", help="Output format (webp/jpeg/png/keep)"),
    image_quality: int = typer.Option(92, "--image-quality", help="Quality for images (1-100)"),
    target_quality: int = typer.Option(95, "--target-quality", help="Quality for targets (1-100)"),
    mask_format: str = typer.Option("webp-lossless", "--mask-format", help="Mask format (webp-lossless/png)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing"),
) -> None:
    """Retroactively transcode an existing manifold's images, in place."""
    if manifold:
        target = Path(manifold)
    elif model:
        target = _resolve_manifold_path(model)
    else:
        console.print("[red]Provide --model or --manifold[/red]")
        raise typer.Exit(code=1)
    if target is None or not target.exists():
        console.print(f"[red]Manifold not found: {target}[/red]")
        raise typer.Exit(code=1)

    cmd = [
        venv_python(), "migrate_manifold_image_format.py",
        "--manifold", str(target),
        "--image-format", image_format,
        "--image-quality", str(image_quality),
        "--target-quality", str(target_quality),
        "--mask-format", mask_format,
    ]
    if dry_run:
        cmd += ["--dry-run"]
    raise typer.Exit(code=_run(cmd))


# ─── Stubs for future phases ────────────────────────────────────────────────
@app.command()
def audit() -> None:
    """Full image audit pass (resolution, black-frame, dedup, hardlinks)."""
    _stub(2, "Image / annotation audit and dedup pipeline.")


format_app = typer.Typer(help="Container-format writes (MDS / LitData / WebDataset / Parquet).")


@format_app.command("write")
def format_write(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(None, "--manifold", help="Direct path to a manifold folder"),
    to: str = typer.Option(..., "--to", help="Comma-separated container formats (mds,litdata,webdataset,parquet)"),
    force_duplicate: bool = typer.Option(False, "--force-duplicate",
                                         help="Proceed despite WARN-tier hardlink fraction"),
    accept_space_loss: bool = typer.Option(False, "--accept-space-loss",
                                           help="Proceed despite BLOCK-tier hardlink fraction"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Write additional container formats for an existing manifold."""
    if manifold:
        target = Path(manifold)
    elif model:
        target = _resolve_manifold_path(model)
    else:
        console.print("[red]Provide --model or --manifold[/red]")
        raise typer.Exit(code=1)
    if target is None or not target.exists():
        console.print(f"[red]Manifold not found: {target}[/red]")
        raise typer.Exit(code=1)

    cmd = [
        venv_python(), "migrate_manifold_format.py",
        "--manifold", str(target),
        "--to", to,
    ]
    if force_duplicate:
        cmd += ["--force-duplicate"]
    if accept_space_loss:
        cmd += ["--accept-space-loss"]
    if dry_run:
        cmd += ["--dry-run"]
    raise typer.Exit(code=_run(cmd))


@format_app.command("migrate")
def format_migrate(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(None, "--manifold", help="Direct path to a manifold folder"),
    to: str = typer.Option(..., "--to", help="Comma-separated container formats"),
    force_duplicate: bool = typer.Option(False, "--force-duplicate"),
    accept_space_loss: bool = typer.Option(False, "--accept-space-loss"),
    purge_source: bool = typer.Option(False, "--purge-source",
                                      help="Remove the directory layout after migration"),
    verify: bool = typer.Option(False, "--verify"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Retroactively write container formats for an existing manifold."""
    if manifold:
        target = Path(manifold)
    elif model:
        target = _resolve_manifold_path(model)
    else:
        console.print("[red]Provide --model or --manifold[/red]")
        raise typer.Exit(code=1)
    if target is None or not target.exists():
        console.print(f"[red]Manifold not found: {target}[/red]")
        raise typer.Exit(code=1)

    cmd = [
        venv_python(), "migrate_manifold_format.py",
        "--manifold", str(target),
        "--to", to,
    ]
    if force_duplicate:
        cmd += ["--force-duplicate"]
    if accept_space_loss:
        cmd += ["--accept-space-loss"]
    if purge_source:
        cmd += ["--purge-source"]
    if verify:
        cmd += ["--verify"]
    if dry_run:
        cmd += ["--dry-run"]
    raise typer.Exit(code=_run(cmd))


app.add_typer(format_app, name="format")


@app.command()
def label() -> None:
    """Smart label generation (BLIP / YOLO / CLIP)."""
    _stub(5, "Label generation strategies.")


@app.command()
def prompt() -> None:
    """Smart prompt generation for diffusion manifolds."""
    _stub(5, "Prompt generation strategies.")


@app.command()
def mask() -> None:
    """Smart mask generation (SAM / ParseNet)."""
    _stub(5, "Mask generation strategies.")


@app.command()
def degrade() -> None:
    """Compiler-time degradation synthesis (blur, noise, haze, film)."""
    _stub(6, "Degradation engine.")


@app.command()
def server() -> None:
    """Start the local HTTP + WebSocket API server."""
    _stub(7, "FastAPI server + CLI unification.")


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()