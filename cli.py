"""
LemGendary Dataset Compiler - Unified CLI.

Phase 1.5 of the 2026 modernization roadmap, extended through Phase 8.

Thin Typer shell over the service layer and existing entry points.

Usage:
    python cli.py --help
    python cli.py compile --model nima_technical
    python cli.py env validate
    python cli.py version
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table
import typer
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect
import yaml

from api.auth import get_or_create_token
from core.cli_args import PROJECT_NAME, __version__, resolve_lem_env, venv_python
from services import (
    AuditService,
    CompilerService,
    DegradeService,
    DocService,
    GenerationService,
    MigrationService,
    SyncService,
)
from services.job_watch import JobNotFoundError, JobStatusError, TERMINAL_STATES, wait_for_job

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8100


app = typer.Typer(
    name="lemgendary",
    help="LemGendary Dataset Compiler Suite - unified CLI.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
logger = logging.getLogger(__name__)

# Sub-apps
server_app = typer.Typer(help="Control the local FastAPI + WebSocket API sidecar server.")
env_app = typer.Typer(help="Delegate code/infra operations to lem-env (Environment Manager).")
sync_app = typer.Typer(help="Kaggle dataset sync (push / pull / metadata).")
docs_app = typer.Typer(help="Documentation regeneration.")
config_app = typer.Typer(help="Configuration inspection and validation.")
format_app = typer.Typer(help="Container-format writes (MDS / LitData / WebDataset / Parquet).")
presets_app = typer.Typer(help="Inspect and manage canonical compiler preset profiles.")
forex_app = typer.Typer(help="Forex universe and Parquet conversion operations.")
modernize_app = typer.Typer(help="Dataset modernization queue, transcode, and scale reduction.")
llm_app = typer.Typer(help="LLM instruction dataset compilation and quality invariant validation.")


def _spawn_detached(cmd: list[str], creationflags: int) -> subprocess.Popen:
    """Launch a long-lived background process and hand ownership to the caller.

    The process deliberately outlives the CLI invocation (detached API server),
    so it cannot be scoped to a ``with`` block that would wait for it to exit.
    """
    return subprocess.Popen(cmd, creationflags=creationflags, close_fds=True)


def _is_server_available(host: str = DEFAULT_SERVER_HOST, port: int = DEFAULT_SERVER_PORT) -> bool:
    """Check if the local FastAPI server is healthy and responding."""
    try:
        resp = requests.get(f"http://{host}:{port}/api/health", timeout=0.8)
        return resp.status_code == 200
    except requests.RequestException as exc:
        logger.debug("API server health probe failed on %s:%s: %s", host, port, exc)
        return False


def _stream_job_logs(job_id: str, host: str = DEFAULT_SERVER_HOST, port: int = DEFAULT_SERVER_PORT) -> int:
    """Connect to WebSocket log feed and stream lines to console until completion."""
    ws_url = f"ws://{host}:{port}/api/ws/jobs/{job_id}/logs"
    try:
        with connect(ws_url) as ws:
            for message in ws:
                data = json.loads(message)
                chunk = data.get("chunk", "")
                console.print(chunk, end="")
                if "[PROCESS_TERMINATED]" in chunk:
                    break
    except (WebSocketException, OSError, json.JSONDecodeError) as exc:
        logger.debug("WebSocket log stream for job %s closed: %s", job_id, exc)
        console.print(f"[dim]WebSocket streaming closed: {exc}[/dim]")

    try:
        token = get_or_create_token()
        headers = {"X-API-Key": token}
        resp = requests.get(f"http://{host}:{port}/api/jobs/{job_id}", headers=headers, timeout=5)
        if resp.status_code == 200:
            job_info = resp.json()
            exit_code = job_info.get("exit_code")
            if exit_code is not None:
                return int(exit_code)
            return 0 if job_info.get("state") == "completed" else 1
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Failed fetching final status for job %s: %s", job_id, exc)
        console.print(f"[dim]Failed fetching final job status: {exc}[/dim]")
    return 0


# ─── Helpers ────────────────────────────────────────────────────────────────
def _run(cmd: list[str]) -> int:
    """Run an external CLI command, print the command, and return its exit code."""
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    try:
        result = subprocess.run(cmd, check=False, cwd=str(Path(__file__).resolve().parent))
        return result.returncode
    except FileNotFoundError as e:
        console.print(f"[red]Command not found:[/red] {e}")
        return 127


def _resolve_manifold_path(model: str) -> Path | None:
    p = Path(__file__).resolve().parent / "unified_data.yaml"
    if not p.exists():
        p = Path("./unified_data.yaml")
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f) or {}
    meta = reg.get("_registry_metadata", {})
    entry = reg.get("datasets", {}).get(model)
    if entry is None:
        return None
    name = entry.get("name", model)
    prefix = meta.get("name_prefix", "LemGendized")
    suffix = meta.get("name_suffix", "")
    out = Path(meta.get("output_folder_name", "../LemGendaryDatasets"))
    if not out.is_absolute():
        out = (Path(__file__).resolve().parent / out).resolve()
    mod_folder = entry.get("modernized_folder") or f"{prefix}{name}{suffix}"
    return out / mod_folder


def _resolve_target(model: str | None, manifold: str | None) -> Path | None:
    """Resolve --model or --manifold into an existing manifold path."""
    if manifold:
        p = Path(manifold)
        return p if p.exists() else None
    if model:
        return _resolve_manifold_path(model)
    console.print("[red]Provide --model or --manifold[/red]")
    return None


# ─── Top-level commands ─────────────────────────────────────────────────────
@app.command()
def version() -> None:
    """Print the current version of the Dataset Compiler Suite."""
    console.print(f"[bold cyan]LemGendary Dataset Compiler Suite[/bold cyan] v{__version__}")
    console.print(f"[dim]Project: {PROJECT_NAME}[/dim]")


@app.command(name="compile")
def compile_manifold(
    model: str | None = typer.Option(None, "--model", "-m", help="Model key to compile"),
    preset: str | None = typer.Option(None, "--preset", "-p", help="Compiler preset profile name (from presets.yaml)"),
    max_gb: float | None = typer.Option(None, "--max-gb", help="Override max_size_gb"),
    suffix: str | None = typer.Option(None, "--suffix", help="Manifold name suffix override"),
    workers: int | None = typer.Option(None, "--workers", help="Parallel worker count"),
    no_vetting: bool = typer.Option(False, "--no-vetting", help="Disable NIMA quality gate"),
    no_labeling: bool = typer.Option(False, "--no-labeling", help="Disable YOLO auto-labeling"),
    no_hash: bool = typer.Option(False, "--no-hash", help="Disable deduplication hash"),
    image_format: str | None = typer.Option(None, "--image-format", help="webp | jpeg | png | keep"),
    image_quality: int | None = typer.Option(None, "--image-quality"),
    target_quality: int | None = typer.Option(None, "--target-quality"),
    mask_format: str | None = typer.Option(None, "--mask-format", help="webp-lossless | png"),
    also_format: str | None = typer.Option(
        None, "--also-format", help="Comma-separated container formats (mds,litdata,...)"
    ),
    force_duplicate: bool = typer.Option(False, "--force-duplicate"),
    accept_space_loss: bool = typer.Option(False, "--accept-space-loss"),
    label_strategy: str | None = typer.Option(None, "--label-strategy"),
    prompt_strategy: str | None = typer.Option(None, "--prompt-strategy"),
    mask_strategy: str | None = typer.Option(None, "--mask-strategy"),
    shard_ext: str = typer.Option(
        "data", "--shard-ext", help="WebDataset shard file extension without dot (default: 'data')"
    ),
    no_server: bool = typer.Option(False, "--no-server", help="Bypass API server and execute in-process"),
) -> None:
    """Compile a manifold from raw sources (delegates to CompilerService or API)."""
    if not no_server and _is_server_available():
        token = get_or_create_token()
        payload = {
            "model": model,
            "preset": preset,
            "max_gb": max_gb,
            "suffix": suffix,
            "workers": workers,
            "no_vetting": no_vetting,
            "no_labeling": no_labeling,
            "no_hash": no_hash,
            "image_format": image_format or "webp",
            "image_quality": image_quality or 92,
            "target_quality": target_quality or 95,
            "mask_format": mask_format or "webp-lossless",
            "also_format": also_format,
            "shard_ext": shard_ext,
            "force_duplicate": force_duplicate,
            "accept_space_loss": accept_space_loss,
            "label_strategy": label_strategy,
            "prompt_strategy": prompt_strategy,
            "mask_strategy": mask_strategy,
        }
        try:
            resp = requests.post(
                f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}/api/jobs/compile",
                json=payload,
                headers={"X-API-Key": token},
                timeout=10,
            )
            if resp.status_code == 200:
                job_id = resp.json()["id"]
                console.print(
                    f"[bold green]Routed through LemGendary Dataset Compiler API[/bold green] "
                    f"[dim](job_id: {job_id})[/dim]"
                )
                exit_code = _stream_job_logs(job_id)
                raise typer.Exit(code=exit_code)
            console.print(
                f"[yellow]API rejected job ({resp.status_code}): {resp.text}. Falling back to in-process...[/yellow]"
            )
        except requests.RequestException as e:
            console.print(f"[yellow]Could not route to API ({e}). Falling back to in-process...[/yellow]")

    code = CompilerService.run_compile(
        model=model,
        preset=preset,
        max_gb=max_gb,
        suffix=suffix,
        workers=workers,
        no_vetting=no_vetting,
        no_labeling=no_labeling,
        no_hash=no_hash,
        image_format=image_format,
        image_quality=image_quality,
        target_quality=target_quality,
        mask_format=mask_format,
        also_format=also_format,
        shard_ext=shard_ext,
        force_duplicate=force_duplicate,
        accept_space_loss=accept_space_loss,
        label_strategy=label_strategy,
        prompt_strategy=prompt_strategy,
        mask_strategy=mask_strategy,
    )
    raise typer.Exit(code=code)


# ─── env sub-app ────────────────────────────────────────────────────────────
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
    code = SyncService.push(model=model, url=url, no_wait=no_wait)
    raise typer.Exit(code=code)


@sync_app.command("pull")
def sync_pull(
    url: str = typer.Option(..., "--url", help="Kaggle dataset slug (user/name)"),
) -> None:
    """Pull a compiled manifold from Kaggle."""
    code = SyncService.pull(url=url)
    raise typer.Exit(code=code)


@sync_app.command("metadata")
def sync_metadata(
    model: str | None = typer.Option(None, "--model", "-m", help="Specific dataset or manifold key"),
    all_datasets: bool = typer.Option(False, "--all", "-a", help="Update metadata for all registered datasets"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview metadata push without mutating Kaggle"),
) -> None:
    """Push Kaggle dataset metadata without re-uploading file archives."""
    if not model and not all_datasets:
        console.print("[red]Error: specify either --model <key> or --all[/red]")
        raise typer.Exit(code=1)

    code = SyncService.push_metadata(model=model, all_datasets=all_datasets, dry_run=dry_run)
    raise typer.Exit(code=code)


app.add_typer(sync_app, name="sync")


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


app.add_typer(docs_app, name="docs")


# ─── config sub-app ─────────────────────────────────────────────────────────
@config_app.command("validate")
def config_validate() -> None:
    """Validate unified_data.yaml against the Pydantic schema."""
    from core.config_schema import _cli_validate
    raise typer.Exit(code=_cli_validate())


@config_app.command("show")
def config_show() -> None:
    p = Path(__file__).resolve().parent / "unified_data.yaml"
    if not p.exists():
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


# ─── transcode (Phase 3) ────────────────────────────────────────────────────
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
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = MigrationService.transcode_images(
        manifold_path=target,
        image_format=image_format,
        image_quality=image_quality,
        target_quality=target_quality,
        mask_format=mask_format,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


# ─── modernize manifolds (Phase 1.6 / Modernization) ────────────────────────
# Registered on the `modernize` sub-app: a top-level `modernize` command would be
# shadowed by the sub-app of the same name and never be reachable.
@modernize_app.command("manifolds")
def modernize_manifolds(
    all_: bool = typer.Option(False, "--all", help="Select all eligible manifolds"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation gate"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan, do not modify anything"),
    datasets: str | None = typer.Option(None, "--datasets", help="Comma-separated folder names or dataset keys"),
    skip_kaggle: bool = typer.Option(False, "--skip-kaggle", help="Skip Kaggle re-upload"),
    image_format: str = typer.Option(
        "webp", "--image-format", help="Transcode images during modernize (webp/jpeg/png/keep)"
    ),
    image_quality: int = typer.Option(92, "--image-quality", help="Quality for images (1-100)"),
    also_format: str | None = typer.Option(
        None,
        "--also-format",
        help="Optional container format to convert dataset into (e.g., webdataset, mds, litdata, parquet)",
    ),
    skip_transcode: bool = typer.Option(False, "--skip-transcode", help="Skip WebP transcoding"),
    skip_container: bool = typer.Option(False, "--skip-container", help="Skip container format conversion"),
) -> None:
    """Modernize manifolds: retire legacy suffix, transcode images to WebP, and write container formats."""
    code = MigrationService.modernize_manifolds(
        all_=all_,
        yes=yes,
        dry_run=dry_run,
        datasets=datasets,
        skip_kaggle=skip_kaggle,
        image_format=image_format,
        image_quality=image_quality,
        also_format=also_format,
        skip_transcode=skip_transcode,
        skip_container=skip_container,
    )
    raise typer.Exit(code=code)


# ─── format sub-app (Phase 4) ───────────────────────────────────────────────
@format_app.command("write")
def format_write(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(None, "--manifold", help="Direct path to a manifold folder"),
    to: str = typer.Option(..., "--to", help="Comma-separated container formats (mds,litdata,webdataset,parquet)"),
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
    manifold: str | None = typer.Option(None, "--manifold", help="Direct path to a manifold folder"),
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


app.add_typer(format_app, name="format")


# ─── presets (Phase 8) ───────────────────────────────────────────────────────
@presets_app.command("list")
def presets_list() -> None:
    """List available compiler presets with format and gate details."""
    from core import presets
    all_p = presets.list_presets()
    table = Table(title="LemGendary Compiler Presets")
    table.add_column("Preset", style="cyan", no_wrap=True)
    table.add_column("Format", style="green")
    table.add_column("Quality", style="magenta")
    table.add_column("Vetting", style="yellow")
    table.add_column("Labeling", style="blue")
    table.add_column("Description", style="dim")

    for name, p in sorted(all_p.items()):
        table.add_row(
            name,
            p.image_format,
            f"{p.image_quality}%",
            "Enabled" if p.vetting_enabled else "Bypassed",
            "Enabled" if p.labeling_enabled else "Disabled",
            p.description,
        )
    console.print(table)


app.add_typer(presets_app, name="presets")


# ─── audit (Phase 2) ────────────────────────────────────────────────────────
@app.command()
def audit(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key to audit"),
    manifold: str | None = typer.Option(None, "--manifold", help="Direct path to a manifold folder"),
    sample: int | None = typer.Option(None, "--sample", help="Cap image scan for a quick preview"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Full image audit pass (resolution, black-frame, dedup, hardlinks)."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = AuditService.audit_manifold(
        manifold_path=target,
        sample=sample,
        as_json=as_json,
    )
    raise typer.Exit(code=code)


# ─── smart generation (Phase 5) ─────────────────────────────────────────────
@app.command()
def label(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(None, "--manifold", help="Direct path to a manifold folder"),
    strategy: str = typer.Option(
        "blip_caption",
        "--strategy",
        help="blip_caption | clip_zeroshot | yolo_detection | parsenet_segmentation | nima_quality",
    ),
    device: str = typer.Option("cpu", "--device", help="cpu | cuda | cuda:N"),
    sample: int | None = typer.Option(None, "--sample", help="Cap the image scan"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate labels for an existing manifold."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = GenerationService.generate(
        manifold=target,
        kind="label",
        strategy=strategy,
        device=device,
        sample=sample,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


@app.command()
def prompt(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(None, "--manifold", help="Direct path to a manifold folder"),
    template: str = typer.Option(
        "diffusers-v1", "--template", help="diffusers-v1 | sd-v1 | flux-v1 | minimal"
    ),
    device: str = typer.Option("cpu", "--device", help="cpu | cuda | cuda:N"),
    sample: int | None = typer.Option(None, "--sample", help="Cap the image scan"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate structured prompts for diffusion manifolds."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = GenerationService.generate(
        manifold=target,
        kind="prompt",
        strategy=template,
        template=template,
        device=device,
        sample=sample,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


@app.command()
def mask(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(None, "--manifold", help="Direct path to a manifold folder"),
    strategy: str = typer.Option("parsenet", "--strategy", help="parsenet | sam | modnet"),
    device: str = typer.Option("cpu", "--device", help="cpu | cuda | cuda:N"),
    sample: int | None = typer.Option(None, "--sample", help="Cap the image scan"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate masks for a segmentation manifold."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = GenerationService.generate(
        manifold=target,
        kind="mask",
        strategy=strategy,
        device=device,
        sample=sample,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


# ─── degradation engine (Phase 6) ───────────────────────────────────────────
@app.command()
def degrade(
    source: str = typer.Option(..., "--source", "-s", help="Source clean dataset directory or manifold name"),
    output: str = typer.Option(..., "--output", "-o", help="Target synthetic manifold name"),
    profile: str = typer.Option(
        "motion-blur+iso-noise", "--profile", "-p", help="Degradation profile expression or preset alias"
    ),
    intensity: str = typer.Option("medium", "--intensity", help="low | medium | high"),
    pairs: int | None = typer.Option(None, "--pairs", help="Max sample pairs to synthesize"),
    val_split: float = typer.Option(0.12, "--val-split", help="Validation split ratio"),
    seed: int = typer.Option(42, "--seed", help="Deterministic RNG seed"),
    image_format: str = typer.Option("webp", "--image-format", help="webp | jpeg | png"),
    workers: int | None = typer.Option(None, "--workers", "-w", help="Worker thread count"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview plan without writing"),
    no_server: bool = typer.Option(False, "--no-server", help="Bypass API server and execute in-process"),
) -> None:
    """Compiler-time degradation synthesis (blur, noise, haze, rain, jpeg, film)."""
    if not no_server and _is_server_available():
        token = get_or_create_token()
        payload = {
            "source": source,
            "output": output,
            "profile": profile,
            "intensity": intensity,
            "pairs": pairs,
            "val_split": val_split,
            "seed": seed,
            "image_format": image_format,
            "workers": workers,
            "dry_run": dry_run,
        }
        try:
            resp = requests.post(
                f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}/api/jobs/degrade",
                json=payload,
                headers={"X-API-Key": token},
                timeout=10,
            )
            if resp.status_code == 200:
                job_id = resp.json()["id"]
                console.print(
                    f"[bold green]Routed through LemGendary Dataset Compiler API[/bold green] "
                    f"[dim](job_id: {job_id})[/dim]"
                )
                exit_code = _stream_job_logs(job_id)
                raise typer.Exit(code=exit_code)
            console.print(
                f"[yellow]API rejected job ({resp.status_code}): {resp.text}. Falling back to in-process...[/yellow]"
            )
        except requests.RequestException as e:
            console.print(f"[yellow]Could not route to API ({e}). Falling back to in-process...[/yellow]")

    code = DegradeService.run_degrade(
        source=source,
        output=output,
        profile=profile,
        intensity=intensity,
        pairs=pairs,
        val_split=val_split,
        seed=seed,
        image_format=image_format,
        workers=workers,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


# ─── server sub-app (Phase 7) ───────────────────────────────────────────────
@server_app.command("start")
def server_start(
    host: str = typer.Option(DEFAULT_SERVER_HOST, "--host", "-h", help="Bind host address"),
    port: int = typer.Option(DEFAULT_SERVER_PORT, "--port", "-p", help="Bind port number"),
    background: bool = typer.Option(False, "--background", "-d", help="Run server in background daemon process"),
    reload: bool = typer.Option(False, "--reload", help="Enable automatic code reloading"),
) -> None:
    """Start the local HTTP and WebSocket API sidecar server."""
    if _is_server_available(host, port):
        console.print(f"[yellow]Server is already running at http://{host}:{port}[/yellow]")
        return

    if background:
        cmd = [venv_python(), "-m", "api.server"]
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        proc = _spawn_detached(cmd, flags)
        time.sleep(1.5)
        if _is_server_available(host, port):
            console.print(
                f"[bold green]Started LemGendary Dataset Compiler API server in background[/bold green] "
                f"[dim](http://{host}:{port}, PID: {proc.pid})[/dim]"
            )
            console.print(f"[dim]Interactive documentation available at http://{host}:{port}/docs[/dim]")
        else:
            console.print(
                f"[yellow]Server process launched (PID {proc.pid}), still initializing at http://{host}:{port}...[/yellow]"
            )
    else:
        console.print(
            f"[bold green]Starting LemGendary Dataset Compiler API server on http://{host}:{port}...[/bold green]"
        )
        console.print(f"[dim]Interactive documentation available at http://{host}:{port}/docs[/dim]")
        from api.server import run_server
        run_server(host=host, port=port, reload=reload)


@server_app.command("stop")
def server_stop() -> None:
    """Stop the running background API server."""
    pid_file = Path(".lgd_server/server.pid")
    stopped = False
    if pid_file.exists():
        import psutil
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                stopped = True
                console.print(f"[bold green]Terminated server process (PID {pid}).[/bold green]")
            pid_file.unlink()
        except (OSError, ValueError, psutil.Error) as exc:
            logger.warning("Failed stopping server process from %s: %s", pid_file, exc)
            console.print(f"[yellow]Warning while stopping server process: {exc}[/yellow]")

    if not stopped and _is_server_available():
        console.print("[yellow]Server is running under an unmanaged PID. Please terminate via Task Manager.[/yellow]")
    elif stopped:
        console.print("[bold green]LemGendary Dataset Compiler API server stopped.[/bold green]")
    else:
        console.print("[yellow]No active server detected.[/yellow]")


@server_app.command("status")
def server_status(
    host: str = typer.Option(DEFAULT_SERVER_HOST, "--host", "-h"),
    port: int = typer.Option(DEFAULT_SERVER_PORT, "--port", "-p"),
) -> None:
    """Check the health, uptime, and hardware sensors of the API server."""
    if not _is_server_available(host, port):
        console.print(f"[yellow]LemGendary Dataset Compiler API server is not running on http://{host}:{port}[/yellow]")
        return

    try:
        resp = requests.get(f"http://{host}:{port}/api/health/full", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            table = Table(title="LemGendary Dataset Compiler API - Status")
            table.add_column("Property", style="bold cyan")
            table.add_column("Value", style="green")

            table.add_row("Status", data.get("status", "ok"))
            table.add_row("Version", data.get("version", "unknown"))
            table.add_row("Uptime", f"{data.get('uptime_seconds', 0):.1f} s")
            table.add_row("Active Jobs", str(data.get("active_jobs", 0)))

            hw = data.get("hardware", {})
            table.add_row("CPU Cores", str(hw.get("cpu_count", "unknown")))
            table.add_row("RAM (Total / Avail)", f"{hw.get('ram_total_gb', 0)} GB / {hw.get('ram_available_gb', 0)} GB")
            table.add_row("CUDA Acceleration", "Available" if hw.get("cuda_available") else "Disabled (CPU)")
            table.add_row("Primary Device", str(hw.get("device_name", "CPU")))
            table.add_row("Swagger Documentation", f"http://{host}:{port}/docs")

            console.print(table)
        else:
            console.print(f"[red]Health endpoint returned status {resp.status_code}[/red]")
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Error probing API server on %s:%s: %s", host, port, exc)
        console.print(f"[red]Error probing server: {exc}[/red]")


app.add_typer(server_app, name="server")


# ─── forex sub-app ──────────────────────────────────────────────────────────
@forex_app.command("convert")
def forex_convert(
    year: int | None = typer.Option(None, "--year", "-y", help="Specific year to convert (2019..2026)"),
    all_: bool = typer.Option(False, "--all", help="Convert all years 2019..2026 sequentially"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate conversion without writing"),
    skip_cleanup: bool = typer.Option(False, "--skip-cleanup", help="Keep source .npy directories after conversion"),
) -> None:
    """Convert raw .npy Forex manifolds into unified Apache Parquet files."""
    from forex.converter import convert_year
    datasets_root = Path(__file__).resolve().parent.parent / "LemGendaryDatasets"
    base_manifold = (datasets_root / "LemGendizedForexUniverse").resolve()
    if not base_manifold.exists() and (datasets_root / "LemGendizedForexUniverseLarge").exists():
        base_manifold = (datasets_root / "LemGendizedForexUniverseLarge").resolve()
    if not base_manifold.exists():
        console.print(f"[red]Base manifold directory does not exist: {base_manifold}[/red]")
        raise typer.Exit(code=1)

    years = [year] if year else (list(range(2019, 2027)) if all_ else [])
    if not years:
        console.print("[yellow]Specify --year <YYYY> or --all[/yellow]")
        raise typer.Exit(code=0)

    for yr in years:
        ok = convert_year(base_manifold, yr, dry_run=dry_run, skip_cleanup=skip_cleanup)
        if not ok:
            raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@forex_app.command("embed")
def forex_embed() -> None:
    """Embed column descriptions and schema metadata into Forex Parquet files."""
    from forex.injector import main as run_embed
    run_embed()


app.add_typer(forex_app, name="forex")


# ─── modernize sub-app ──────────────────────────────────────────────────────
@modernize_app.command("queue")
def modernize_queue() -> None:
    """Display modernization queue status and pending conversions."""
    from tools.modernization_queue import load_queue_manifest
    datasets_root = Path(__file__).resolve().parent.parent / "LemGendaryDatasets"
    registry_yaml = Path(__file__).resolve().parent / "unified_data.yaml"
    items = load_queue_manifest(datasets_root.resolve(), registry_yaml.resolve())

    table = Table(title="LemGendary Dataset Modernization Queue")
    table.add_column("Dataset Key", style="cyan")
    table.add_column("Folder", style="green")
    table.add_column("Format", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Reason", style="dim")

    for item in items:
        if item.is_containerized:
            status_label, reason = "CONTAINERIZED", f"{item.container_bytes / (1024**3):.1f} GB ready"
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


@modernize_app.command("run")
def modernize_run(
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Specific dataset key to modernize"),
    workers: int | None = typer.Option(None, "--workers", "-w", help="Worker threads for WebP transcoding"),
    push_kaggle: bool = typer.Option(False, "--push-kaggle", help="Push to Kaggle after conversion"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview plan without executing conversions"),
    force: bool = typer.Option(False, "--force", help="Force re-conversion"),
    purge_source: bool = typer.Option(
        False, "--purge-source", help="Delete zip, legacy folder and loose images after a successful conversion"
    ),
) -> None:
    """Run batch modernization queue conversion (same service as POST /api/modernize/run)."""
    code = MigrationService.run_modernization_queue(
        dataset=dataset,
        workers=workers,
        push_kaggle=push_kaggle,
        dry_run=dry_run,
        force=force,
        purge_source=purge_source,
    )
    raise typer.Exit(code=code)


@modernize_app.command("reduce")
def modernize_reduce(
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Manifold folder name to reduce"),
    fraction: float | None = typer.Option(None, "--fraction", help="Share of samples to keep, in (0, 1]"),
    max_gb: float | None = typer.Option(None, "--max-gb", help="Target max size for reduced variant"),
    suffix: str | None = typer.Option(None, "--suffix", help="Suffix for the reduced manifold (default: Reduced)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt; requires --dataset"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report the plan only; write nothing"),
) -> None:
    """Create a downsampled manifold variant (interactive unless --yes)."""
    code = MigrationService.reduce_manifolds(
        max_gb=max_gb, suffix=suffix, dataset=dataset, fraction=fraction, yes=yes, dry_run=dry_run
    )
    raise typer.Exit(code=code)


app.command("reduce")(modernize_reduce)


@modernize_app.command("regen")
def modernize_regen(
    check: bool = typer.Option(False, "--check", help="Dry-run: print summary only"),
) -> None:
    """Regenerate manifolds.md registry index."""
    code = DocService.rebuild_manifolds_md(check_only=check)
    raise typer.Exit(code=code)



# ??? LLM Dataset Sub-Commands ????????????????????????????????????????????????
DEFAULT_JOB_TIMEOUT_S = 3600.0
JOB_POLL_INTERVAL_S = 2.0


def _fetch_job_status(job_id: str, token: str) -> dict:
    """Read one job's status from the sidecar, mapping failures to JobStatusError."""
    url = f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}/api/jobs/{job_id}"
    try:
        resp = requests.get(url, headers={"X-API-Key": token}, timeout=5)
    except requests.RequestException as exc:
        raise JobStatusError(str(exc)) from exc
    if resp.status_code == 404:
        raise JobNotFoundError(job_id)
    if resp.status_code != 200:
        raise JobStatusError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise JobStatusError(f"Invalid JSON in job status: {exc}") from exc


def _print_job_log_tail(job_id: str, token: str, lines: int = 200) -> None:
    """Show the end of the job log so failures are visible without the GUI."""
    url = f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}/api/jobs/{job_id}/logs"
    try:
        resp = requests.get(url, params={"tail": lines}, headers={"X-API-Key": token}, timeout=5)
    except requests.RequestException as exc:
        console.print(f"[yellow]Could not fetch job log: {exc}[/yellow]")
        return
    if resp.status_code != 200:
        console.print(f"[yellow]Could not fetch job log: HTTP {resp.status_code}[/yellow]")
        return
    console.print(resp.json().get("logs", ""), markup=False, highlight=False)


def _wait_for_sidecar_job(job_id: str, token: str, timeout_s: float) -> int:
    """Poll a sidecar job to a terminal state (or timeout) and return the CLI exit code."""
    result = wait_for_job(
        lambda: _fetch_job_status(job_id, token),
        timeout_s=timeout_s,
        poll_interval_s=JOB_POLL_INTERVAL_S,
        on_state=lambda state: console.print(f"[dim]Job {job_id}: {state}[/dim]"),
    )
    if result.state in TERMINAL_STATES:
        _print_job_log_tail(job_id, token)
    if result.exit_code == 0:
        console.print(f"[bold green]Job {job_id} completed.[/bold green]")
    else:
        console.print(f"[bold red]Job {job_id} {result.state}:[/bold red] {result.detail}")
    return result.exit_code


@llm_app.command("compile")
def llm_compile(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset key declared in unified_data.yaml (e.g. test_llm)"),
    no_server: bool = typer.Option(False, "--no-server", help="Bypass API server and execute in-process"),
    timeout: float = typer.Option(
        DEFAULT_JOB_TIMEOUT_S, "--timeout", min=1.0,
        help="Seconds to wait for the sidecar job before giving up (exit 1)",
    ),
) -> None:
    """Compile an LLM instruction manifold using BaseLLMCompiler and unified_data.yaml."""
    if not no_server and _is_server_available():
        token = get_or_create_token()
        payload = {"dataset_key": dataset, "async_mode": True}
        try:
            resp = requests.post(
                f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}/api/llm/compile",
                json=payload,
                headers={"X-API-Key": token},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                job_id = data.get("job_id")
                if not job_id:
                    console.print(f"[bold red]Sidecar accepted the request but returned no job id:[/bold red] {data}")
                    raise typer.Exit(1)
                console.print(f"[green]Enqueued LLM compile job {job_id} on sidecar daemon.[/green]")
                raise typer.Exit(code=_wait_for_sidecar_job(job_id, token, timeout))
            console.print(f"[yellow]API returned {resp.status_code}. Executing in-process...[/yellow]")
        except requests.RequestException as e:
            console.print(f"[yellow]Could not route to API ({e}). Executing in-process...[/yellow]")

    from tools.llm_helper.base_compiler import BaseLLMCompiler
    try:
        compiler = BaseLLMCompiler()
        console.print(f"[bold cyan]Compiling LLM manifold '{dataset}'...[/bold cyan]")
        res = compiler.compile_manifold(dataset)
        console.print(f"[bold green]Successfully compiled {dataset}:[/bold green]")
        console.print(f"  Total records: {res.get('total_records')}")
        console.print(f"  Train shard:   {res.get('train_path')} ({res.get('train_count')} samples)")
        console.print(f"  Val shard:     {res.get('val_path')} ({res.get('val_count')} samples)")
        return
    except typer.Exit:
        raise
    except (KeyError, ValueError, NotImplementedError, OSError, ImportError, RuntimeError, yaml.YAMLError) as exc:
        logger.error("LLM compilation failed for dataset %s: %s", dataset, exc)
        console.print(f"[bold red]LLM compilation error:[/bold red] {exc}")
        console.print_exception()
        raise typer.Exit(1) from exc


@llm_app.command("validate")
def llm_validate(
    file_path: Path = typer.Option(..., "--file", "-f", help="Path to .jsonl dataset file to validate"),
) -> None:
    """Run QualityValidator on an LLM dataset JSONL file for 4 quality invariants."""
    from tools.llm_helper.quality_validator import QualityValidator
    if not file_path.exists():
        console.print(f"[bold red]File not found:[/bold red] {file_path}")
        raise typer.Exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Could not read {file_path}:[/bold red] {exc}")
        raise typer.Exit(1)

    console.print(f"[cyan]Loaded {len(records)} records from {file_path.name}. Running QualityValidator...[/cyan]")
    is_valid, issues = QualityValidator().validate_corpus(records)
    errors = [iss for iss in issues if iss.severity == "ERROR"]
    warnings = [iss for iss in issues if iss.severity == "WARNING"]
    if is_valid:
        console.print(
            f"[bold green][PASS] Quality verification passed with 0 errors "
            f"and {len(warnings)} warnings across {len(records)} records.[/bold green]"
        )
        return

    console.print(f"[bold red][FAIL] Validation failed with {len(errors)} errors and {len(warnings)} warnings.[/bold red]")
    shown = 20
    for iss in (errors + warnings)[:shown]:
        console.print(f"  - {iss.severity} [{iss.field}] ({iss.record_id}): {iss.issue}", markup=False, highlight=False)
    if len(issues) > shown:
        console.print(f"  ... and {len(issues) - shown} more issues.")
    raise typer.Exit(1)


app.add_typer(modernize_app, name="modernize")
app.add_typer(llm_app, name="llm")


# ─── Entry point ────────────────────────────────────────────────────────────

@app.command("test")
def test_datasets(
    pattern: str = typer.Option("test_*.py", "--pattern", "-p", help="Test file pattern to execute"),
) -> None:
    """Run full test suite for the Dataset Compiler Suite in-process using unittest."""
    import unittest
    project_root = Path(__file__).resolve().parent
    tests_dir = project_root / "tests"
    console.print(f"[bold cyan]Running LemGendary Dataset Compiler tests in {tests_dir}...[/bold cyan]")
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        console.print(f"[bold red][FAIL] Tests failed ({len(result.failures)} failures, {len(result.errors)} errors)[/bold red]")
        raise typer.Exit(1)
    console.print(f"[bold green][PASS] All {result.testsRun} tests passed successfully![/bold green]")


if __name__ == "__main__":
    app()
