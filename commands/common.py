"""Helpers shared by the CLI command groups: console, sidecar server probes and job polling."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import requests
import typer
import yaml
from rich.console import Console
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from api.auth import get_or_create_token
from services.job_watch import (
    TERMINAL_STATES,
    JobNotFoundError,
    JobStatusError,
    wait_for_job,
)

logger = logging.getLogger(__name__)
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8100
console = Console()


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


def _stream_job_logs(
    job_id: str, host: str = DEFAULT_SERVER_HOST, port: int = DEFAULT_SERVER_PORT
) -> int:
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


def route_through_sidecar(endpoint: str, payload: dict) -> None:
    """Submit a job to the API sidecar and stream it; return only to fall back in-process.

    Raises ``typer.Exit`` with the job's exit code when the sidecar accepted
    the job. Returns when the sidecar rejected it or could not be reached.
    """
    token = get_or_create_token()
    try:
        resp = requests.post(
            f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}/api/jobs/{endpoint}",
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
            f"[yellow]API rejected job ({resp.status_code}): {resp.text}. Falling back to"
            " in-process...[/yellow]"
        )
    except requests.RequestException as e:
        console.print(
            f"[yellow]Could not route to API ({e}). Falling back to in-process...[/yellow]"
        )


#: Repository root (the parent of this ``commands`` package); paths that the
#: monolithic ``cli.py`` resolved from its own directory resolve from here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Helpers ────────────────────────────────────────────────────────────────
def _run(cmd: list[str]) -> int:
    """Run an external CLI command, print the command, and return its exit code."""
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    try:
        result = subprocess.run(cmd, check=False, cwd=str(PROJECT_ROOT))
        return result.returncode
    except FileNotFoundError as e:
        console.print(f"[red]Command not found:[/red] {e}")
        return 127


def _resolve_manifold_path(model: str) -> Path | None:
    p = PROJECT_ROOT / "unified_data.yaml"
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
        out = (PROJECT_ROOT / out).resolve()
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
