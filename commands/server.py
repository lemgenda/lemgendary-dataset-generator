"""`server` sub-commands: start, stop and inspect the API sidecar."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests
import typer
from rich.table import Table

from api.server import run_server
from commands.common import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    _is_server_available,
    _spawn_detached,
    console,
)
from core.cli_args import venv_python

logger = logging.getLogger(__name__)

# Sub-apps
server_app = typer.Typer(help="Control the local FastAPI + WebSocket API sidecar server.")


# ─── server sub-app (Phase 7) ───────────────────────────────────────────────
@server_app.command("start")
def server_start(
    host: str = typer.Option(DEFAULT_SERVER_HOST, "--host", "-h", help="Bind host address"),
    port: int = typer.Option(DEFAULT_SERVER_PORT, "--port", "-p", help="Bind port number"),
    background: bool = typer.Option(
        False, "--background", "-d", help="Run server in background daemon process"
    ),
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
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
        proc = _spawn_detached(cmd, flags)
        time.sleep(1.5)
        if _is_server_available(host, port):
            console.print(
                "[bold green]Started LemGendary Dataset Compiler API server in background[/bold"
                f" green] [dim](http://{host}:{port}, PID: {proc.pid})[/dim]"
            )
            console.print(
                f"[dim]Interactive documentation available at http://{host}:{port}/docs[/dim]"
            )
        else:
            console.print(
                f"[yellow]Server process launched (PID {proc.pid}), still initializing at"
                f" http://{host}:{port}...[/yellow]"
            )
    else:
        console.print(
            "[bold green]Starting LemGendary Dataset Compiler API server on"
            f" http://{host}:{port}...[/bold green]"
        )
        console.print(
            f"[dim]Interactive documentation available at http://{host}:{port}/docs[/dim]"
        )
        run_server(host=host, port=port, reload=reload)


@server_app.command("stop")
def server_stop() -> None:
    """Stop the running background API server."""
    pid_file = Path(".lgd_server/server.pid")
    stopped = False
    if pid_file.exists():
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
        console.print(
            "[yellow]Server is running under an unmanaged PID. Please terminate via Task"
            " Manager.[/yellow]"
        )
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
        console.print(
            "[yellow]LemGendary Dataset Compiler API server is not running on"
            f" http://{host}:{port}[/yellow]"
        )
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
            table.add_row(
                "RAM (Total / Avail)",
                f"{hw.get('ram_total_gb', 0)} GB / {hw.get('ram_available_gb', 0)} GB",
            )
            table.add_row(
                "CUDA Acceleration", "Available" if hw.get("cuda_available") else "Disabled (CPU)"
            )
            table.add_row("Primary Device", str(hw.get("device_name", "CPU")))
            table.add_row("Swagger Documentation", f"http://{host}:{port}/docs")

            console.print(table)
        else:
            console.print(f"[red]Health endpoint returned status {resp.status_code}[/red]")
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Error probing API server on %s:%s: %s", host, port, exc)
        console.print(f"[red]Error probing server: {exc}[/red]")
