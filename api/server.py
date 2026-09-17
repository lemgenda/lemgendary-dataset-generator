"""FastAPI application entry point and server startup engine."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.auth import verify_token
from api.events import drain_event_queue, manager
from api.routes.config import router as config_router
from api.routes.datasets import router as datasets_router
from api.routes.env import router as env_router
from api.routes.gates import router as gates_router
from api.routes.health import router as health_router
from api.routes.jobs import router as jobs_router, ws_router as jobs_ws_router
from api.routes.kaggle import router as kaggle_router
from api.routes.sources import router as sources_router
from cli_args import __version__

logger = logging.getLogger("lemgendary.api.server")
_PID_FILE = Path(".lgd_server/server.pid")


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager controlling event queue draining and server lifecycle."""
    del app_instance
    drain_task = asyncio.create_task(drain_event_queue())
    _write_pid()
    logger.info("LemGendary Dataset Compiler API server initialized (version: %s)", __version__)
    try:
        yield
    finally:
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            logger.debug("Drain task cancelled on shutdown")
        _remove_pid()
        logger.info("LemGendary Dataset Compiler API server shut down")


def _write_pid() -> None:
    try:
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write PID file %s: %s", _PID_FILE, exc)


def _remove_pid() -> None:
    try:
        if _PID_FILE.exists():
            _PID_FILE.unlink()
    except OSError as exc:
        logger.warning("Failed to remove PID file %s: %s", _PID_FILE, exc)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app_instance = FastAPI(
        title="LemGendary Dataset Compiler API",
        version=__version__,
        description="REST and WebSocket sidecar service for LemGendary AI Studio.",
        lifespan=lifespan,
    )

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Public diagnostic and inspection routers
    app_instance.include_router(health_router, prefix="/api")
    app_instance.include_router(datasets_router, prefix="/api")
    app_instance.include_router(sources_router, prefix="/api")
    app_instance.include_router(gates_router, prefix="/api")
    app_instance.include_router(jobs_ws_router, prefix="/api")

    # Protected operational routers requiring API token authentication
    app_instance.include_router(config_router, prefix="/api", dependencies=[Depends(verify_token)])
    app_instance.include_router(jobs_router, prefix="/api", dependencies=[Depends(verify_token)])
    app_instance.include_router(kaggle_router, prefix="/api", dependencies=[Depends(verify_token)])
    app_instance.include_router(env_router, prefix="/api", dependencies=[Depends(verify_token)])

    @app_instance.websocket("/api/ws/events")
    async def global_events_stream(websocket: WebSocket) -> None:
        """Global WebSocket event broadcast stream for real-time dashboard listeners."""
        await manager.connect_global(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await manager.disconnect_global(websocket)
        except Exception:
            await manager.disconnect_global(websocket)

    return app_instance


app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 8100, reload: bool = False) -> None:
    """Launch the uvicorn ASGI server with the dataset compiler API application."""
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
