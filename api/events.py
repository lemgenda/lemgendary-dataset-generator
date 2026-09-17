"""WebSocket connection manager and event broadcasting engine."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from fastapi import WebSocket

logger = logging.getLogger("lemgendary.api.events")


class ConnectionManager:
    """Manages active WebSocket connections for broadcasts and job-specific log streams."""

    def __init__(self) -> None:
        self.global_connections: List[WebSocket] = []
        self.job_connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect_global(self, websocket: WebSocket) -> None:
        """Register a WebSocket client for global server event broadcasts."""
        await websocket.accept()
        async with self._lock:
            self.global_connections.append(websocket)
        logger.debug("WebSocket client registered for global events (total: %d)", len(self.global_connections))

    async def disconnect_global(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket client from global server broadcasts."""
        async with self._lock:
            if websocket in self.global_connections:
                self.global_connections.remove(websocket)
        logger.debug("WebSocket client removed from global events (total: %d)", len(self.global_connections))

    async def connect_job(self, job_id: str, websocket: WebSocket) -> None:
        """Register a WebSocket client for streaming logs of a specific job."""
        await websocket.accept()
        async with self._lock:
            if job_id not in self.job_connections:
                self.job_connections[job_id] = []
            self.job_connections[job_id].append(websocket)
        logger.debug("WebSocket client registered for job %s stream (subscribers: %d)", job_id, len(self.job_connections[job_id]))

    async def disconnect_job(self, job_id: str, websocket: WebSocket) -> None:
        """Unregister a WebSocket client from streaming logs of a specific job."""
        async with self._lock:
            if job_id in self.job_connections and websocket in self.job_connections[job_id]:
                self.job_connections[job_id].remove(websocket)
                if not self.job_connections[job_id]:
                    del self.job_connections[job_id]
        logger.debug("WebSocket client disconnected from job %s", job_id)

    async def broadcast_global(self, message: Dict[str, Any]) -> None:
        """Send a JSON payload to all active global clients."""
        async with self._lock:
            targets = list(self.global_connections)
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception as exc:
                logger.debug("Error sending broadcast to websocket: %s", exc)
                await self.disconnect_global(ws)

    async def broadcast_job_log(self, job_id: str, line: str) -> None:
        """Stream a log chunk to all clients subscribed to a specific job."""
        async with self._lock:
            targets = list(self.job_connections.get(job_id, []))
        payload = {"job_id": job_id, "type": "log", "chunk": line}
        for ws in targets:
            try:
                await ws.send_json(payload)
            except Exception as exc:
                logger.debug("Error streaming log to websocket for job %s: %s", job_id, exc)
                await self.disconnect_job(job_id, ws)


manager = ConnectionManager()
event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)


async def drain_event_queue() -> None:
    """Continuous background worker draining event queue into global broadcast manager."""
    try:
        while True:
            ev = await event_queue.get()
            await manager.broadcast_global(ev)
            event_queue.task_done()
    except asyncio.CancelledError:
        logger.debug("Event queue drain loop cancelled")
        raise
