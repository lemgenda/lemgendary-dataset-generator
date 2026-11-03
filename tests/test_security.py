"""
LemGendary Dataset Compiler — browser-origin guard tests.

Checks the shared security contract: only the AI Studio GUI origins may call
the sidecar from a browser, browser writes need the client header, and
non-browser clients (no Origin header) keep working. The API token check
still applies behind the guard.
"""

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.auth import API_KEY_HEADER
from api.security import (
    ALLOWED_ORIGINS,
    CLIENT_HEADER,
    CLIENT_HEADER_VALUE,
    origin_guard_middleware,
)
from api.server import app as server_app

GUI_ORIGIN = "http://tauri.localhost"
FOREIGN_ORIGIN = "https://evil.example"
TEST_TOKEN = "test-token-for-origin-guard"


def _build_app() -> FastAPI:
    """Small app wired exactly like api.server.create_app()."""
    test_app = FastAPI()

    @test_app.get("/read")
    async def read() -> dict:
        return {"ok": True}

    @test_app.post("/write")
    async def write() -> dict:
        return {"ok": True}

    @test_app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"ok": True})
        await websocket.close()

    # Copy the real app's middleware stack. add_middleware() prepends, so walk it
    # innermost-first to keep the guard outermost, exactly as in the server.
    for middleware in reversed(server_app.user_middleware):
        test_app.add_middleware(middleware.cls, *middleware.args, **middleware.kwargs)
    return test_app


class TestOriginGuard(unittest.TestCase):
    """Behaviour of origin_guard_middleware together with CORS."""

    def setUp(self) -> None:
        self.client = TestClient(_build_app())

    def test_no_origin_read_allowed(self) -> None:
        """A read without an Origin header is allowed."""
        self.assertEqual(self.client.get("/read").status_code, 200)

    def test_no_origin_write_allowed(self) -> None:
        """A write without an Origin header is allowed."""
        self.assertEqual(self.client.post("/write").status_code, 200)

    def test_gui_origin_read_allowed(self) -> None:
        """A read from the GUI origin is allowed and gets the CORS header."""
        resp = self.client.get("/read", headers={"Origin": GUI_ORIGIN})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("access-control-allow-origin"), GUI_ORIGIN)

    def test_foreign_origin_read_refused(self) -> None:
        """A read from a foreign origin is refused with 403."""
        resp = self.client.get("/read", headers={"Origin": FOREIGN_ORIGIN})
        self.assertEqual(resp.status_code, 403)
        self.assertIn("not allowed", resp.json()["detail"])

    def test_null_origin_refused(self) -> None:
        """The "null" origin is refused."""
        resp = self.client.get("/read", headers={"Origin": "null"})
        self.assertEqual(resp.status_code, 403)

    def test_gui_origin_write_without_header_refused(self) -> None:
        """A GUI-origin write without the client header is refused."""
        resp = self.client.post("/write", headers={"Origin": GUI_ORIGIN})
        self.assertEqual(resp.status_code, 403)
        self.assertIn(CLIENT_HEADER, resp.json()["detail"])

    def test_gui_origin_write_with_wrong_header_refused(self) -> None:
        """A GUI-origin write with the wrong client header value is refused."""
        resp = self.client.post(
            "/write", headers={"Origin": GUI_ORIGIN, CLIENT_HEADER: "other"}
        )
        self.assertEqual(resp.status_code, 403)

    def test_gui_origin_write_with_header_allowed(self) -> None:
        """A GUI-origin write with the client header is allowed."""
        resp = self.client.post(
            "/write", headers={"Origin": GUI_ORIGIN, CLIENT_HEADER: CLIENT_HEADER_VALUE}
        )
        self.assertEqual(resp.status_code, 200)

    def test_foreign_origin_write_with_header_refused(self) -> None:
        """A foreign-origin write is refused even with the client header."""
        resp = self.client.post(
            "/write",
            headers={"Origin": FOREIGN_ORIGIN, CLIENT_HEADER: CLIENT_HEADER_VALUE},
        )
        self.assertEqual(resp.status_code, 403)

    def test_preflight_from_gui_allows_client_and_token_headers(self) -> None:
        """A GUI preflight allows the client and token headers."""
        resp = self.client.options(
            "/write",
            headers={
                "Origin": GUI_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": f"{CLIENT_HEADER}, {API_KEY_HEADER}",
            },
        )
        self.assertEqual(resp.status_code, 200)
        allowed = resp.headers.get("access-control-allow-headers", "").lower()
        self.assertIn(CLIENT_HEADER.lower(), allowed)
        self.assertIn(API_KEY_HEADER.lower(), allowed)

    def test_preflight_from_foreign_origin_refused(self) -> None:
        """A preflight from a foreign origin is refused."""
        resp = self.client.options(
            "/write",
            headers={
                "Origin": FOREIGN_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(resp.status_code, 403)

    def test_websocket_gui_origin_allowed(self) -> None:
        """A WebSocket from the GUI origin is accepted."""
        with self.client.websocket_connect("/ws", headers={"Origin": GUI_ORIGIN}) as ws:
            self.assertEqual(ws.receive_json(), {"ok": True})

    def test_websocket_no_origin_allowed(self) -> None:
        """A WebSocket without an Origin header is accepted."""
        with self.client.websocket_connect("/ws") as ws:
            self.assertEqual(ws.receive_json(), {"ok": True})

    def test_websocket_foreign_origin_refused(self) -> None:
        """A WebSocket from a foreign origin is closed."""
        with (
            self.assertRaises(WebSocketDisconnect),
            self.client.websocket_connect("/ws", headers={"Origin": FOREIGN_ORIGIN}) as ws,
        ):
            ws.receive_json()


class TestServerWiring(unittest.TestCase):
    """The real sidecar app uses the guard and a closed CORS allow-list."""

    def test_server_middleware_configuration(self) -> None:
        """The origin guard is the outermost middleware and CORS uses an allow-list, no wildcard."""
        classes = [m.cls for m in server_app.user_middleware]
        self.assertIn(origin_guard_middleware, classes)
        self.assertIn(CORSMiddleware, classes)
        # user_middleware lists the outermost middleware first.
        self.assertIs(classes[0], origin_guard_middleware)

        cors = next(m for m in server_app.user_middleware if m.cls is CORSMiddleware)
        self.assertEqual(list(cors.kwargs["allow_origins"]), list(ALLOWED_ORIGINS))
        self.assertNotIn("*", cors.kwargs["allow_origins"])
        self.assertFalse(cors.kwargs["allow_credentials"])
        self.assertIn(API_KEY_HEADER, cors.kwargs["allow_headers"])

    def test_real_app_refuses_foreign_origin(self) -> None:
        # No context manager: the lifespan (PID file, event drain) does not run.
        """The real app refuses a foreign-origin POST."""
        client = TestClient(server_app)
        resp = client.post("/api/gui/quick-compile", headers={"Origin": FOREIGN_ORIGIN})
        self.assertEqual(resp.status_code, 403)

    def test_real_app_websocket_refuses_foreign_origin(self) -> None:
        """The real app closes a foreign-origin WebSocket."""
        client = TestClient(server_app)
        with (
            self.assertRaises(WebSocketDisconnect),
            client.websocket_connect(
                "/api/ws/events", headers={"Origin": FOREIGN_ORIGIN}
            ) as ws,
        ):
            ws.receive_text()

    def test_real_app_token_check_still_applies_behind_guard(self) -> None:
        """Behind the origin guard, a missing token gives 401 and a wrong one 403."""
        client = TestClient(server_app)
        gui_headers = {"Origin": GUI_ORIGIN, CLIENT_HEADER: CLIENT_HEADER_VALUE}
        env = {"LEMGENDARY_API_TOKEN": TEST_TOKEN, "LEMGENDARY_DISABLE_AUTH": ""}
        with patch.dict(os.environ, env):
            missing = client.get("/api/config", headers=gui_headers)
            wrong = client.get(
                "/api/config", headers={**gui_headers, API_KEY_HEADER: "wrong"}
            )
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 403)
        self.assertEqual(missing.headers.get("access-control-allow-origin"), GUI_ORIGIN)


if __name__ == "__main__":
    unittest.main()
