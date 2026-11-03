"""
LemGendary Dataset Compiler — API and Service Orchestration Unit Tests.

Phase 1 of Ecosystem Comprehensive Testing Battery.
Tests FastAPI endpoints (Health, GUI state, Datasets, Gates, Config, Jobs, Modernize, Kaggle)
and all 7 services (audit_manifold, run_compile, run_degrade, DocService,
run_generation, MigrationService, SyncService).
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from json import dumps, loads
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

import yaml
from PIL import Image

from api.auth import get_or_create_token
from api.server import app
from services.audit_service import audit_manifold
from services.compiler_service import run_compile
from services.degrade_service import run_degrade
from services.doc_service import DocService
from services.generation_service import run_generation
from services.migration_service import MigrationService
from services.sync_service import SyncService


@dataclass
class _AsgiResponse:
    status_code: int
    body_bytes: bytes
    headers: list[tuple[bytes, bytes]]

    def json(self) -> Any:
        """Decode the body as JSON."""
        return loads(self.body_bytes.decode("utf-8"))

    @property
    def text(self) -> str:
        """Decode the body as UTF-8 text."""
        return self.body_bytes.decode("utf-8", errors="replace")


def _http_scope(method: str, path: str, extra_headers: list[tuple[bytes, bytes]]) -> dict[str, Any]:
    """Build the ASGI HTTP scope for one in-process request."""
    parsed = urlsplit(path)
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": parsed.path,
        "raw_path": parsed.path.encode("utf-8"),
        "query_string": parsed.query.encode("utf-8"),
        "headers": [(b"host", b"testserver"), *extra_headers],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


def _single_body_receiver(body_bytes: bytes) -> Callable[[], Awaitable[dict[str, Any]]]:
    """Return an ASGI ``receive`` that yields the body once, then disconnects."""
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.disconnect"}

    return receive


class _ResponseCollector:
    """ASGI ``send`` sink that accumulates status, headers and body."""

    def __init__(self) -> None:
        self.status = 500
        self.headers: list[tuple[bytes, bytes]] = []
        self.body = bytearray()

    async def send(self, msg: dict[str, Any]) -> None:
        """Record one ASGI response message."""
        if msg["type"] == "http.response.start":
            self.status = int(msg["status"])
            self.headers = list(msg.get("headers", []))
        elif msg["type"] == "http.response.body":
            self.body.extend(msg.get("body", b""))

    def response(self) -> _AsgiResponse:
        """Return the collected response."""
        return _AsgiResponse(
            status_code=self.status, body_bytes=bytes(self.body), headers=self.headers
        )


class InProcessAsgiClient:
    """Zero-socket in-process ASGI client compatible with Windows sandbox environments."""

    def __init__(self, asgi_app: Any) -> None:
        self._app = asgi_app

    def request(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> _AsgiResponse:
        """Send one request through the ASGI app and collect the response."""
        body_bytes = b""
        extra: list[tuple[bytes, bytes]] = []
        if json_body is not None:
            body_bytes = dumps(json_body).encode("utf-8")
            extra.append((b"content-type", b"application/json"))
        for k, v in (headers or {}).items():
            extra.append((k.lower().encode("utf-8"), v.encode("utf-8")))
        collector = _ResponseCollector()
        # A real event loop: sync routes run in the threadpool through anyio.
        asyncio.run(self._app(
            _http_scope(method, path, extra), _single_body_receiver(body_bytes), collector.send
        ))
        return collector.response()

    def get(self, path: str, headers: dict[str, str] | None = None) -> _AsgiResponse:
        """Send a GET request."""
        return self.request("GET", path, headers=headers)

    def post(
        self,
        path: str,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> _AsgiResponse:
        """Send a POST request with an optional JSON body."""
        return self.request("POST", path, json_body=json, headers=headers)


class TestApiRoutes(unittest.TestCase):
    """Test FastAPI sidecar endpoints via InProcessAsgiClient."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = InProcessAsgiClient(app)
        cls.token = get_or_create_token()
        cls.auth_headers = {"X-API-Key": cls.token}

    def test_health_endpoint(self) -> None:
        """GET /api/health reports the service name, status and version."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "LemGendary Dataset Compiler API")
        self.assertIn("version", data)
        self.assertIn("uptime_seconds", data)

    def test_health_full_endpoint(self) -> None:
        """GET /api/health/full includes the hardware block."""
        resp = self.client.get("/api/health/full")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("hardware", data)
        hw = data["hardware"]
        self.assertIn("cpu_count", hw)
        self.assertIn("ram_total_gb", hw)
        self.assertIn("cuda_available", hw)

    def test_gui_state_and_presets_endpoints(self) -> None:
        """GET /api/gui/state lists presets, and the presets endpoint answers."""
        resp = self.client.get("/api/gui/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("service"), "LemGendary Dataset Compiler API")
        self.assertIn("presets", data)
        self.assertIsInstance(data["presets"], list)

        presets_resp = self.client.get("/api/gui/presets")
        self.assertEqual(presets_resp.status_code, 200)
        presets_data = presets_resp.json()
        self.assertIn("presets", presets_data)
        self.assertGreater(presets_data.get("total", 0), 0)

    def test_datasets_list_and_details_endpoints(self) -> None:
        """The dataset list and detail endpoints read a manifold from the output root."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            m_dir = tmp_root / "LemGendizedTestManifold"
            (m_dir / "shards").mkdir(parents=True)
            (m_dir / "shards" / "shard-00000.data").write_bytes(b"0" * 128)
            (m_dir / "dataset_info.yaml").write_text(
                yaml.safe_dump({"task": "restoration", "total_samples": 42}),
                encoding="utf-8",
            )
            (m_dir / "classes.txt").write_text("clean\ndegraded\n", encoding="utf-8")

            with patch("api.routes.datasets._resolve_output_root", return_value=tmp_root):
                resp = self.client.get("/api/datasets")
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["total"], 1)
                ds = data["datasets"][0]
                self.assertEqual(ds["name"], "LemGendizedTestManifold")
                self.assertEqual(ds["sample_count"], 42)
                self.assertIn("wds", ds["formats"])

                detail_resp = self.client.get("/api/datasets/LemGendizedTestManifold")
                self.assertEqual(detail_resp.status_code, 200)
                detail = detail_resp.json()
                self.assertEqual(detail["dataset_info"]["total_samples"], 42)
                self.assertEqual(detail["classes"], ["clean", "degraded"])

    def test_datasets_not_found(self) -> None:
        """An unknown dataset name returns 404."""
        resp = self.client.get("/api/datasets/non_existent_manifold_xyz_123")
        self.assertEqual(resp.status_code, 404)

    def test_gates_hardlinks_endpoint(self) -> None:
        """The hardlink gate endpoint returns a verdict and percentage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            resp = self.client.get(f"/api/gates/hardlinks?path={tmp_dir}")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("verdict", data)
            self.assertIn("hardlink_pct", data)

        resp_err = self.client.get("/api/gates/hardlinks?path=C:/invalid_nonexistent_dir_999")
        self.assertEqual(resp_err.status_code, 404)

    def test_protected_config_and_jobs_endpoints(self) -> None:
        # Unauthenticated request is rejected
        """Config and job endpoints return 401 without a token and succeed with one."""
        unauth = self.client.get("/api/config")
        self.assertEqual(unauth.status_code, 401)

        # Authenticated config fetch & validate
        cfg_resp = self.client.get("/api/config", headers=self.auth_headers)
        self.assertEqual(cfg_resp.status_code, 200)
        self.assertIn("datasets", cfg_resp.json())

        val_resp = self.client.post("/api/config/validate", headers=self.auth_headers)
        self.assertEqual(val_resp.status_code, 200)
        self.assertEqual(val_resp.json().get("status"), "valid")

        jobs_resp = self.client.get("/api/jobs", headers=self.auth_headers)
        self.assertEqual(jobs_resp.status_code, 200)
        self.assertIn("jobs", jobs_resp.json())

    def test_sources_and_kaggle_and_modernize_endpoints(self) -> None:
        """The sources, Kaggle and modernize endpoints answer."""
        src_resp = self.client.get("/api/sources")
        self.assertEqual(src_resp.status_code, 200)
        self.assertIn("total_configured", src_resp.json())

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            with patch("api.routes.kaggle._resolve_output_root", return_value=tmp_root):
                kag_resp = self.client.get("/api/kaggle/registry-datasets")
                self.assertEqual(kag_resp.status_code, 200)
                self.assertGreater(len(kag_resp.json()), 0)

        with patch("api.routes.modernize.load_queue_manifest", return_value=[]):
            mod_resp = self.client.get("/api/modernize/queue")
            self.assertEqual(mod_resp.status_code, 200)
            self.assertEqual(mod_resp.json(), [])


class TestAuditManifold(unittest.TestCase):
    """Test audit_manifold image quality and hardlink verification."""

    def test_audit_manifold_nonexistent(self) -> None:
        """Auditing a missing manifold returns 1."""
        ret = audit_manifold(Path("C:/does_not_exist_xyz_dir_999"), as_json=True)
        self.assertEqual(ret, 1)

    def test_audit_manifold_valid_directory(self) -> None:
        """Auditing a manifold with one valid image succeeds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            m_path = Path(tmp_dir) / "TestManifold"
            img_dir = m_path / "images" / "train"
            img_dir.mkdir(parents=True)

            img = Image.new("RGB", (64, 64), color=(128, 64, 32))
            img_file = img_dir / "sample_001.webp"
            img.save(img_file, format="WEBP")

            ret = audit_manifold(m_path, sample=5, as_json=True)
            self.assertEqual(ret, 0)


class TestAllServicesOrchestration(unittest.TestCase):
    """Test all 7 service facade layers for in-process dispatch."""

    def test_compiler_service_dry_dispatch(self) -> None:
        """run_compile passes its options to process_dataset."""
        with patch("core.manifold_compile.process_dataset") as mock_proc:
            mock_proc.return_value = None
            ret = run_compile(
                preset="restoration-hardlinked",
                no_vetting=True,
                no_labeling=True,
                no_hash=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_proc.called)

    def test_compiler_service_handles_error(self) -> None:
        """run_compile returns 1 when process_dataset raises."""
        with patch("core.manifold_compile.process_dataset", side_effect=RuntimeError("Test error")):
            ret = run_compile(
                preset="restoration-hardlinked",
            )
            self.assertEqual(ret, 1)

    def test_migration_service_transcode_dispatch(self) -> None:
        """MigrationService.transcode_images calls the image-format migration."""
        with patch("tools.migrate_manifold_image_format.run", return_value=0) as mock_run:
            ret = MigrationService.transcode_images(
                manifold_path=Path("C:/test_manifold"),
                dry_run=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_run.called)

    def test_migration_service_modernize_dispatch(self) -> None:
        """MigrationService.modernize_manifolds calls modernize_manifold.run."""
        with patch("tools.modernize_manifold.run", return_value=0) as mock_run:
            ret = MigrationService.modernize_manifolds(
                dry_run=True,
                skip_kaggle=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_run.called)

    def test_migration_service_reduce_and_queue_dispatch(self) -> None:
        """MigrationService dispatches reduce and queue runs with the given options."""
        with patch("tools.manifold_reduce.reduce_dataset", return_value=0) as mock_red:
            ret = MigrationService.reduce_manifolds(max_gb=50.0, suffix="Reduced")
            self.assertEqual(ret, 0)
            mock_red.assert_called_once_with(
                max_gb_override=50.0,
                suffix_override="Reduced",
                dataset=None,
                fraction=None,
                assume_yes=False,
                dry_run=False,
            )

    def test_degrade_service_dispatch(self) -> None:
        """run_degrade calls synthesize_manifold."""
        with patch("tools.generate_degrade.synthesize_manifold", return_value=0) as mock_deg:
            ret = run_degrade(
                source="clean_src",
                output="synth_out",
                profile="motion-blur+iso-noise",
                dry_run=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_deg.called)

    def test_doc_service_dispatch(self) -> None:
        """DocService.rebuild_manifolds_md calls regenerate_manifolds_md.main."""
        with patch("tools.regenerate_manifolds_md.main", return_value=0) as mock_md:
            ret = DocService.rebuild_manifolds_md(check_only=True)
            self.assertEqual(ret, 0)
            self.assertTrue(mock_md.called)

    def test_generation_service_dispatch(self) -> None:
        """run_generation calls generate_cli.run."""
        with patch("tools.generate_cli.run", return_value=0) as mock_gen:
            ret = run_generation(
                manifold=Path("C:/test_manifold"),
                kind="label",
                strategy="blip_caption",
                dry_run=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_gen.called)

    def test_sync_service_push_pull_and_metadata_dispatch(self) -> None:
        """SyncService dispatches push, pull and metadata calls to the sync tools."""
        with patch("tools.manifold_sync.action_sync") as mock_push:
            ret = SyncService.push(model="nima_technical", url=None, no_wait=True)
            self.assertEqual(ret, 0)
            mock_push.assert_called_once_with(
                manifold_name="nima_technical", repo_id=None, no_wait=True
            )

        with patch("tools.manifold_sync.action_get") as mock_pull:
            ret = SyncService.pull(url="lemtreursi/lemgendizednimatechnical")
            self.assertEqual(ret, 0)
            mock_pull.assert_called_once()

        with patch("tools.sync_kaggle_metadata.sync_metadata", return_value=0) as mock_meta:
            ret = SyncService.push_metadata(model="nima_technical", dry_run=True)
            self.assertEqual(ret, 0)
            mock_meta.assert_called_once()


if __name__ == "__main__":
    unittest.main()
