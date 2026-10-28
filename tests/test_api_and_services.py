"""
LemGendary Dataset Compiler — API and Service Orchestration Unit Tests.

Phase 1 of Ecosystem Comprehensive Testing Battery.
Tests FastAPI endpoints (Health, GUI state, Datasets, Gates, Config, Jobs, Modernize, Kaggle)
and all 7 service layers (AuditService, CompilerService, DegradeService, DocService,
GenerationService, MigrationService, SyncService).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from PIL import Image
import yaml

from api.auth import get_or_create_token
from api.server import app
from services.audit_service import AuditService
from services.compiler_service import CompilerService
from services.degrade_service import DegradeService
from services.doc_service import DocService
from services.generation_service import GenerationService
from services.migration_service import MigrationService
from services.sync_service import SyncService


@dataclass
class _AsgiResponse:
    status_code: int
    body_bytes: bytes
    headers: list[tuple[bytes, bytes]]

    def json(self) -> Any:
        return json.loads(self.body_bytes.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.body_bytes.decode("utf-8", errors="replace")


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
        parsed = urlsplit(path)
        raw_headers: list[tuple[bytes, bytes]] = [(b"host", b"testserver")]
        body_bytes = b""
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode("utf-8")
            raw_headers.append((b"content-type", b"application/json"))
        for k, v in (headers or {}).items():
            raw_headers.append((k.lower().encode("utf-8"), v.encode("utf-8")))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode("utf-8"),
            "query_string": parsed.query.encode("utf-8"),
            "headers": raw_headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }

        sent = False

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return {"type": "http.disconnect"}

        out_status = 500
        out_headers: list[tuple[bytes, bytes]] = []
        out_body = bytearray()

        async def send(msg: dict[str, Any]) -> None:
            nonlocal out_status, out_headers
            if msg["type"] == "http.response.start":
                out_status = int(msg["status"])
                out_headers = list(msg.get("headers", []))
            elif msg["type"] == "http.response.body":
                out_body.extend(msg.get("body", b""))

        coro = self._app(scope, receive, send)
        try:
            while True:
                coro.send(None)
        except StopIteration:
            pass

        return _AsgiResponse(
            status_code=out_status,
            body_bytes=bytes(out_body),
            headers=out_headers,
        )

    def get(self, path: str, headers: dict[str, str] | None = None) -> _AsgiResponse:
        return self.request("GET", path, headers=headers)

    def post(
        self,
        path: str,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> _AsgiResponse:
        return self.request("POST", path, json_body=json, headers=headers)


class TestApiRoutes(unittest.TestCase):
    """Test FastAPI sidecar endpoints via InProcessAsgiClient."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = InProcessAsgiClient(app)
        cls.token = get_or_create_token()
        cls.auth_headers = {"X-API-Key": cls.token}

    def test_health_endpoint(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "LemGendary Dataset Compiler API")
        self.assertIn("version", data)
        self.assertIn("uptime_seconds", data)

    def test_health_full_endpoint(self) -> None:
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
        resp = self.client.get("/api/datasets/non_existent_manifold_xyz_123")
        self.assertEqual(resp.status_code, 404)

    def test_gates_hardlinks_endpoint(self) -> None:
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


class TestAuditService(unittest.TestCase):
    """Test AuditService image quality and hardlink verification."""

    def test_audit_manifold_nonexistent(self) -> None:
        ret = AuditService.audit_manifold(Path("C:/does_not_exist_xyz_dir_999"), as_json=True)
        self.assertEqual(ret, 1)

    def test_audit_manifold_valid_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            m_path = Path(tmp_dir) / "TestManifold"
            img_dir = m_path / "images" / "train"
            img_dir.mkdir(parents=True)

            img = Image.new("RGB", (64, 64), color=(128, 64, 32))
            img_file = img_dir / "sample_001.webp"
            img.save(img_file, format="WEBP")

            ret = AuditService.audit_manifold(m_path, sample=5, as_json=True)
            self.assertEqual(ret, 0)


class TestAllServicesOrchestration(unittest.TestCase):
    """Test all 7 service facade layers for in-process dispatch."""

    def test_compiler_service_dry_dispatch(self) -> None:
        with patch("core.manifold_compile.process_dataset") as mock_proc:
            mock_proc.return_value = None
            ret = CompilerService.run_compile(
                preset="restoration-hardlinked",
                no_vetting=True,
                no_labeling=True,
                no_hash=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_proc.called)

    def test_compiler_service_handles_error(self) -> None:
        with patch("core.manifold_compile.process_dataset", side_effect=RuntimeError("Test error")):
            ret = CompilerService.run_compile(
                preset="restoration-hardlinked",
            )
            self.assertEqual(ret, 1)

    def test_migration_service_transcode_dispatch(self) -> None:
        with patch("tools.migrate_manifold_image_format.run", return_value=0) as mock_run:
            ret = MigrationService.transcode_images(
                manifold_path=Path("C:/test_manifold"),
                dry_run=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_run.called)

    def test_migration_service_modernize_dispatch(self) -> None:
        with patch("tools.modernize_manifold.run", return_value=0) as mock_run:
            ret = MigrationService.modernize_manifolds(
                dry_run=True,
                skip_kaggle=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_run.called)

    def test_migration_service_reduce_and_queue_dispatch(self) -> None:
        with patch("tools.manifold_reduce.reduce_dataset") as mock_red:
            ret = MigrationService.reduce_manifolds(max_gb=50.0, suffix="Reduced")
            self.assertEqual(ret, 0)
            mock_red.assert_called_once_with(max_gb_override=50.0, suffix_override="Reduced")

    def test_degrade_service_dispatch(self) -> None:
        with patch("tools.generate_degrade.synthesize_manifold", return_value=0) as mock_deg:
            ret = DegradeService.run_degrade(
                source="clean_src",
                output="synth_out",
                profile="motion-blur+iso-noise",
                dry_run=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_deg.called)

    def test_doc_service_dispatch(self) -> None:
        with patch("tools.regenerate_manifolds_md.main", return_value=0) as mock_md:
            ret = DocService.rebuild_manifolds_md(check_only=True)
            self.assertEqual(ret, 0)
            self.assertTrue(mock_md.called)

    def test_generation_service_dispatch(self) -> None:
        with patch("tools.generate_cli.run", return_value=0) as mock_gen:
            ret = GenerationService.generate(
                manifold=Path("C:/test_manifold"),
                kind="label",
                strategy="blip_caption",
                dry_run=True,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(mock_gen.called)

    def test_sync_service_push_pull_and_metadata_dispatch(self) -> None:
        with patch("tools.manifold_sync.action_sync") as mock_push:
            ret = SyncService.push(model="nima_technical", url=None, no_wait=True)
            self.assertEqual(ret, 0)
            mock_push.assert_called_once_with(manifold_name="nima_technical", repo_id=None, no_wait=True)

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
