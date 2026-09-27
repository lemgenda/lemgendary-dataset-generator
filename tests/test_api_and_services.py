"""
LemGendary Dataset Compiler — API and Service Orchestration Unit Tests.

Phase 1 of Ecosystem Comprehensive Testing Battery.
Tests FastAPI endpoints (Health, GUI state, Datasets, Gates) via TestClient,
and service layers (AuditService, CompilerService, MigrationService).
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from api.server import app
from services.audit_service import AuditService
from services.compiler_service import CompilerService
from services.migration_service import MigrationService


class TestApiRoutes(unittest.TestCase):
    """Test FastAPI public sidecar endpoints via TestClient."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

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

    def test_gui_state_endpoint(self) -> None:
        resp = self.client.get("/api/gui/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("service"), "LemGendary Dataset Compiler API")
        self.assertIn("presets", data)
        self.assertIsInstance(data["presets"], list)

    def test_datasets_list_endpoint(self) -> None:
        resp = self.client.get("/api/datasets")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("datasets", data)
        self.assertIn("total", data)

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

        # Non-existent path returns 404
        resp_err = self.client.get("/api/gates/hardlinks?path=C:/invalid_nonexistent_dir_999")
        self.assertEqual(resp_err.status_code, 404)


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

            # Create synthetic WebP image
            img = Image.new("RGB", (64, 64), color=(128, 64, 32))
            img_file = img_dir / "sample_001.webp"
            img.save(img_file, format="WEBP")

            ret = AuditService.audit_manifold(m_path, sample=5, as_json=True)
            self.assertEqual(ret, 0)


class TestCompilerAndMigrationServices(unittest.TestCase):
    """Test service facade layer orchestration."""

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


if __name__ == "__main__":
    unittest.main()
