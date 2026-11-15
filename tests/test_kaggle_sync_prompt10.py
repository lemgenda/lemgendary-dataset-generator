"""Unit, integration, and failure-injection tests for Kaggle push/pull hardening.

Tests cover:
  1. Deterministic manifest computation and per-file SHA-256 + Merkle root verification.
  2. Sidecar `.data.index.json` byte-size integrity checks.
  3. Idempotent push (unchanged manifold is a no-op unless ``--force``).
  4. Failure-injection on interrupted upload (retry/backoff + staging archive resumption).
  5. Sandbox pull with manifest checksum verification (intact vs tampered archive).
  6. Version pinning (``--version``, ``owner/slug@<version>``, ``owner/slug/versions/<N>``).
  7. Structured machine-readable ``SyncReport`` output (``--report`` and ``--json``).
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from cli import app as cli_app
from core.common_sync import (
    DownloadRequest,
    SyncReport,
    compute_sync_manifest,
    parse_version_pin,
    perform_dataset_download,
    perform_dataset_upload,
    resolve_pinned_version,
    verify_sync_manifest,
)
from tests.tar_fixtures import write_tar


def _build_sandbox_manifold(manifold_dir: Path, sample_payload: bytes = b"webp_bytes_001") -> Path:
    """Create a realistic sandbox manifold with shards, index sidecar, and metadata."""
    manifold_dir.mkdir(parents=True, exist_ok=True)
    shard_path = manifold_dir / "shards" / "train" / "shard-00000.data"
    write_tar(shard_path, [("sample_0001.webp", sample_payload), ("sample_0001.json", b"{}")])
    sidecar_path = manifold_dir / "shards" / "train" / "shard-00000.data.index.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "shard": "shard-00000.data",
                "size": shard_path.stat().st_size,
                "sample_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (manifold_dir / "index.json").write_text(
        json.dumps([{"name": "sample_0001", "split": "train"}]),
        encoding="utf-8",
    )
    (manifold_dir / "dataset_info.yaml").write_text("count: 1\n", encoding="utf-8")
    return manifold_dir


class SyncManifestTests(unittest.TestCase):
    """Per-file SHA-256 and Merkle root verification tests."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sync_manifest_test_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_deterministic_manifest_and_verification_pass(self) -> None:
        """Computing a manifest twice on unchanged files produces the identical Merkle root."""
        manifold = _build_sandbox_manifold(self.root / "LemGendizedSandbox")
        m1 = compute_sync_manifest(manifold, write_file=True)
        m2 = compute_sync_manifest(manifold, write_file=True)
        self.assertEqual(m1.manifest_sha256, m2.manifest_sha256)
        self.assertEqual(m1.total_files, 4)
        self.assertTrue((manifold / "sync_manifest.json").is_file())

        ok, root_sha, errors = verify_sync_manifest(manifold, expected_sha256=m1.manifest_sha256)
        self.assertTrue(ok, errors)
        self.assertEqual(root_sha, m1.manifest_sha256)
        self.assertEqual(errors, [])

    def test_verification_detects_corrupted_shard_bytes(self) -> None:
        """A single bit flip inside a shard fails SHA-256 verification."""
        manifold = _build_sandbox_manifold(self.root / "LemGendizedSandbox")
        compute_sync_manifest(manifold, write_file=True)
        shard = manifold / "shards" / "train" / "shard-00000.data"
        raw = bytearray(shard.read_bytes())
        raw[-1] ^= 0xFF
        shard.write_bytes(bytes(raw))

        ok, _, errors = verify_sync_manifest(manifold)
        self.assertFalse(ok)
        self.assertTrue(any("SHA-256 mismatch" in e for e in errors), errors)

    def test_verification_detects_missing_file(self) -> None:
        """Removing a listed file fails manifest verification."""
        manifold = _build_sandbox_manifold(self.root / "LemGendizedSandbox")
        compute_sync_manifest(manifold, write_file=True)
        (manifold / "dataset_info.yaml").unlink()

        ok, _, errors = verify_sync_manifest(manifold)
        self.assertFalse(ok)
        self.assertTrue(
            any("Missing file from manifest: dataset_info.yaml" in e for e in errors), errors
        )

    def test_verification_detects_sidecar_index_size_mismatch(self) -> None:
        """Mismatch between .data.index.json recorded size and actual shard size is flagged."""
        manifold = _build_sandbox_manifold(self.root / "LemGendizedSandbox")
        sidecar = manifold / "shards" / "train" / "shard-00000.data.index.json"
        sidecar.write_text(json.dumps({"size": 999999}), encoding="utf-8")
        compute_sync_manifest(manifold, write_file=True)

        ok, _, errors = verify_sync_manifest(manifold)
        self.assertFalse(ok)
        self.assertTrue(any("Sidecar index size mismatch" in e for e in errors), errors)

    def test_partial_only_verification_checks_subset(self) -> None:
        """When --only is used, verification checks only files matching the requested prefixes."""
        manifold = _build_sandbox_manifold(self.root / "LemGendizedSandbox")
        compute_sync_manifest(manifold, write_file=True)
        # Simulate partial download where shards/ was not fetched, only dataset_info.yaml
        shutil.rmtree(manifold / "shards")

        ok_full, _, _ = verify_sync_manifest(manifold)
        self.assertFalse(ok_full)

        ok_sub, _, errs_sub = verify_sync_manifest(manifold, only=("dataset_info.yaml",))
        self.assertTrue(ok_sub, errs_sub)


class IdempotentPushAndRetryTests(unittest.TestCase):
    """Idempotent push detection and failure-injection upload tests."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sync_push_test_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_idempotent_push_skips_second_upload_unless_forced(self) -> None:
        """Second push of an unchanged manifold is a no-op; --force or file change re-uploads."""
        manifold = _build_sandbox_manifold(self.root / "LemGendizedSandbox")
        report1 = SyncReport()
        with patch("core.sync_upload.ensure_kaggle_dataset_exists", return_value=True), \
                patch("core.sync_upload.get_dataset_version_info",
                      return_value={"latest_version": 2}), \
                patch("core.sync_upload.kagglehub.dataset_upload") as upload, \
                redirect_stdout(io.StringIO()):
            ok1 = perform_dataset_upload(
                manifold, "owner/sandbox", no_wait=True, report_out=report1
            )
        self.assertTrue(ok1)
        self.assertEqual(upload.call_count, 1)
        self.assertEqual(report1.status, "passed")
        self.assertFalse(report1.idempotent_noop)
        self.assertTrue(report1.checksum_verified)
        self.assertTrue((manifold / ".sync_state.json").is_file())

        # Second push without changes -> idempotent no-op
        report2 = SyncReport()
        with patch("core.sync_upload.ensure_kaggle_dataset_exists", return_value=True), \
                patch("core.sync_upload.get_dataset_version_info",
                      return_value={"latest_version": 3}), \
                patch("core.sync_upload.kagglehub.dataset_upload") as upload2, \
                redirect_stdout(io.StringIO()):
            ok2 = perform_dataset_upload(
                manifold, "owner/sandbox", no_wait=True, report_out=report2
            )
        self.assertTrue(ok2)
        upload2.assert_not_called()
        self.assertEqual(report2.status, "noop")
        self.assertTrue(report2.idempotent_noop)
        self.assertEqual(report2.manifest_sha256, report1.manifest_sha256)

        # Third push with force=True -> uploads again
        report3 = SyncReport()
        with patch("core.sync_upload.ensure_kaggle_dataset_exists", return_value=True), \
                patch("core.sync_upload.get_dataset_version_info",
                      return_value={"latest_version": 3}), \
                patch("core.sync_upload.kagglehub.dataset_upload") as upload3, \
                redirect_stdout(io.StringIO()):
            ok3 = perform_dataset_upload(
                manifold, "owner/sandbox", no_wait=True, force=True, report_out=report3
            )
        self.assertTrue(ok3)
        self.assertEqual(upload3.call_count, 1)
        self.assertEqual(report3.status, "passed")
        self.assertFalse(report3.idempotent_noop)

    def test_failure_injection_interrupted_upload_retries_and_recovers(self) -> None:
        """Transient ConnectionError on attempt 1 retries with backoff and succeeds on attempt 2."""
        manifold = _build_sandbox_manifold(self.root / "LemGendizedSandbox")
        attempts = 0

        def _flaky_upload(*_args: object, **_kwargs: object) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("Simulated network drop at 60% upload")

        sleeps: list[float] = []
        report = SyncReport()
        with patch("core.sync_upload.ensure_kaggle_dataset_exists", return_value=True), \
                patch("core.sync_upload.get_dataset_version_info",
                      return_value={"latest_version": 1}), \
                patch("core.sync_upload.kagglehub.dataset_upload", side_effect=_flaky_upload), \
                redirect_stdout(io.StringIO()):
            ok = perform_dataset_upload(
                manifold,
                "owner/sandbox",
                no_wait=True,
                max_retries=3,
                retry_backoff_s=0.25,
                sleep=sleeps.append,
                report_out=report,
            )
        self.assertTrue(ok)
        self.assertEqual(attempts, 2)
        self.assertEqual(report.retries, 1)
        self.assertEqual(sleeps, [0.25])
        self.assertEqual(report.status, "passed")

    def test_failure_injection_exhausted_upload_preserves_staging_zip_for_resume(self) -> None:
        """When all upload retries fail, staging zip is kept and reused on the next invocation."""
        manifold = _build_sandbox_manifold(self.root / "LemGendizedSandbox")
        report_fail = SyncReport()
        with patch("core.sync_upload.ensure_kaggle_dataset_exists", return_value=True), \
                patch("core.sync_upload.get_dataset_version_info",
                      return_value={"latest_version": 1}), \
                patch("core.sync_upload.kagglehub.dataset_upload",
                      side_effect=ConnectionError("Persistent outage")), \
                redirect_stdout(io.StringIO()):
            ok_fail = perform_dataset_upload(
                manifold,
                "owner/sandbox",
                no_wait=True,
                max_retries=2,
                retry_backoff_s=0.01,
                sleep=lambda _: None,
                report_out=report_fail,
            )
        self.assertFalse(ok_fail)
        self.assertEqual(report_fail.status, "failed")
        self.assertEqual(report_fail.retries, 1)

        staged_zip = (
            self.root / ".staging_LemGendizedSandbox" / "LemGendizedSandbox.zip"
        )
        self.assertTrue(staged_zip.is_file(), "Staged archive must be preserved for resumption")

        # Now resume: network recovers, staged archive must be reused without re-zipping
        report_resume = SyncReport()
        with patch("core.sync_upload.ensure_kaggle_dataset_exists", return_value=True), \
                patch("core.sync_upload.get_dataset_version_info",
                      return_value={"latest_version": 1}), \
                patch("core.sync_upload.kagglehub.dataset_upload") as upload_ok, \
                patch("core.sync_upload.create_archive") as rezip_spy, \
                redirect_stdout(io.StringIO()):
            ok_resume = perform_dataset_upload(
                manifold,
                "owner/sandbox",
                no_wait=True,
                report_out=report_resume,
            )
        self.assertTrue(ok_resume)
        rezip_spy.assert_not_called()
        upload_ok.assert_called_once()
        self.assertTrue(report_resume.resumed)
        self.assertEqual(report_resume.status, "passed")


class SandboxPullAndVersionPinTests(unittest.TestCase):
    """Sandbox dataset pull, manifest checksum verification, and version pinning tests."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sync_pull_test_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _create_remote_sandbox_zip(self, zip_path: Path, *, tamper: bool = False) -> str:
        """Build a zip archive from a sandbox manifold with sync_manifest.json."""
        src_dir = self.root / "remote_build" / "LemGendizedSandbox"
        _build_sandbox_manifold(src_dir)
        manifest = compute_sync_manifest(src_dir, write_file=True)
        if tamper:
            # Corrupt shard bytes AFTER sync_manifest.json was signed
            shard = src_dir / "shards" / "train" / "shard-00000.data"
            shard.write_bytes(shard.read_bytes() + b"CORRUPTED_TAIL")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for file_path in src_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(src_dir))
        shutil.rmtree(src_dir.parent, ignore_errors=True)
        return manifest.manifest_sha256

    def test_sandbox_pull_verifies_manifest_checksum(self) -> None:
        """Pulling a valid sandbox archive verifies all per-file SHA-256 digests."""
        datasets_root = self.root / "datasets"
        target_dir = datasets_root / "LemGendizedSandbox"
        archive_path = datasets_root / "sandbox.zip"
        expected_sha = self._create_remote_sandbox_zip(archive_path, tamper=False)

        req = DownloadRequest(
            clean_repo_id="owner/sandbox",
            target_dir=target_dir,
            root_datasets_dir=datasets_root,
            manifold_name="LemGendizedSandbox",
            slug="sandbox",
        )
        report = SyncReport()
        with redirect_stdout(io.StringIO()):
            ok = perform_dataset_download(req, report_out=report)
        self.assertTrue(ok)
        self.assertEqual(report.status, "passed")
        self.assertTrue(report.checksum_verified)
        self.assertEqual(report.manifest_sha256, expected_sha)
        self.assertTrue(report.resumed)

    def test_sandbox_pull_rejects_tampered_archive(self) -> None:
        """Pulling an archive whose payload does not match sync_manifest.json fails cleanly."""
        datasets_root = self.root / "datasets"
        target_dir = datasets_root / "LemGendizedSandbox"
        archive_path = datasets_root / "sandbox.zip"
        self._create_remote_sandbox_zip(archive_path, tamper=True)

        req = DownloadRequest(
            clean_repo_id="owner/sandbox",
            target_dir=target_dir,
            root_datasets_dir=datasets_root,
            manifold_name="LemGendizedSandbox",
            slug="sandbox",
        )
        report = SyncReport()
        with redirect_stdout(io.StringIO()):
            ok = perform_dataset_download(req, report_out=report)
        self.assertFalse(ok)
        self.assertEqual(report.status, "failed")
        self.assertFalse(report.checksum_verified)
        self.assertTrue(report.errors)

    def test_version_pin_parsing_and_resolution(self) -> None:
        """parse_version_pin and resolve_pinned_version handle @vN, /versions/N, and latest."""
        self.assertEqual(parse_version_pin("owner/slug@3"), ("owner/slug", "3"))
        self.assertEqual(parse_version_pin("owner/slug@v4"), ("owner/slug", "v4"))
        self.assertEqual(parse_version_pin("owner/slug/versions/5"), ("owner/slug", "5"))
        self.assertEqual(parse_version_pin("owner/slug", 2), ("owner/slug", 2))

        fake_info = {"latest_version": 3, "versions": {1: "READY", 2: "READY", 3: "READY"}}
        with patch("core.sync_remote.get_dataset_version_info", return_value=fake_info):
            self.assertEqual(resolve_pinned_version("owner/slug", "v2"), 2)
            self.assertEqual(resolve_pinned_version("owner/slug", "latest"), 3)
            with self.assertRaises(ValueError):
                resolve_pinned_version("owner/slug", 99)

    def test_cli_pull_with_version_and_report_file(self) -> None:
        """`sync pull --version 2 --report <path> --json` pins v2 and writes JSON report."""
        datasets_root = self.root / "datasets"
        report_file = self.root / "pull_report.json"
        fake_info = {"latest_version": 3, "versions": {1: "READY", 2: "READY", 3: "READY"}}

        def _fake_cli_download(req: DownloadRequest, resolved_version: int | None = None) -> bool:
            self.assertEqual(resolved_version, 2)
            zip_dest = req.root_datasets_dir / f"{req.slug}_v2.zip"
            self._create_remote_sandbox_zip(zip_dest, tamper=False)
            return True

        with patch("tools.manifold_sync.OUT_PARENT", datasets_root), \
                patch("tools.manifold_sync.setup_auth", return_value=True), \
                patch("core.sync_remote.get_dataset_version_info", return_value=fake_info), \
                patch("core.sync_download._download_via_api_or_cli",
                      side_effect=_fake_cli_download):
            result = CliRunner().invoke(
                cli_app,
                [
                    "sync",
                    "pull",
                    "--url",
                    "owner/sandbox@v2",
                    "--output",
                    "LemGendizedSandbox",
                    "--force",
                    "--report",
                    str(report_file),
                    "--json",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(report_file.is_file())
        payload = json.loads(report_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["action"], "pull")
        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["checksum_verified"])


if __name__ == "__main__":
    unittest.main()
