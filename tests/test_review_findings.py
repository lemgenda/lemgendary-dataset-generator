"""
Regression tests for the Medium/Low findings of REVIEW_REPORT.md (D-08 to D-20).

Each test names the finding it pins. Tests that need the heavy compile stack
skip when it is not installed.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from api import manifold_scan
from api.jobs import JobManager
from api.models import CompileJobRequest, JobState, JobType
from api.routes.jobs import _compile_command
from api.server import create_app
from commands import common
from commands.common import _stream_job_logs
from core import paths
from sources.base import load_kaggle_credentials

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCompileOptionsForwarded(unittest.TestCase):
    """D-08: the API forwards preset and shard_ext and leaves unset options to the preset."""

    def test_preset_and_shard_ext_reach_the_compiler(self) -> None:
        """The job argv carries --preset and --shard-ext."""
        argv = _compile_command(CompileJobRequest(
            model="nima_aesthetic", preset="quality-vision", shard_ext="tar",
        ))
        self.assertIn("--preset", argv)
        self.assertEqual(argv[argv.index("--shard-ext") + 1], "tar")

    def test_unset_image_options_do_not_override_the_preset(self) -> None:
        """No encoding flag is sent unless the caller set it."""
        argv = _compile_command(CompileJobRequest(model="nima_aesthetic", preset="quality-vision"))
        for flag in ("--image-format", "--image-quality", "--target-quality", "--mask-format"):
            self.assertNotIn(flag, argv)


class TestPathsIgnoreWorkingDirectory(unittest.TestCase):
    """D-11: state and data paths are anchored on the project root."""

    def test_paths_are_absolute_and_rooted(self) -> None:
        """Registry, raw sets and config resolve under the repository."""
        for path in (paths.UNIFIED_DATA, paths.RAW_SETS, paths.CONFIG_JSON):
            self.assertTrue(path.is_absolute())
            self.assertEqual(path.parent, PROJECT_ROOT)

    def test_state_dir_follows_the_override_only(self) -> None:
        """Token and PID file live in the state dir, whatever the CWD."""
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with patch.dict(os.environ, {"LEMGENDARY_STATE_DIR": ""}):
                    self.assertEqual(paths.token_file(), PROJECT_ROOT / ".lgd_server" / "token")
                    self.assertEqual(paths.pid_file().parent, PROJECT_ROOT / ".lgd_server")
            finally:
                os.chdir(cwd)

    def test_relative_output_folder_resolves_from_the_root(self) -> None:
        """A relative output_folder_name is taken from the project root."""
        self.assertEqual(paths.output_root("../Out"), (PROJECT_ROOT.parent / "Out").resolve())


@unittest.skipUnless(
    all(importlib.util.find_spec(m) for m in ("pandas", "torch")), "needs pandas and torch"
)
class TestRegistryTask(unittest.TestCase):
    """D-12: the registry ``task`` field is honoured before the name heuristic."""

    def test_task_field_wins_over_name_pattern(self) -> None:
        """A registered task overrides the name heuristic."""
        worker = importlib.import_module("core.compiler_worker")
        registry = {"retinaface_mobilenet": {"name": "RetinaFace", "task": "detection"}}
        with patch.dict(worker.DATASETS_META, registry, clear=True):
            self.assertEqual(worker.detect_task("retinaface_mobilenet"), "detection")


class TestKaggleCredentialProbe(unittest.TestCase):
    """D-15: a username alone (the default owner) is not a credential."""

    def test_username_only_is_not_enough(self) -> None:
        """Without a token or key the probe is False."""
        env = {"KAGGLE_USERNAME": "someone"}
        with patch.dict(os.environ, env, clear=True), \
                patch("sources.base.read_token_file", return_value=None):
            self.assertFalse(load_kaggle_credentials())
        with patch.dict(os.environ, {**env, "KAGGLE_KEY": "k"}, clear=True), \
                patch("sources.base.read_token_file", return_value=None):
            self.assertTrue(load_kaggle_credentials())


class TestUnknownJobStatus(unittest.TestCase):
    """D-16: the CLI must not report success when the final job status is unknown."""

    def test_unreachable_status_is_a_failure(self) -> None:
        """Both the log stream and the status request fail: exit code 1."""
        with patch.object(common, "connect", side_effect=OSError("refused")), \
                patch.object(common.requests, "get",
                             side_effect=requests.ConnectionError("refused")):
            self.assertEqual(_stream_job_logs("job-1"), 1)


class TestJobManager(unittest.TestCase):
    """D-17: connections close, cancelled jobs stay cancelled, log paths are relative."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="lgd_jobs_"))
        self.addCleanup(shutil.rmtree, self.tmp)
        self.jobs = JobManager(db_path=self.tmp / "jobs.db", logs_dir=self.tmp / "logs")

    def test_log_path_is_relative_to_the_state_dir(self) -> None:
        """The job record exposes logs/<id>.log, not an absolute path."""
        job = self.jobs.create_job(JobType.COMPILE, ["python", "-c", "pass"], {})
        self.assertEqual(job.log_file, f"logs/{job.id}.log")

    def test_job_cancelled_while_pending_is_not_run(self) -> None:
        """Starting a job cancelled while pending neither runs it nor reopens it."""
        job = self.jobs.create_job(JobType.COMPILE, ["python", "-c", "pass"], {})
        self.assertTrue(self.jobs.cancel_job(job.id))

        async def start() -> None:
            self.jobs.start_job(job.id, asyncio.get_running_loop())

        asyncio.run(start())
        for thread in threading.enumerate():
            if thread.name == f"job-runner-{job.id}":
                thread.join(timeout=10)
        stored = self.jobs.get_job(job.id)
        assert stored is not None
        self.assertEqual(stored.state, JobState.CANCELLED)

    def test_database_can_be_removed_after_use(self) -> None:
        """No connection is left open (Windows cannot delete an open SQLite file)."""
        self.jobs.create_job(JobType.COMPILE, ["python", "-c", "pass"], {})
        (self.tmp / "jobs.db").unlink()
        self.assertFalse((self.tmp / "jobs.db").exists())


class TestStatsCacheInvalidation(unittest.TestCase):
    """D-18: the scan cache notices changes inside the split folders."""

    def test_new_file_in_a_split_folder_is_counted(self) -> None:
        """Adding an image under images/train invalidates the cached scan."""
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "LemGendizedX" / "images" / "train"
            split.mkdir(parents=True)
            (split / "a.webp").write_bytes(b"x")
            first = manifold_scan.calculate_dataset_file_stats(split.parent.parent)
            (split / "b.webp").write_bytes(b"y")
            second = manifold_scan.calculate_dataset_file_stats(split.parent.parent)
        self.assertEqual(first.formats.webp + 1, second.formats.webp)


class TestImportSideEffects(unittest.TestCase):
    """D-19: importing the CLI does not build the API app or open jobs.db."""

    def test_cli_import_does_not_load_the_job_manager(self) -> None:
        """A fresh interpreter importing cli leaves api.jobs unloaded."""
        probe = "import cli, sys; print('api.jobs' in sys.modules)"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        out = subprocess.run([sys.executable, "-c", probe], cwd=PROJECT_ROOT, check=True,
                             capture_output=True, text=True, env=env).stdout.strip()
        self.assertEqual(out, "False")


class TestKaggleRouteErrors(unittest.TestCase):
    """D-20: bad Kaggle requests are client errors, not 500s."""

    def test_invalid_reference_is_400(self) -> None:
        """A reference without an owner is rejected with 400."""
        client = TestClient(create_app())
        with patch.dict(os.environ, {"LEMGENDARY_DISABLE_AUTH": "1"}):
            response = client.post("/api/kaggle/download", json={"kaggle_ref": "noslash"})
        self.assertEqual(response.status_code, 400, response.text)


class TestDegradeRegistryConnectionsClosed(unittest.TestCase):
    """The degrade registry helpers close their SQLite connections."""

    def test_registry_file_is_deletable(self) -> None:
        """init_synthetic_registry leaves no open handle."""
        generate_degrade_registry = importlib.import_module("tools.generate_degrade_registry")
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifold_registry.db"
            generate_degrade_registry.init_synthetic_registry(db_path)
            with closing(sqlite3.connect(db_path)) as conn:
                self.assertTrue(conn.execute("SELECT name FROM sqlite_master").fetchall())
            db_path.unlink()


if __name__ == "__main__":
    unittest.main()
