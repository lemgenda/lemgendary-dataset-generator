"""
LemGendary Dataset Compiler: custom-compile contract tests (DS D-02, DS D-03).

Every test works on temp copies of presets.yaml and unified_data.yaml; the
real registry files are only read, never written.
"""

from __future__ import annotations

import ast
import asyncio
import gc
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.jobs import JobManager
from api.models import JobResponse, JobState, JobType
from api.routes import gui
from core import presets, unified_registry
from core.compile_outcome import ManifoldOutcome, finalize_compile
from core.config_schema import load_unified_data
from formats.directory import purge_loose_layout

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_PRESETS = PROJECT_ROOT / "presets.yaml"
REAL_UNIFIED = PROJECT_ROOT / "unified_data.yaml"


class _FakeJobManager:
    """Records job creation and start without spawning a compiler process."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.started: list[str] = []

    def create_job(self, job_type: JobType, command: list[str], parameters: dict[str, Any]) -> JobResponse:
        """Record the job and return a pending JobResponse."""
        self.created.append({"command": command, "parameters": parameters})
        return JobResponse(
            id=f"job-{len(self.created)}",
            job_type=job_type,
            state=JobState.PENDING,
            created_at="2026-11-02T00:00:00+00:00",
            log_file="logs/test.log",
            parameters=parameters,
        )

    def start_job(self, job_id: str, _loop: asyncio.AbstractEventLoop) -> None:
        """Record that the job was started."""
        self.started.append(job_id)


class _TempRegistryCase(unittest.TestCase):
    """Copies presets.yaml / unified_data.yaml into a temp dir and points the code at them."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.presets_path = self.tmp / "presets.yaml"
        self.unified_path = self.tmp / "unified_data.yaml"
        shutil.copyfile(REAL_PRESETS, self.presets_path)
        shutil.copyfile(REAL_UNIFIED, self.unified_path)

        for target, value in (
            ("core.presets.DEFAULT_PRESETS_FILE", self.presets_path),
            ("core.unified_registry.DEFAULT_UNIFIED_DATA_PATH", self.unified_path),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.jobs = _FakeJobManager()
        jobs_patcher = patch.object(gui, "job_manager", self.jobs)
        jobs_patcher.start()
        self.addCleanup(jobs_patcher.stop)

        app = FastAPI()
        app.include_router(gui.router)
        self.client = TestClient(app)

    def _post(self, **body: Any):
        payload = {"custom_name": "Unit Test Faces", "sources": ["https://www.kaggle.com/datasets/owner/faces"]}
        payload.update(body)
        return self.client.post("/gui/custom-compile", json=payload)


class TestDefaultPreset(_TempRegistryCase):
    """The custom-compile default preset comes from presets.yaml."""
    def test_real_default_preset_exists_in_presets_yaml(self) -> None:
        """The shipped default preset is one of the listed presets."""
        default_name = presets.get_default_preset_name(REAL_PRESETS)
        self.assertIn(default_name, presets.list_presets(REAL_PRESETS))

    def test_default_comes_from_presets_yaml(self) -> None:
        """Changing default_preset in presets.yaml changes the preset a request gets."""
        text = self.presets_path.read_text(encoding="utf-8")
        self.presets_path.write_text(
            text.replace('default_preset: "quality-vision"', 'default_preset: "cloud-archival"'),
            encoding="utf-8",
        )
        response = self._post()
        self.assertEqual(response.status_code, 200, response.text)
        command = self.jobs.created[0]["command"]
        self.assertEqual(command[command.index("--preset") + 1], "cloud-archival")
        self.assertEqual(self.jobs.started, ["job-1"])

    def test_default_naming_an_unknown_preset_is_rejected(self) -> None:
        """A default_preset that names no preset raises ValueError."""
        text = self.presets_path.read_text(encoding="utf-8")
        self.presets_path.write_text(
            text.replace('default_preset: "quality-vision"', 'default_preset: "streaming-webdataset"'),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            presets.get_default_preset_name()

    def test_unknown_preset_returns_422_with_valid_list(self) -> None:
        """An unknown preset returns 422 listing the valid presets and leaves the registry untouched."""
        before = self.unified_path.read_bytes()
        response = self._post(preset="streaming-webdataset")
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        for name in presets.list_presets():
            self.assertIn(name, detail)
        self.assertEqual(self.jobs.created, [])
        self.assertEqual(self.unified_path.read_bytes(), before)


class TestRegistryWrite(_TempRegistryCase):
    """Writing a custom entry to unified_data.yaml is safe."""
    def test_comments_and_formatting_preserved(self) -> None:
        """Comments and formatting in unified_data.yaml survive adding an entry."""
        original = self.unified_path.read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in original else "\n"
        commented = original.replace(f"datasets:{newline}", f"# top comment{newline}datasets:{newline}"
                                     f"  # dataset comment{newline}", 1)
        self.unified_path.write_bytes(commented.encode("utf-8"))

        response = self._post(preset="detection-variable", canonical_format="parquet")
        self.assertEqual(response.status_code, 200, response.text)

        updated = self.unified_path.read_bytes().decode("utf-8")
        self.assertIn("# top comment", updated)
        self.assertIn("  # dataset comment", updated)
        # datasets: is the last block, so the existing text is kept byte for byte
        # and the new entry follows it.
        self.assertTrue(updated.startswith(commented))
        self.assertIn("unit_test_faces:", updated[len(commented):])

        data = load_unified_data(self.unified_path)
        entry = data.datasets["unit_test_faces"]
        self.assertEqual(entry.canonical_format, "parquet")
        self.assertEqual(entry.refs[0].ref, "kaggle://owner/faces")
        self.assertEqual(self.jobs.started, ["job-1"])

    def test_duplicate_key_returns_409(self) -> None:
        """A name that maps to an existing key returns 409 and writes nothing."""
        before = self.unified_path.read_bytes()
        existing_key = next(iter(yaml.safe_load(before)["datasets"]))
        name = " ".join(existing_key.split("_"))
        response = self._post(custom_name=name)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.unified_path.read_bytes(), before)
        self.assertEqual(self.jobs.created, [])

    def test_invalid_entry_rejected_before_write(self) -> None:
        """An invalid entry returns 422 before anything is written or started."""
        before = self.unified_path.read_bytes()
        response = self._post(canonical_format="zip")
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.unified_path.read_bytes(), before)
        self.assertEqual(self.jobs.created, [])

    def test_failed_write_does_not_start_job(self) -> None:
        """A failed registry write returns 500 and starts no job."""
        before = self.unified_path.read_bytes()
        with patch("core.unified_registry.os.replace", side_effect=PermissionError("locked")):
            response = self._post()
        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(self.unified_path.read_bytes(), before)
        self.assertEqual(self.jobs.created, [])
        self.assertEqual(self.jobs.started, [])
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["presets.yaml", "unified_data.yaml"])

    def test_registry_service_refuses_existing_key(self) -> None:
        """add_dataset_entry raises RegistryEntryExistsError for an existing key."""
        existing_key = next(iter(yaml.safe_load(self.unified_path.read_bytes())["datasets"]))
        with self.assertRaises(unified_registry.RegistryEntryExistsError):
            unified_registry.add_dataset_entry(existing_key, {"name": "X"})


class TestFinalizeCompile(unittest.TestCase):
    """finalize_compile exit codes and loose-layout purging."""
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        self.root = tmp / "LemGendizedX"
        self.purged: list[Path] = []

    def _purge(self, root: Path) -> list[Path]:
        self.purged.append(root)
        return []

    def test_nothing_compiled_fails_without_purge(self) -> None:
        """No compiled samples gives exit code 1 and purges nothing."""
        self.assertEqual(finalize_compile([], purge_loose=True, purge=self._purge), 1)
        zero = [ManifoldOutcome("x", self.root, samples=0, containers_written=("parquet",))]
        self.assertEqual(finalize_compile(zero, purge_loose=True, purge=self._purge), 1)
        self.assertEqual(self.purged, [])

    def test_purge_runs_after_successful_compile_with_container(self) -> None:
        """A successful compile with a container purges the loose layout."""
        ok = [ManifoldOutcome("x", self.root, samples=10, containers_written=("parquet",))]
        self.assertEqual(finalize_compile(ok, purge_loose=True, purge=self._purge), 0)
        self.assertEqual(self.purged, [self.root])

    def test_purge_keeps_only_copy_when_no_container(self) -> None:
        """Without a container the loose layout is the only copy and is kept."""
        ok = [ManifoldOutcome("x", self.root, samples=10)]
        self.assertEqual(finalize_compile(ok, purge_loose=True, purge=self._purge), 0)
        self.assertEqual(self.purged, [])

    def test_no_purge_when_not_requested(self) -> None:
        """Nothing is purged when purging was not requested."""
        ok = [ManifoldOutcome("x", self.root, samples=10, containers_written=("mds",))]
        self.assertEqual(finalize_compile(ok, purge_loose=False, purge=self._purge), 0)
        self.assertEqual(self.purged, [])

    def test_purge_loose_layout_keeps_containers(self) -> None:
        """purge_loose_layout removes images/ and targets/ and keeps container folders."""
        for sub in ("images/train", "targets/val", "parquet"):
            (self.root / sub).mkdir(parents=True)
        removed = purge_loose_layout(self.root)
        self.assertEqual(sorted(p.name for p in removed), ["images", "targets"])
        self.assertTrue((self.root / "parquet").exists())
        self.assertFalse((self.root / "images").exists())


class TestCompilerWiring(unittest.TestCase):
    """manifold_compile imports torch, so its wiring is checked on the source tree."""

    def setUp(self) -> None:
        self.tree = ast.parse((PROJECT_ROOT / "core" / "manifold_compile.py").read_text(encoding="utf-8"))

    def test_cleanup_does_not_short_circuit_process_dataset(self) -> None:
        """process_dataset does not call a cleanup that ends it early."""
        func = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_dataset")
        attrs = {n.attr for n in ast.walk(func) if isinstance(n, ast.Attribute)}
        self.assertNotIn("cleanup", attrs)

    def test_main_exits_with_finalize_compile(self) -> None:
        """The __main__ block exits with finalize_compile's result."""
        main = next(
            n for n in self.tree.body
            if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)
        )
        calls = {n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("finalize_compile", calls)


class TestJobStatusPropagation(unittest.TestCase):
    """A job's final state follows its process exit code."""
    def test_compile_that_produced_nothing_ends_failed(self) -> None:
        """A compile process that exits 1 ends in the FAILED state."""
        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(db_path=Path(tmp) / "jobs.db", logs_dir=Path(tmp) / "logs")
            code = (
                "import sys; from core.compile_outcome import finalize_compile; "
                "sys.exit(finalize_compile([], purge_loose=True))"
            )
            job = manager.create_job(JobType.COMPILE, [sys.executable, "-c", code], {})
            loop = asyncio.new_event_loop()
            runner = threading.Thread(target=loop.run_forever)
            runner.start()
            # Drive the job through the public start_job entry point, recording
            # the runner thread it spawns so the test can join it.
            spawned: list[threading.Thread] = []
            real_thread = threading.Thread

            def _recording_thread(*, target: Any, **kwargs: Any) -> threading.Thread:
                thread = real_thread(target=target, **kwargs)
                spawned.append(thread)
                return thread

            try:
                with patch("api.jobs.threading.Thread", side_effect=_recording_thread):
                    manager.start_job(job.id, loop)
                for thread in spawned:
                    thread.join(timeout=120)
            finally:
                loop.call_soon_threadsafe(loop.stop)
                runner.join()
                loop.close()
            finished = manager.get_job(job.id)
            # JobManager leaves sqlite connections to the garbage collector;
            # collect them so Windows can delete the temp jobs.db.
            del manager
            gc.collect()
            self.assertIsNotNone(finished)
            assert finished is not None
            self.assertEqual(finished.state, JobState.FAILED)
            self.assertEqual(finished.exit_code, 1)


if __name__ == "__main__":
    unittest.main()
