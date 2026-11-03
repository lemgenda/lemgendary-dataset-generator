"""
LemGendary Dataset Compiler: LLM job start, CLI polling and validate report
(DS D-04, DS D-10).

The real LLM compile is never run: the job command is replaced by a tiny
Python stub, and the in-process compiler is patched.
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import cli
from api.jobs import JobManager
from api.models import JobResponse, JobState, JobType
from api.routes import llm as llm_routes
from services.job_watch import JobNotFoundError, JobStatusError, wait_for_job

VALID_RECORD = {
    "id": "rec_1",
    "category": "troubleshooting_diagnostics",
    "messages": [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Help me."},
        {
            "role": "assistant",
            "content": "[CONFIRMED] Stage 1 triage, Stage 2 diagnosis, Stage 3 remediation.",
        },
    ],
}


class _StubCommandJobManager(JobManager):
    """Real JobManager whose job runs a stub command instead of the LLM compiler."""

    def __init__(self, root: Path, stub_code: str) -> None:
        super().__init__(db_path=root / "jobs.db", logs_dir=root / "logs")
        self.stub_code = stub_code
        self.requested: list[dict[str, Any]] = []

    def create_job(
        self, job_type: JobType, command: list[str], parameters: dict[str, Any]
    ) -> JobResponse:
        self.requested.append({"job_type": job_type, "command": command})
        return super().create_job(job_type, [sys.executable, "-c", self.stub_code], parameters)


class TestAsyncLLMCompileJob(unittest.TestCase):
    """The LLM compile route runs as a background job."""
    def _run_async_compile(self, stub_code: str) -> tuple[_StubCommandJobManager, JobResponse]:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        manager = _StubCommandJobManager(tmp, stub_code)
        app = FastAPI()
        app.include_router(llm_routes.router, prefix="/api")
        with patch.object(llm_routes, "job_manager", manager), TestClient(app) as client:
            resp = client.post(
                "/api/llm/compile", json={"dataset_key": "test_llm", "async_mode": True}
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertEqual(body["status"], "enqueued")
            job_id = body["job_id"]
            # The runner thread uses the app's event loop for broadcasts, so
            # join it before the TestClient shuts that loop down.
            runner = next(
                (t for t in threading.enumerate() if t.name == f"job-runner-{job_id}"), None
            )
            if runner is not None:
                runner.join(timeout=30)
            job = manager.get_job(job_id)
        assert job is not None
        # JobManager leaves sqlite connections to the garbage collector;
        # collect them so Windows can delete the temp jobs.db.
        gc.collect()
        return manager, job

    def test_async_job_starts_and_completes(self) -> None:
        """A job whose process exits 0 ends COMPLETED."""
        manager, job = self._run_async_compile("print('stub compile ok')")
        self.assertEqual(job.state, JobState.COMPLETED)
        self.assertEqual(job.exit_code, 0)
        self.assertEqual(job.job_type, JobType.LLM)
        requested = manager.requested[0]
        self.assertEqual(requested["job_type"], JobType.LLM)
        self.assertIn("--no-server", requested["command"])
        self.assertEqual(requested["command"][1:4], ["cli.py", "llm", "compile"])
        self.assertIn("stub compile ok", manager.get_log_content(job.id))

    def test_async_job_failure_is_reported_as_failed(self) -> None:
        """A job whose process exits non-zero ends FAILED with the exit code."""
        _, job = self._run_async_compile("import sys; sys.exit(3)")
        self.assertEqual(job.state, JobState.FAILED)
        self.assertEqual(job.exit_code, 3)
        self.assertIn("3", job.error_message or "")

    def test_sync_compile_runs_compiler_and_maps_errors(self) -> None:
        """The synchronous route runs the compiler and maps its errors to HTTP codes."""
        app = FastAPI()
        app.include_router(llm_routes.router, prefix="/api")
        compiler = MagicMock()
        compiler.return_value.compile_manifold.return_value = {"total_records": 2}
        with patch.object(llm_routes, "BaseLLMCompiler", compiler), TestClient(app) as client:
            ok = client.post("/api/llm/compile", json={"dataset_key": "test_llm"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["details"], {"total_records": 2})

        compiler.return_value.compile_manifold.side_effect = KeyError("test_llm")
        with patch.object(llm_routes, "BaseLLMCompiler", compiler), TestClient(app) as client, \
                self.assertLogs("api.routes.llm", level="ERROR") as logs:
            bad = client.post("/api/llm/compile", json={"dataset_key": "test_llm"})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("Traceback", "\n".join(logs.output))


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the fake clock instead of sleeping."""
        self.now += seconds


class TestWaitForJob(unittest.TestCase):
    """wait_for_job maps job states to exit codes."""
    def _wait(self, responses: list[Any], timeout_s: float = 10.0):
        clock = _FakeClock()
        feed = iter(responses)

        def fetch() -> dict[str, Any]:
            item = next(feed, responses[-1])
            if isinstance(item, Exception):
                raise item
            return item

        return wait_for_job(
            fetch, timeout_s=timeout_s, poll_interval_s=1.0, sleep=clock.sleep, clock=clock
        )

    def test_completed_exits_zero(self) -> None:
        """A completed job gives exit code 0."""
        result = self._wait(
            [{"state": "pending"}, {"state": "running"}, {"state": "completed", "exit_code": 0}]
        )
        self.assertEqual((result.state, result.exit_code), ("completed", 0))

    def test_failed_propagates_exit_code(self) -> None:
        """A failed job passes its exit code and error message through."""
        result = self._wait(
            [{"state": "running"}, {"state": "failed", "exit_code": 2, "error_message": "boom"}]
        )
        self.assertEqual((result.state, result.exit_code, result.detail), ("failed", 2, "boom"))

    def test_cancelled_without_exit_code_is_non_zero(self) -> None:
        """A cancelled job without an exit code gives 1."""
        self.assertEqual(self._wait([{"state": "cancelled", "exit_code": None}]).exit_code, 1)

    def test_timeout_is_non_zero(self) -> None:
        """A job still running at the timeout gives state timeout and exit code 1."""
        result = self._wait([{"state": "running"}], timeout_s=5.0)
        self.assertEqual((result.state, result.exit_code), ("timeout", 1))
        self.assertIn("running", result.detail)

    def test_unknown_state_and_missing_job_are_non_zero(self) -> None:
        """An unknown state or a missing job gives exit code 1."""
        self.assertEqual(self._wait([{"state": "weird"}]).state, "unknown")
        missing = self._wait([JobNotFoundError("job-x")])
        self.assertEqual((missing.state, missing.exit_code), ("unknown", 1))

    def test_transient_status_errors_are_retried(self) -> None:
        """A transient status error is retried."""
        result = self._wait(
            [JobStatusError("connection reset"), {"state": "completed", "exit_code": 0}]
        )
        self.assertEqual(result.exit_code, 0)


def _response(status_code: int, payload: Any) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


class TestCliLLMCompilePolling(unittest.TestCase):
    """llm compile waits for the sidecar job and exits with its code."""
    def _invoke(self, job_states: list[dict[str, Any]], *extra: str):
        states = iter(job_states)

        def fake_get(url: str, **_: Any) -> MagicMock:
            if url.endswith("/logs"):
                return _response(200, {"job_id": "job-1", "logs": "stub log line\n"})
            return _response(200, next(states, job_states[-1]))

        with (
            patch.object(cli, "_is_server_available", return_value=True),
            patch.object(cli, "get_or_create_token", return_value="token"),
            patch.object(cli, "JOB_POLL_INTERVAL_S", 0.01),
            patch.object(
                cli.requests,
                "post",
                return_value=_response(200, {"status": "enqueued", "job_id": "job-1"}),
            ),
            patch.object(cli.requests, "get", side_effect=fake_get),
        ):
            return CliRunner().invoke(cli.app, ["llm", "compile", "--dataset", "test_llm", *extra])

    def test_exits_zero_on_completed(self) -> None:
        """A completed job exits 0 and prints the job log."""
        result = self._invoke([{"state": "running"}, {"state": "completed", "exit_code": 0}])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("stub log line", result.output)

    def test_exits_with_job_code_on_failed(self) -> None:
        """A failed job exits with the job's code and prints its error."""
        result = self._invoke(
            [{"state": "failed", "exit_code": 4, "error_message": "compile crashed"}]
        )
        self.assertEqual(result.exit_code, 4, result.output)
        self.assertIn("compile crashed", result.output)

    def test_exits_non_zero_on_timeout(self) -> None:
        """--timeout stops the wait with exit code 1."""
        started = time.monotonic()
        result = self._invoke([{"state": "running"}], "--timeout", "1")
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("Timed out", result.output)
        self.assertLess(time.monotonic() - started, 10)

    def test_in_process_failure_prints_traceback(self) -> None:
        """An in-process compile failure exits 1 and prints the traceback."""
        with patch("tools.llm_helper.base_compiler.BaseLLMCompiler") as compiler:
            compiler.return_value.compile_manifold.side_effect = KeyError("test_llm")
            result = CliRunner().invoke(
                cli.app, ["llm", "compile", "--dataset", "test_llm", "--no-server"]
            )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Traceback", result.output)


class TestCliLLMValidate(unittest.TestCase):
    """llm validate reports issues through its exit code."""
    def _validate(self, records: list[dict[str, Any]]):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            return CliRunner().invoke(cli.app, ["llm", "validate", "--file", str(path)])

    def test_failure_prints_issues_and_exits_one(self) -> None:
        """A corpus with an invalid record prints the issues and exits 1."""
        bad = dict(VALID_RECORD, id="rec_bad", category="")
        result = self._validate([VALID_RECORD, bad])
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIsInstance(result.exception, SystemExit)  # a clean typer.Exit, not a crash
        self.assertIn("[FAIL]", result.output)
        self.assertIn("[category] (rec_bad): Missing required category", result.output)
        self.assertNotIn("Validation error", result.output)

    def test_valid_corpus_exits_zero(self) -> None:
        """A valid corpus exits 0."""
        result = self._validate([VALID_RECORD])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("[PASS]", result.output)

    def test_unparseable_file_exits_one(self) -> None:
        """A file that is not valid JSONL exits 1."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            path.write_text("{not json", encoding="utf-8")
            result = CliRunner().invoke(cli.app, ["llm", "validate", "--file", str(path)])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Could not read", result.output)


if __name__ == "__main__":
    unittest.main()
