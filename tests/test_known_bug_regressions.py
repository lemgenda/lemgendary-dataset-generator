"""
Regression tests for the compiler bugs found during the L8 module split.

Each test pins one defect that existed before the split and was kept
unchanged by it: unconfigured compile CLI, missing ``labels/<split>``
folders, corrupt Parquet shards, tz-naive MT5 fallback, the degrade audit
insert, malformed ``dataset_info.yaml`` files and script-path entry points.
Tests that need the heavy compile stack skip when it is not installed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import sqlite3
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.server import create_app
from core import compiler_config
from tools.generate_degrade_registry import commit_records, init_synthetic_registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _has(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


requires_pandas = unittest.skipUnless(_has("pandas"), "needs pandas")
requires_compile_stack = unittest.skipUnless(
    _has("pandas", "torch"), "needs the compile stack (pandas, torch)"
)


@contextmanager
def _preserved_compiler_state() -> Iterator[None]:
    """Restore CONFIG and the installed CLI Namespace when the block exits."""
    config = dict(compiler_config.CONFIG)
    try:
        args: argparse.Namespace | None = compiler_config.get_args()
    except RuntimeError:
        args = None
    try:
        yield
    finally:
        compiler_config.CONFIG.clear()
        compiler_config.CONFIG.update(config)
        compiler_config.set_args(args)


class TestCompileArgsInstalled(unittest.TestCase):
    """``python -m core.manifold_compile`` must install its args (bug: never configured)."""

    def setUp(self) -> None:
        self.enterContext(_preserved_compiler_state())

    def test_install_args_makes_get_args_work_and_applies_overrides(self) -> None:
        """install_args() installs the Namespace and applies its CONFIG overrides."""
        compiler_config.set_args(None)
        ns = compiler_config.build_parser().parse_args(
            ["--model", "nima_aesthetic", "--workers", "3", "--image-format", "png"]
        )
        self.assertIs(compiler_config.install_args(ns), ns)
        self.assertIs(compiler_config.get_args(), ns)
        self.assertEqual(compiler_config.CONFIG["num_workers"], 3)
        self.assertEqual(compiler_config.CONFIG["image_format"], "png")

    @requires_compile_stack
    def test_prepare_compile_installs_parsed_args(self) -> None:
        """prepare_compile() installs the parsed args before the pool starts."""
        compile_setup = importlib.import_module("core.compile_setup")
        compiler_config.set_args(None)
        args = compile_setup.parse_compile_args(
            ["--model", "nima_aesthetic", "--workers", "2", "--mask-format", "png"]
        )
        with patch.object(compile_setup, "_preflight_models"), \
                patch.object(compile_setup, "_dped_canon_paths", return_value=set()):
            settings = compile_setup.prepare_compile(args)
        self.assertIs(compiler_config.get_args(), args)
        self.assertEqual(compiler_config.CONFIG["mask_format"], "png")
        self.assertEqual(settings.max_workers, 2)


@requires_pandas
class TestLabelFolderCreated(unittest.TestCase):
    """Auto-labels of restoration/segmentation/diffusion manifolds need ``labels/<split>``."""

    def test_write_label_file_creates_split_folder(self) -> None:
        """A label file can be written before labels/<split> exists."""
        sample_annotations = importlib.import_module("core.sample_annotations")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "LemGendizedRestore" / "labels" / "train" / "s_000000001.txt"
            sample_annotations.write_label_file(target, ["0 0.5 0.5 0.1 0.1"])
            self.assertEqual(target.read_text(encoding="utf-8"), "0 0.5 0.5 0.1 0.1\n")


@requires_pandas
class TestCorruptParquetShard(unittest.TestCase):
    """A corrupt shard must not produce records without a ``name`` (KeyError in PASS 1)."""

    def test_corrupt_shard_yields_nothing_to_register(self) -> None:
        """An unreadable shard yields one None per planned row."""
        sample_batch = importlib.import_module("core.sample_batch")
        sample_jobs = importlib.import_module("core.sample_jobs")
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "broken.parquet"
            bad.write_bytes(b"not a parquet file")
            job = sample_jobs.ShardJob(
                pq_path=bad,
                key=sample_jobs.SampleKey("nima", "ava", 0),
                context=sample_jobs.SourceContext("quality", None, tmp),
                train_prob=0.8,
                existing=sample_jobs.ExistingSamples(),
                keep_prob=1.0,
                num_rows=3,
            )
            with patch.object(sample_batch.pd, "read_parquet",
                              side_effect=OSError("corrupt footer")):
                results = sample_batch.batch_worker([(sample_batch.process_parquet_shard, job)])
        self.assertEqual(results, [None, None, None])


@requires_pandas
class TestMt5ChunkedFallback(unittest.TestCase):
    """The chunked MT5 fallback compared tz-aware bars with a naive start date."""

    def test_fallback_filters_by_start_date(self) -> None:
        """The fallback keeps only the bars on or after the start date."""
        np = importlib.import_module("numpy")
        mt5_download = importlib.import_module("forex.mt5_download")
        dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                 ("close", "f8"), ("tick_volume", "i8")]

        def make_bar(day: int) -> tuple[Any, ...]:
            ts = int(datetime(2024, 1, day, tzinfo=UTC).timestamp())
            return (ts, 1.0, 1.0, 1.0, 1.0, 10)

        chunks = iter([np.array([make_bar(5), make_bar(3), make_bar(1)], dtype=dtype), None])

        def fake_call(method: str, *_args: Any) -> Any:
            if method == "symbol_select":
                return True
            if method == "copy_rates_from_pos":
                return next(chunks)
            return None  # copy_rates_range is empty, so the chunked fallback runs

        with patch.object(mt5_download, "mt5_call", side_effect=fake_call), \
                patch.object(mt5_download, "resolve_symbol", return_value=None), \
                patch.object(mt5_download, "remember_symbol"), \
                patch.object(mt5_download.mt5_bridge, "get_timeframe_const", return_value=16385):
            df = mt5_download.download_bars("EURUSD", 60, start_date="2024-01-02", retries=1)
        self.assertEqual(len(df), 2)


class TestDegradeAuditEvents(unittest.TestCase):
    """A non-dry-run degrade failed at commit: the insert named a missing column."""

    def test_commit_records_writes_audit_event_keyed_by_registry_id(self) -> None:
        """The audit event references the registry row of its sample."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifold_registry.db"
            init_synthetic_registry(db_path)
            commit_records(db_path, [{
                "name": "deg_div2k_000000001", "source": "div2k", "task": "restoration",
                "split": "train", "img_format": "webp", "img_size": 10, "target_size": 20,
                "label_payload": {"blur": 1.5},
            }])
            with closing(sqlite3.connect(db_path)) as conn:
                row = conn.execute(
                    "SELECT r.name, a.event_type FROM audit_events a "
                    "JOIN registry r ON r.id = a.sample_id"
                ).fetchone()
        self.assertEqual(row, ("deg_div2k_000000001", "degradation_synthesis"))


class TestMalformedDatasetInfo(unittest.TestCase):
    """``/gui/datasets/with-stats`` returned 500 for a malformed ``dataset_info.yaml``."""

    def _catalog(self, info_bytes: bytes) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            manifold = Path(tmp) / "LemGendizedBroken"
            (manifold / "images").mkdir(parents=True)
            (manifold / "images" / "a.webp").write_bytes(b"x")
            (manifold / "dataset_info.yaml").write_bytes(info_bytes)
            with patch("api.routes.gui._resolve_output_root", return_value=Path(tmp)):
                response = TestClient(create_app()).get("/api/gui/datasets/with-stats")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_non_mapping_yaml(self) -> None:
        """A YAML list is ignored; the count falls back to the images."""
        body = self._catalog(b"- just\n- a list\n")
        self.assertEqual(body["datasets"][0]["sample_count"], 1)

    def test_non_utf8_yaml(self) -> None:
        """A Latin-1 file is ignored; the count falls back to the images."""
        body = self._catalog("task: r\xe9storation\n".encode("latin-1"))
        self.assertEqual(body["datasets"][0]["sample_count"], 1)

    def test_non_numeric_count(self) -> None:
        """A non-numeric count is ignored; the task is still read."""
        body = self._catalog(b"task: quality\ncount: many\n")
        self.assertEqual(body["datasets"][0]["task"], "quality")


class TestModuleEntryPoints(unittest.TestCase):
    """Package modules import ``core``/``tools`` absolutely, so they run as ``python -m``.

    ``python tools/x.py`` puts ``tools/`` (not the project root) on
    ``sys.path``; nothing in the repository may launch them that way.
    """

    _SCRIPT_PATH_CALL = re.compile(
        r"python(?:\.exe)?['\"]?\s*,?\s*['\"]?(?:\./)?"
        r"(?:api|audit|commands|converters|core|degrade|forex|formats|generators|"
        r"runtime|services|sources|tools|utils)[/\\][\w/\\]+\.py"
    )

    def test_no_launcher_or_doc_runs_a_package_module_by_path(self) -> None:
        """No launcher, route or document runs a package module by path."""
        offenders = []
        for pattern in ("*.ps1", "*.md", "api/**/*.py", "commands/**/*.py",
                        "services/**/*.py", "core/**/*.py", "tools/**/*.py"):
            for path in PROJECT_ROOT.glob(pattern):
                if path.name == "REVIEW_REPORT.md":
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                offenders += [f"{path.name}: {m.group(0)}"
                              for m in self._SCRIPT_PATH_CALL.finditer(text)]
        self.assertEqual(offenders, [])

    def test_compile_entry_point_parses_as_module(self) -> None:
        """core.manifold_compile resolves as a module and shares the parser."""
        parser = compiler_config.build_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self.assertTrue(importlib.util.find_spec("core.manifold_compile"))


if __name__ == "__main__":
    unittest.main()
