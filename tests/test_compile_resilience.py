"""PASS 1 resilience: worker init, interrupts, warm-up and resume dedup.

Covers audit items 2.1.5 (per-thread worker init race, ``os._exit`` on
Ctrl+C) and the compile part of 2.1.6 (warm-up bypassing dedup, resumed runs
recompiling registered samples, perceptual dedup never applied).
"""

import argparse
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from audit.dedup import NearDupIndex, PerceptualHasher
from core import compile_passes, compiler_worker
from core.compile_passes import (
    ExtractionRun,
    _absorb,
    _PoolState,
    _seed_state,
    _warmup,
    run_extraction,
)
from core.compile_registry import _stem_lower, load_registry_hashes, registry_row
from core.compile_setup import CompileSettings, make_executor
from core.registry import initialize_registry

_ZERO = "0" * 16


def _flip(hex_value: str, bits: int) -> str:
    """Flip the lowest ``bits`` bits of a 64-bit hex hash."""
    return f"{int(hex_value, 16) ^ ((1 << bits) - 1):016x}"


def _result(name: str, exact: str | None, perceptual: str | None = None) -> dict[str, Any]:
    return {
        "name": name, "source": "src", "task": "quality", "split": "train",
        "hash": exact, "perceptual_hash": perceptual, "size": 10, "nima_score": 5.0,
    }


def _run(conn: sqlite3.Connection, no_hash: bool = False) -> ExtractionRun:
    executor = mock.MagicMock()
    return ExtractionRun(conn, executor, max_workers=1, max_gb=1.0, no_hash=no_hash,
                         batch_size=4)


class TestNearDupIndex(unittest.TestCase):
    """Bucketed near-duplicate lookup."""

    def test_within_and_beyond_radius(self) -> None:
        """Distance 5 (the radius) matches; distance 20 on both hashes does not."""
        index = NearDupIndex(PerceptualHasher())
        index.add((_ZERO, _ZERO))
        self.assertTrue(index.has_near_dup((_flip(_ZERO, 5), _flip(_ZERO, 30))))
        self.assertFalse(index.has_near_dup((_flip(_ZERO, 20), _flip(_ZERO, 20))))

    def test_parse_rejects_malformed(self) -> None:
        """Only ``phash:dhash`` strings parse."""
        self.assertEqual(NearDupIndex.parse("ab:cd"), ("ab", "cd"))
        self.assertIsNone(NearDupIndex.parse("abcd"))
        self.assertIsNone(NearDupIndex.parse(":cd"))


class TestResumeAndWarmupDedup(unittest.TestCase):
    """Dedup state is seeded from the registry and applied during warm-up."""

    def setUp(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        self.conn = initialize_registry(Path(tmp) / "manifold_registry.db")
        self.conn.execute(
            "INSERT INTO registry (name, source, hash, perceptual_hash) VALUES (?, ?, ?, ?)",
            ("old", "src", "h-old", f"{_ZERO}:{_ZERO}"),
        )
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()

    def test_registry_hashes_seed_state(self) -> None:
        """A resumed run knows the exact and perceptual hashes of registered samples."""
        self.assertEqual(load_registry_hashes(self.conn), ({"h-old"}, [f"{_ZERO}:{_ZERO}"]))
        config = {"enable_dedup": True, "enable_perceptual_dedup": True}
        with mock.patch.dict(compile_passes.CONFIG, config):
            state = _seed_state(_run(self.conn))
            self.assertIn("h-old", state.seen_hashes)
            self.assertIsNotNone(state.near_dups)
            entries: list[tuple[Any, ...]] = []
            near = f"{_flip(_ZERO, 3)}:{'f' * 16}"
            _absorb(_run(self.conn), state, [
                _result("dup", "h-old"), _result("near", "h-new", near),
                _result("fresh", "h-fresh", f"{'f' * 16}:{'f' * 16}"),
            ], entries)
        self.assertEqual([row[0] for row in entries], ["fresh"])

    def test_warmup_results_are_deduplicated(self) -> None:
        """Serial warm-up results pass through the same dedup as pool results."""
        def _task(name: str) -> dict[str, Any]:
            return _result(name, "same-hash")

        tasks: list[Any] = [(_task, "w1"), (_task, "w2")]
        with mock.patch.dict(compile_passes.CONFIG, {"enable_dedup": True}):
            state = _PoolState()
            _warmup(_run(self.conn), state, tasks, mock.MagicMock())
        names = [r[0] for r in self.conn.execute("SELECT name FROM registry ORDER BY name")]
        self.assertEqual(names, ["old", "w1"])
        self.assertEqual(state.compiled_bytes, 10)

    def test_registry_row_carries_perceptual_hash(self) -> None:
        """The perceptual hash is persisted so a later resume can seed from it."""
        row = registry_row(_result("x", "h", "aa:bb"))
        self.assertEqual(row[-1], "aa:bb")


class TestInterrupt(unittest.TestCase):
    """Ctrl+C commits and re-raises instead of killing the process."""

    def test_keyboard_interrupt_commits_and_reraises(self) -> None:
        """The registry is committed, the pool shut down and the interrupt propagates."""
        def _boom() -> None:
            raise KeyboardInterrupt

        conn = mock.MagicMock()
        run = _run(conn, no_hash=True)
        executor = mock.MagicMock()
        run.executor = executor
        with mock.patch.object(compile_passes, "print"), self.assertRaises(KeyboardInterrupt):
            run_extraction(run, [(_boom,)], 0, "test")
        conn.commit.assert_called()
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


class TestWorkerInit(unittest.TestCase):
    """The shared worker state is initialised once per process."""

    def test_init_worker_runs_once(self) -> None:
        """Repeated calls (one per thread before the fix) load models only once."""
        with (
            mock.patch.object(compiler_worker, "_init_worker_locked") as body,
            mock.patch.object(compiler_worker, "_INIT_DONE", set()),
        ):
            compiler_worker.init_worker({})
            compiler_worker.init_worker({})
        body.assert_called_once()

    def test_make_executor_has_no_per_thread_initializer(self) -> None:
        """make_executor initialises once itself and builds a plain thread pool."""
        settings = mock.MagicMock(spec=CompileSettings)
        settings.max_workers = 2
        settings.dped_canon_paths = set()
        with (
            mock.patch("core.compile_setup.init_worker") as init,
            mock.patch("core.compile_setup.ThreadPoolExecutor") as pool_cls,
        ):
            make_executor(argparse.Namespace(), settings, set())
        init.assert_called_once()
        pool_cls.assert_called_once_with(max_workers=2)


class TestStemKey(unittest.TestCase):
    """Companion files share their sample's key (first-dot rule)."""

    def test_companions_map_to_sample_key(self) -> None:
        """``x.target.webp`` belongs to sample ``x``; the audit's last-dot claim is wrong."""
        self.assertEqual(_stem_lower("A.target.webp"), "a")
        self.assertEqual(_stem_lower("a.mask.webp"), "a")
        self.assertEqual(_stem_lower("noext"), "noext")


if __name__ == "__main__":
    unittest.main()
