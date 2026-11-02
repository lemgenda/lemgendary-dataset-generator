"""Modernize and reduce plumbing: scope, dry-run, force and non-interactive reduce (review D-05, D-06, D-09)."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from fastapi import HTTPException
from typer.testing import CliRunner

import cli
from api.routes import modernize as modernize_route
from api.routes.modernize import (
    ManifoldReduceRequest,
    ModernizeRunRequest,
    build_modernize_command,
    build_reduce_command,
)
from services.migration_service import MigrationService
from tools import modernization_queue

# tools.manifold_reduce imports core.compiler_core, which needs the heavy compile stack.
HAS_REDUCE_DEPS = all(importlib.util.find_spec(m) is not None for m in ("pandas", "torch", "webdataset"))
requires_reduce_deps = unittest.skipUnless(
    HAS_REDUCE_DEPS, "tools.manifold_reduce needs pandas, torch and webdataset"
)


def _manifold_reduce():
    return importlib.import_module("tools.manifold_reduce")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _make_datasets_root(root: Path) -> Path:
    """Two datasets with destructible sources (zip + legacy dir, loose files) and a registry."""
    registry = root / "lemgendary-datasets" / "unified_data.yaml"
    registry.parent.mkdir()
    registry.write_text(
        yaml.safe_dump({
            "datasets": {
                "alpha": {"name": "Alpha", "modernized_folder": "LemGendizedAlpha"},
                "alpha_v2": {"name": "AlphaV2", "modernized_folder": "LemGendizedAlphaV2"},
            }
        }),
        encoding="utf-8",
    )
    data = root / "LemGendaryDatasets"
    (data / "LemGendizedAlphaLarge" / "images" / "train").mkdir(parents=True)
    (data / "LemGendizedAlphaLarge" / "images" / "train" / "a.png").write_bytes(b"legacy")
    (data / "lemgendizedalpha.zip").write_bytes(b"zip-bytes")
    (data / "LemGendizedAlphaV2" / "images" / "train").mkdir(parents=True)
    (data / "LemGendizedAlphaV2" / "images" / "train" / "b.png").write_bytes(b"loose")
    return registry


class TestModernizationQueueSafety(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = _make_datasets_root(self.root)
        self.data = self.root / "LemGendaryDatasets"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _items(self) -> list[modernization_queue.ManifoldQueueItem]:
        return modernization_queue.load_queue_manifest(self.data, self.registry)

    def test_dry_run_changes_nothing_even_with_destructive_flags(self) -> None:
        before = _snapshot(self.root)
        with patch.object(modernization_queue, "_convert_item") as convert, \
                patch.object(modernization_queue, "push_manifold_to_kaggle") as push:
            code = modernization_queue.run_queue(
                items=self._items(),
                push_kaggle=True,
                delete_zip=True,
                delete_legacy_dir=True,
                purge_loose=True,
                dry_run=True,
                force=True,
            )
        self.assertEqual(code, 0)
        convert.assert_not_called()
        push.assert_not_called()
        self.assertEqual(_snapshot(self.root), before)

    def test_dry_run_reports_planned_deletions(self) -> None:
        alpha = next(it for it in self._items() if it.key == "alpha")
        plan = modernization_queue.describe_planned_actions(
            alpha, push_kaggle=False, delete_zip=True, delete_legacy_dir=True, purge_loose=False, force=False
        )
        self.assertTrue(any("would delete zip" in line for line in plan))
        self.assertTrue(any("would delete legacy folder" in line for line in plan))

    def test_dataset_scope_prefers_exact_match(self) -> None:
        with patch.object(modernization_queue, "_convert_item", return_value=True) as convert:
            code = modernization_queue.run_queue(items=self._items(), filter_keys=["alpha"])
        self.assertEqual(code, 0)
        self.assertEqual([c.kwargs["it"].key for c in convert.call_args_list], ["alpha"])

    def test_unknown_dataset_runs_nothing_and_fails(self) -> None:
        with patch.object(modernization_queue, "_convert_item") as convert:
            code = modernization_queue.run_queue(items=self._items(), filter_keys=["nope"])
        self.assertEqual(code, 1)
        convert.assert_not_called()

    def test_force_reconverts_containerized_item(self) -> None:
        items = self._items()
        alpha = next(it for it in items if it.key == "alpha")
        alpha.is_containerized = True
        with patch.object(modernization_queue, "_convert_item", return_value=True) as convert:
            modernization_queue.run_queue(items=items, filter_keys=["alpha"], force=False)
            convert.assert_not_called()
            modernization_queue.run_queue(items=items, filter_keys=["alpha"], force=True)
            convert.assert_called_once()

    def test_main_parses_shared_args(self) -> None:
        args = modernization_queue.build_queue_args(
            datasets=["alpha", "alpha_v2"], workers=4, dry_run=True, force=True
        )
        self.assertIn("--keep-zip", args)
        self.assertIn("--keep-legacy-dir", args)
        self.assertNotIn("--purge-loose", args)
        self.assertEqual(args[args.index("--datasets") + 1], "alpha,alpha_v2")
        with patch.object(modernization_queue, "_ROOT", self.registry.parent), \
                patch.object(modernization_queue, "run_queue", return_value=0) as run:
            self.assertEqual(modernization_queue.main(args), 0)
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["filter_keys"], ["alpha", "alpha_v2"])
        self.assertTrue(kwargs["dry_run"])
        self.assertTrue(kwargs["force"])
        self.assertFalse(kwargs["delete_zip"])
        self.assertFalse(kwargs["delete_legacy_dir"])
        self.assertFalse(kwargs["purge_loose"])


class TestModernizeRoute(unittest.TestCase):
    def test_gui_payload_fields_reach_the_command(self) -> None:
        req = ModernizeRunRequest(dataset="alpha", dry_run=True, force=True, workers=3)
        cmd = build_modernize_command(req)
        self.assertEqual(cmd[1], "tools/modernization_queue.py")
        self.assertEqual(cmd.count("--datasets"), 1)
        self.assertEqual(cmd[cmd.index("--datasets") + 1], "alpha")
        self.assertIn("--dry-run", cmd)
        self.assertIn("--force", cmd)
        self.assertNotIn("--purge-loose", cmd)

    def test_datasets_list_is_joined_once(self) -> None:
        cmd = build_modernize_command(ModernizeRunRequest(datasets=["a", "b"], dataset="a"))
        self.assertEqual(cmd.count("--datasets"), 1)
        self.assertEqual(cmd[cmd.index("--datasets") + 1], "a,b")

    def test_run_route_creates_job_with_command(self) -> None:
        with patch.object(modernize_route, "job_manager") as jm:
            jm.create_job.return_value.id = "job-1"
            asyncio.run(modernize_route.run_modernization(ModernizeRunRequest(dataset="alpha", dry_run=True)))
        cmd = jm.create_job.call_args.kwargs["command"]
        self.assertIn("--dry-run", cmd)
        jm.start_job.assert_called_once()


class TestReduceRoute(unittest.TestCase):
    def test_gui_payload_builds_supported_non_interactive_command(self) -> None:
        cmd = build_reduce_command(ManifoldReduceRequest(manifold="LemGendizedAlpha", scale=0.25))
        self.assertNotIn("--model", cmd)
        self.assertNotIn("--scale", cmd)
        self.assertIn("--yes", cmd)
        self.assertEqual(cmd[cmd.index("--dataset") + 1], "LemGendizedAlpha")
        self.assertEqual(float(cmd[cmd.index("--fraction") + 1]), 0.25)

    def test_scale_factor_maps_to_fraction(self) -> None:
        cmd = build_reduce_command(ManifoldReduceRequest(dataset_key="LemGendizedAlpha", scale=4))
        self.assertEqual(float(cmd[cmd.index("--fraction") + 1]), 0.25)

    def test_missing_manifold_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            build_reduce_command(ManifoldReduceRequest())
        self.assertEqual(ctx.exception.status_code, 400)

    @requires_reduce_deps
    def test_command_is_accepted_by_tool_parser(self) -> None:
        """The tool's real argparse accepts the route's arguments; no input() may be called."""
        manifold_reduce = _manifold_reduce()
        cmd = build_reduce_command(ManifoldReduceRequest(manifold="LemGendizedAlpha", scale=0.5, dry_run=True))
        with patch.object(manifold_reduce, "reduce_dataset", return_value=0) as reduce_fn, \
                patch("builtins.input", side_effect=AssertionError("input() called")):
            self.assertEqual(manifold_reduce.main(cmd[2:]), 0)
        kwargs = reduce_fn.call_args.kwargs
        self.assertEqual(kwargs["dataset"], "LemGendizedAlpha")
        self.assertTrue(kwargs["assume_yes"])
        self.assertTrue(kwargs["dry_run"])
        self.assertEqual(kwargs["fraction"], 0.5)


@requires_reduce_deps
class TestNonInteractiveReduce(unittest.TestCase):
    def setUp(self) -> None:
        self.mr = _manifold_reduce()
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        src = self.out / "LemGendizedAlpha" / "images" / "train"
        src.mkdir(parents=True)
        for i in range(8):
            (src / f"img_s{i % 2}_{i}.png").write_bytes(b"x" * 10)
        self.patches = [
            patch.object(self.mr, "OUT_PARENT", self.out),
            patch.object(self.mr, "generate_dataset_docs"),
            patch("builtins.input", side_effect=AssertionError("input() called")),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self.patches):
            p.stop()
        self._tmp.cleanup()

    def test_yes_without_dataset_fails(self) -> None:
        self.assertEqual(self.mr.reduce_dataset(assume_yes=True), 1)

    def test_unknown_dataset_fails(self) -> None:
        self.assertEqual(self.mr.reduce_dataset(dataset="LemGendizedNope", assume_yes=True), 1)

    def test_dry_run_writes_nothing(self) -> None:
        before = _snapshot(self.out)
        code = self.mr.reduce_dataset(dataset="LemGendizedAlpha", fraction=0.5, assume_yes=True, dry_run=True)
        self.assertEqual(code, 0)
        self.assertEqual(_snapshot(self.out), before)

    def test_fraction_limits_samples(self) -> None:
        code = self.mr.reduce_dataset(dataset="LemGendizedAlpha", fraction=0.25, assume_yes=True)
        self.assertEqual(code, 0)
        copied = list((self.out / "LemGendizedAlphaReduced" / "images" / "train").iterdir())
        self.assertEqual(len(copied), 2)

    def test_invalid_fraction_rejected(self) -> None:
        self.assertEqual(self.mr.reduce_dataset(dataset="LemGendizedAlpha", fraction=1.5, assume_yes=True), 1)


class TestCliModernize(unittest.TestCase):
    def test_modernize_manifolds_is_reachable(self) -> None:
        with patch.object(MigrationService, "modernize_manifolds", return_value=0) as svc:
            result = CliRunner().invoke(cli.app, ["modernize", "manifolds", "--dry-run", "--datasets", "a,b"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(svc.call_args.kwargs["dry_run"])
        self.assertEqual(svc.call_args.kwargs["datasets"], "a,b")

    def test_modernize_run_uses_queue_service(self) -> None:
        with patch.object(modernization_queue, "main", return_value=0) as qmain:
            result = CliRunner().invoke(cli.app, ["modernize", "run", "--dataset", "alpha", "--dry-run", "--force"])
        self.assertEqual(result.exit_code, 0, result.output)
        args = qmain.call_args.args[0]
        self.assertEqual(args, modernization_queue.build_queue_args(datasets=["alpha"], dry_run=True, force=True))

    def test_reduce_yes_passes_through(self) -> None:
        for command in (["reduce"], ["modernize", "reduce"]):
            with patch.object(MigrationService, "reduce_manifolds", return_value=0) as svc:
                result = CliRunner().invoke(
                    cli.app, [*command, "--dataset", "LemGendizedAlpha", "--fraction", "0.5", "--yes"]
                )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(svc.call_args.kwargs["yes"])
            self.assertEqual(svc.call_args.kwargs["dataset"], "LemGendizedAlpha")
            self.assertEqual(svc.call_args.kwargs["fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
