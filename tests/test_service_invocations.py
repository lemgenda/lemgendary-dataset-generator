"""
LemGendary Dataset Compiler: service entry points run by API jobs.

Every long-running API job runs ``python -m <service module> <args>``. These
tests check that the arguments built from each option table are parsed back
by the same service's ``main`` into the same service call, and that the job
manager turns an invocation into the project interpreter's argv.
"""

from __future__ import annotations

import importlib
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

from api.gui_compile import quick_compile_command
from api.jobs import job_command
from api.models import QuickCompileRequest
from core.cli_args import build_parser, venv_python
from generators import LabelGenerator, MaskGenerator, PromptGenerator
from services import degrade_service, llm_service, sync_service
from services.compiler_service import COMPILE_FLAGS, CompileOptions, compile_invocation
from services.invocation import ServiceInvocation
from services.migration_service import queue_invocation, reduce_invocation

_COMPILE_VALUES: dict[str, Any] = {
    "model": "nima_aesthetic",
    "preset": "quality-vision",
    "max_gb": 2.5,
    "suffix": "Mini",
    "workers": 3,
    "no_vetting": True,
    "no_labeling": True,
    "no_hash": True,
    "image_format": "png",
    "image_quality": 90,
    "target_quality": 97,
    "mask_format": "png",
    "also_format": "mds",
    "shard_ext": "tar",
    "start_date": "2019-01-01",
    "end_date": "2026-11-14",
    "force_duplicate": True,
    "accept_space_loss": True,
    "label_strategy": LabelGenerator.strategies()[0],
    "prompt_strategy": PromptGenerator.strategies()[0],
    "mask_strategy": MaskGenerator.strategies()[0],
    "cleanup": True,
}


class TestJobCommand(unittest.TestCase):
    """The job manager runs an invocation on the project interpreter."""

    def test_invocation_becomes_unbuffered_module_run(self) -> None:
        """``python -u -m module args`` on the venv interpreter."""
        argv = job_command(ServiceInvocation("services.llm_service", ("--dataset", "x")))
        self.assertEqual(
            argv, [venv_python(), "-u", "-m", "services.llm_service", "--dataset", "x"]
        )

    def test_every_invocation_module_has_a_main(self) -> None:
        """Each service module an API job runs exposes ``main(argv)``."""
        invocations = (
            compile_invocation({"model": "m"}),
            degrade_service.degrade_invocation("src", "out", {}),
            llm_service.llm_compile_invocation("test_llm"),
            queue_invocation(datasets=["a"]),
            reduce_invocation(dataset="a", yes=True),
            sync_service.kaggle_invocation(action="status", repo_id="o/d"),
            sync_service.manifold_sync_invocation(action="sync", model="m"),
        )
        for invocation in invocations:
            with self.subTest(module=invocation.module):
                module = importlib.import_module(invocation.module)
                self.assertTrue(callable(getattr(module, "main", None)))


class TestCompileContract(unittest.TestCase):
    """The compile option table matches the compile parser."""

    def test_every_option_round_trips_through_the_compile_parser(self) -> None:
        """Each option reaches the parser attribute of the same name."""
        self.assertEqual({f.name for f in COMPILE_FLAGS}, set(_COMPILE_VALUES))
        options = cast(CompileOptions, dict(_COMPILE_VALUES))
        parsed = build_parser().parse_args(list(compile_invocation(options).args))
        for name, value in _COMPILE_VALUES.items():
            with self.subTest(option=name):
                self.assertEqual(getattr(parsed, name), value)

    def test_unknown_option_is_rejected(self) -> None:
        """An option outside the table raises TypeError."""
        with self.assertRaises(TypeError):
            compile_invocation(cast(CompileOptions, {"bogus": 1}))

    def test_quick_compile_forwards_shard_ext(self) -> None:
        """QuickCompileRequest.shard_ext reaches the compile parser; unset keeps its default."""
        with_ext = quick_compile_command(QuickCompileRequest(model="m", shard_ext="tar"))
        default = quick_compile_command(QuickCompileRequest(model="m"))
        self.assertEqual(build_parser().parse_args(list(with_ext.args)).shard_ext, "tar")
        self.assertEqual(build_parser().parse_args(list(default.args)).shard_ext, "data")


class TestEntryPointsRoundTrip(unittest.TestCase):
    """Each service ``main`` turns the built arguments back into the same call."""

    def test_degrade(self) -> None:
        """Degrade options reach run_degrade unchanged."""
        options: degrade_service.DegradeOptions = {
            "profile": "haze", "intensity": "high", "pairs": 10, "val_split": 0.2,
            "seed": 7, "image_format": "png", "workers": 2, "dry_run": True,
        }
        invocation = degrade_service.degrade_invocation("clean", "out", options)
        with patch.object(degrade_service, "run_degrade", return_value=0) as run:
            self.assertEqual(degrade_service.main(list(invocation.args)), 0)
        run.assert_called_once_with("clean", "out", **options)

    def test_llm(self) -> None:
        """The dataset key reaches the LLM compiler."""
        invocation = llm_service.llm_compile_invocation("test_llm")
        with patch("tools.llm_helper.base_compiler.BaseLLMCompiler") as compiler:
            compiler.return_value.compile_manifold.return_value = {"total_records": 1}
            self.assertEqual(llm_service.main(list(invocation.args)), 0)
        compiler.return_value.compile_manifold.assert_called_once_with("test_llm")

    def test_llm_failure_exits_1(self) -> None:
        """An expected compile error becomes exit code 1."""
        with patch("tools.llm_helper.base_compiler.BaseLLMCompiler") as compiler, \
                self.assertLogs("services.llm_service", level="ERROR"):
            compiler.return_value.compile_manifold.side_effect = KeyError("nope")
            self.assertEqual(llm_service.main(["--dataset", "nope"]), 1)

    def test_kaggle(self) -> None:
        """Kaggle options reach run_kaggle_action as one request."""
        invocation = sync_service.kaggle_invocation(
            action="upload", repo_id="owner/data", output_dir="out", no_wait=True
        )
        with patch("sources.kaggle.run_kaggle_action", return_value=0) as run:
            self.assertEqual(sync_service.main(list(invocation.args)), 0)
        request = run.call_args.args[0]
        self.assertEqual(
            (request.action, request.repo_id, request.output_dir, request.no_wait),
            ("upload", "owner/data", "out", True),
        )
        self.assertFalse(request.all_datasets)

    def test_manifold_sync(self) -> None:
        """Push/pull options reach run_sync_action."""
        invocation = sync_service.manifold_sync_invocation(action="get", model="LemGendizedX")
        with patch("tools.manifold_sync.run_sync_action", return_value=0) as run:
            self.assertEqual(sync_service.main(list(invocation.args)), 0)
        run.assert_called_once_with("get", "LemGendizedX", None, no_wait=False)

    def test_queue(self) -> None:
        """Queue arguments reach the modernization queue unchanged."""
        invocation = queue_invocation(datasets=["a", "b"], dry_run=True, force=True)
        main_mock = MagicMock(return_value=0)
        with patch("tools.modernization_queue.main", main_mock):
            module = importlib.import_module(invocation.module)
            self.assertEqual(module.main(list(invocation.args)), 0)
        self.assertEqual(main_mock.call_args.args[0], list(invocation.args[1:]))

    def test_unknown_subcommand_exits_2(self) -> None:
        """A missing or unknown subcommand is a usage error."""
        self.assertEqual(sync_service.main([]), 2)
        self.assertEqual(sync_service.main(["bogus"]), 2)


if __name__ == "__main__":
    unittest.main()
