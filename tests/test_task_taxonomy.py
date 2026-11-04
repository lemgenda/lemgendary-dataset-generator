"""D-12: ``task_taxonomy.json`` loader, registry consistency and ``task_type`` output.

The taxonomy is the single source of task ids, layout folders, label/mask
strategies and heuristic name patterns; every ``unified_data.yaml`` entry
names its task explicitly and the compiler writes it as ``task_type``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from core import task_taxonomy
from core.compile_forex import compile_forex_manifold
from core.config_schema import load_unified_data
from core.docs import generate_dataset_info_yaml
from core.paths import PROJECT_ROOT, UNIFIED_DATA
from core.task_taxonomy import (
    TASK_TAXONOMY_ENV,
    TaskTaxonomyError,
    resolve_task,
    taxonomy,
)
from generators.labels import LabelGenerator
from generators.masks import MaskGenerator

SCHEMA_FILE = PROJECT_ROOT / "task_taxonomy.schema.json"
PENDING_OWNER_CONFIRMATION = {
    "retinaface_mobilenet": "pose",
    "nima_authenticity": "quality",
    "forex_universe": "detection",
    "ai_helper_lemgend": "detection",
}


def _registry() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(UNIFIED_DATA.read_text(encoding="utf-8"))["datasets"]


def _minimal_taxonomy(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": 1,
        "heuristic": {"fallback_task": "alpha", "unnamed_task": "alpha"},
        "tasks": [
            {"id": "alpha", "layout_dir": "labels", "label_strategy": None,
             "mask_strategy": None, "name_patterns": []},
            {"id": "beta", "layout_dir": None, "label_strategy": "clip_zeroshot",
             "mask_strategy": None, "name_patterns": ["bet"]},
        ],
        "documentation_tasks": {"LemGendizedBeta": "beta_docs"},
    }
    data.update(overrides)
    return data


class TaxonomyLoaderTests(unittest.TestCase):
    """Lookup order, typed accessors, caching and loud failure."""

    def tearDown(self) -> None:
        task_taxonomy.clear_cache()

    def _load_from(self, data: Any) -> task_taxonomy.TaskTaxonomy:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task_taxonomy.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with patch.dict(os.environ, {TASK_TAXONOMY_ENV: str(path)}):
                task_taxonomy.clear_cache()
                return taxonomy()

    def test_repository_file_is_the_default(self) -> None:
        """Without the env var the loader reads the repository file."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(TASK_TAXONOMY_ENV, None)
            self.assertEqual(taxonomy().path, PROJECT_ROOT / "task_taxonomy.json")

    def test_loader_is_cached(self) -> None:
        """Repeated calls return the same parsed object."""
        self.assertIs(taxonomy(), taxonomy())

    def test_env_var_selects_another_file(self) -> None:
        """LEMGENDARY_TASK_TAXONOMY_FILE wins, and the accessors read that file."""
        tx = self._load_from(_minimal_taxonomy())
        self.assertEqual(tx.task_ids, ("alpha", "beta"))
        self.assertEqual(tx.layout_dir("alpha"), "labels")
        self.assertEqual(tx.label_strategy("beta"), "clip_zeroshot")
        self.assertIsNone(tx.mask_strategy("beta"))
        self.assertEqual(tx.infer_from_name("my_BETA_set"), "beta")
        self.assertEqual(tx.infer_from_name("other"), "alpha")
        self.assertEqual(tx.documentation_tasks["LemGendizedBeta"], "beta_docs")

    def test_unknown_task_has_no_layout_or_strategy(self) -> None:
        """A task id outside the taxonomy adds no folder and no strategy."""
        tx = taxonomy()
        self.assertIsNone(tx.get("not-a-task"))
        self.assertIsNone(tx.layout_dir("not-a-task"))
        self.assertIsNone(tx.label_strategy("not-a-task"))

    def test_missing_file_names_the_expected_path(self) -> None:
        """A missing file raises TaskTaxonomyError with the path."""
        missing = Path(tempfile.gettempdir()) / "no_such_dir_lgd" / "task_taxonomy.json"
        with patch.dict(os.environ, {TASK_TAXONOMY_ENV: str(missing)}):
            task_taxonomy.clear_cache()
            with self.assertRaises(TaskTaxonomyError) as ctx:
                taxonomy()
        self.assertIn(str(missing), str(ctx.exception))

    def test_invalid_files_fail_loudly(self) -> None:
        """Duplicate ids, unknown heuristic tasks and bad layout folders are rejected."""
        base = _minimal_taxonomy()
        duplicate = _minimal_taxonomy(tasks=base["tasks"] + [base["tasks"][0]])
        bad_fallback = _minimal_taxonomy(
            heuristic={"fallback_task": "gamma", "unnamed_task": "alpha"}
        )
        bad_layout = _minimal_taxonomy(tasks=[{**base["tasks"][0], "layout_dir": "images"}])
        no_strategy = _minimal_taxonomy(
            tasks=[{k: v for k, v in base["tasks"][0].items() if k != "mask_strategy"}]
        )
        for data in (duplicate, bad_fallback, bad_layout, no_strategy, [], {"tasks": []}):
            with self.subTest(data=data), self.assertRaises(TaskTaxonomyError):
                self._load_from(data)

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "needs jsonschema")
    def test_file_matches_its_schema(self) -> None:
        """task_taxonomy.json validates against task_taxonomy.schema.json."""
        jsonschema = importlib.import_module("jsonschema")
        schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
        data = json.loads(taxonomy().path.read_text(encoding="utf-8"))
        jsonschema.validate(data, schema)
        self.assertEqual(data["$schema"], f"./{SCHEMA_FILE.name}")


class RegistryTaxonomyConsistencyTests(unittest.TestCase):
    """Every registry entry names a taxonomy task, and resolution returns it."""

    def test_every_entry_has_a_taxonomy_task(self) -> None:
        """unified_data.yaml sets `task` on every entry, from the taxonomy ids."""
        ids = set(taxonomy().task_ids)
        for key, entry in _registry().items():
            with self.subTest(dataset=key):
                self.assertIn("task", entry)
                self.assertIn(entry["task"], ids)

    def test_resolution_returns_the_registry_task(self) -> None:
        """resolve_task yields the entry's task for its key and its name."""
        registry = _registry()
        for key, entry in registry.items():
            with self.subTest(dataset=key):
                self.assertEqual(resolve_task(registry, key), entry["task"])
                self.assertEqual(resolve_task(registry, entry["name"]), entry["task"])

    def test_pending_owner_confirmation_entries(self) -> None:
        """Exactly the four heuristic-derived tasks carry task_confirmed: false."""
        registry = _registry()
        pending = {k: e["task"] for k, e in registry.items() if e.get("task_confirmed") is False}
        self.assertEqual(pending, PENDING_OWNER_CONFIRMATION)
        for key, task in pending.items():
            self.assertEqual(taxonomy().infer_from_name(key), task)

    def test_schema_validation_rejects_unknown_tasks(self) -> None:
        """load_unified_data refuses a task id the taxonomy does not define."""
        raw = yaml.safe_load(UNIFIED_DATA.read_text(encoding="utf-8"))
        self.assertEqual(len(load_unified_data(UNIFIED_DATA).datasets), len(raw["datasets"]))
        first = next(iter(raw["datasets"]))
        raw["datasets"][first]["task"] = "not-a-task"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unified_data.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_unified_data(path)
        self.assertIn("not-a-task", str(ctx.exception))

    def test_strategies_are_known_to_the_generators(self) -> None:
        """Label and mask strategies are ones the generators accept."""
        labels = set(LabelGenerator.strategies())
        masks = set(MaskGenerator.strategies())
        for spec in taxonomy().tasks:
            with self.subTest(task=spec.id):
                if spec.label_strategy:
                    self.assertIn(spec.label_strategy, labels)
                if spec.mask_strategy:
                    self.assertIn(spec.mask_strategy, masks)

    def test_registry_folders_have_a_documentation_task(self) -> None:
        """Every modernized_folder has a documentation profile key."""
        doc_tasks = taxonomy().documentation_tasks
        for key, entry in _registry().items():
            with self.subTest(dataset=key):
                self.assertIn(entry["modernized_folder"], doc_tasks)


class TaskTypeWrittenTests(unittest.TestCase):
    """dataset_info.yaml always carries the compiler task id as `task_type`."""

    def test_override_is_written(self) -> None:
        """The compiler's task_type override lands in dataset_info.yaml."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "LemGendizedRetinaFace"
            root.mkdir()
            out = generate_dataset_info_yaml(root, overrides={"task_type": "pose"})
            info = yaml.safe_load(out.read_text(encoding="utf-8"))
        self.assertEqual(info["task_type"], "pose")
        self.assertEqual(info["task"], "detection")

    def test_docs_only_regeneration_uses_the_registry_task(self) -> None:
        """Without an override, task_type is the registry entry's task."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "LemGendizedNafNetDebluring"
            root.mkdir()
            out = generate_dataset_info_yaml(root)
            info = yaml.safe_load(out.read_text(encoding="utf-8"))
        self.assertEqual(info["task_type"], "restoration")

    def test_forex_compile_writes_task_type(self) -> None:
        """The forex path writes task_type and passes it to the docs engine."""
        registry = _registry()
        config = registry["forex_universe"]
        with tempfile.TemporaryDirectory() as tmp, \
                patch("core.compile_forex._run_acquisition"), \
                patch("core.compile_forex.write_forex_notebooks"), \
                patch("core.compile_forex.generate_dataset_docs") as docs:
            root = Path(tmp) / "LemGendizedForexUniverse"
            compile_forex_manifold("forex_universe", config, root, root.name,
                                   config["task"])
            info = yaml.safe_load((root / "dataset_info.yaml").read_text(encoding="utf-8"))
        self.assertEqual(info["task_type"], "detection")
        self.assertEqual(docs.call_args.kwargs["overrides"]["task_type"], "detection")

    @unittest.skipUnless(
        all(importlib.util.find_spec(m) for m in ("pandas", "torch")), "needs pandas and torch"
    )
    def test_manifold_compile_passes_task_type(self) -> None:
        """finish_manifold hands the manifold task to the docs engine as task_type."""
        manifold_compile = importlib.import_module("core.manifold_compile")
        compile_tasks = importlib.import_module("core.compile_tasks")
        ctx = compile_tasks.ManifoldContext(
            "parsenet", _registry()["parsenet"], "segmentation", "ParseNet",
            Path(tempfile.gettempdir()) / "LemGendizedParseNet", None,
        )
        with patch.object(manifold_compile, "cluster_styles"), \
                patch.object(manifold_compile, "write_index", return_value=[]), \
                patch.object(manifold_compile, "remove_empty_dirs"), \
                patch.object(manifold_compile, "write_containers", return_value=()), \
                patch.object(manifold_compile, "write_manifold_notebooks"), \
                patch.object(manifold_compile, "generate_dataset_docs") as docs:
            manifold_compile.finish_manifold(
                argparse.Namespace(), ctx, None, ("", "")
            )
        self.assertEqual(docs.call_args.kwargs["overrides"], {"task_type": "segmentation"})


if __name__ == "__main__":
    unittest.main()
