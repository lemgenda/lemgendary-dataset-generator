"""The shared ecosystem file loader and the facts read from it.

``tests/conftest.py`` points ``LEMGENDARY_ECOSYSTEM_FILE`` at
``tests/fixtures/ecosystem.json``, whose values differ from the real file so
these tests prove the code reads the file rather than a literal.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from commands.common import server_host, server_port, sidecar_url
from core import ecosystem
from core.cli_args import resolve_lem_env
from core.compile_notebooks import get_training_suite_notebook_generators
from core.docs.templates import (
    ReadmeCommon,
    StandardOverview,
    StandardReadmeFields,
    render_standard_readme,
)
from core.kaggle_identity import default_handle, default_owner

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ecosystem.json"


class EcosystemFileTests(unittest.TestCase):
    """Lookup order, typed accessors and loud failure."""

    def tearDown(self) -> None:
        ecosystem.clear_cache()

    def test_env_var_selects_the_fixture(self) -> None:
        """LEMGENDARY_ECOSYSTEM_FILE wins over the workspace default."""
        self.assertEqual(ecosystem.ecosystem_file(), FIXTURE)

    def test_typed_accessors_read_the_fixture(self) -> None:
        """Ports (by id or alias), owner, docs URL and host come from the file."""
        self.assertEqual(ecosystem.service_port("datasets"), 18100)
        self.assertEqual(ecosystem.service_port("dataset-compiler"), 18100)
        self.assertEqual(ecosystem.service_port("TRAINING"), 18200)
        self.assertEqual(ecosystem.kaggle_owner(), "fixtureowner")
        self.assertEqual(ecosystem.docs_url(), "https://docs.example.test/hub")
        self.assertEqual(ecosystem.bind_host(), "127.0.0.1")

    def test_unknown_service_is_an_error(self) -> None:
        """An unknown service id fails loudly."""
        with self.assertRaisesRegex(ecosystem.EcosystemFileError, "Unknown service"):
            ecosystem.service_port("no-such-service")

    def test_default_path_is_under_the_workspace_root(self) -> None:
        """Without the override the file is found under the workspace root; missing names it."""
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {ecosystem.WORKSPACE_ROOT_ENV: tmp}
        ):
            os.environ.pop(ecosystem.ECOSYSTEM_FILE_ENV)
            expected = (
                Path(tmp).resolve() / "lemgendary-env-manager" / "ecosystem" / "ecosystem.json"
            )
            self.assertEqual(ecosystem.ecosystem_file(), expected)
            with self.assertRaises(ecosystem.EcosystemFileError) as ctx:
                ecosystem.ecosystem()
            self.assertIn(str(expected), str(ctx.exception))

    def test_invalid_file_names_the_key(self) -> None:
        """A wrongly typed value fails naming the key."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "ecosystem.json"
            raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
            raw["services"][1]["port"] = "8100"
            bad.write_text(json.dumps(raw), encoding="utf-8")
            with (
                patch.dict(os.environ, {ecosystem.ECOSYSTEM_FILE_ENV: str(bad)}),
                self.assertRaisesRegex(ecosystem.EcosystemFileError, "'port' must be int"),
            ):
                ecosystem.ecosystem()


class ConsumerTests(unittest.TestCase):
    """Ports, Kaggle owner and env-manager folder come from the file."""

    def test_cli_sidecar_address(self) -> None:
        """The CLI talks to the sidecar at the address in the file."""
        self.assertEqual(server_host(), "127.0.0.1")
        self.assertEqual(server_port(), 18100)
        self.assertEqual(sidecar_url("/api/health"), "http://127.0.0.1:18100/api/health")

    def test_kaggle_owner(self) -> None:
        """The default Kaggle owner is the file's ``kaggle_owner``."""
        self.assertEqual(default_owner(), "fixtureowner")
        self.assertEqual(default_handle("slug"), "fixtureowner/slug")

    def test_readme_links_the_owner_it_is_given(self) -> None:
        """README templates take the Kaggle owner as a parameter."""
        text = render_standard_readme(
            StandardReadmeFields(
                common=ReadmeCommon(
                    manifold_name="LemGendizedDemo",
                    models_markdown="",
                    structure_text="",
                    kaggle_owner="someowner",
                ),
                overview=StandardOverview(
                    resolved_desc="d", category="c", total_samples_display="1",
                    arch_base="a", resolved_obj="o",
                ),
                table_text="",
                manifest_text="",
            )
        )
        self.assertIn("https://www.kaggle.com/datasets/someowner/lemgendizeddemo", text)

    def test_lem_env_is_looked_up_in_the_env_manager_project(self) -> None:
        """lem-env discovery uses the env-manager project folder from the file."""
        resolve_lem_env.cache_clear()
        try:
            with (
                patch("core.cli_args.shutil.which", return_value=None),
                self.assertRaises(FileNotFoundError) as ctx,
            ):
                resolve_lem_env()
        finally:
            resolve_lem_env.cache_clear()
        self.assertIn("fixture-env-manager", str(ctx.exception))


class NotebookDelegationContractTests(unittest.TestCase):
    """The compiler delegates notebook generation to the training suite.

    ``core.manifold_compile._get_training_suite_notebook_generators`` returns
    this same function's result.
    """

    def test_generators_are_real_callables(self) -> None:
        """The training suite's notebook generators are importable callables."""
        suite_dir = ecosystem.project_dir(ecosystem.TRAINING_SERVICE)
        if not (suite_dir / "training" / "notebooks").is_dir():
            self.skipTest(f"training-suite checkout not found at {suite_dir}")
        generators = get_training_suite_notebook_generators()
        self.assertEqual(len(generators), 2)
        for gen in generators:
            self.assertTrue(callable(gen))
            self.assertTrue(gen.__module__.startswith("training.notebooks"))
            params = list(inspect.signature(gen).parameters)
            self.assertGreaterEqual(len(params), 3, f"{gen.__name__}{params}")


if __name__ == "__main__":
    unittest.main()
