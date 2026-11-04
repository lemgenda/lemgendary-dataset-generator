"""
LemGendary Dataset Compiler: one version string.

``core.cli_args.__version__`` (served by the API, ``cli.py version`` and
``openapi.json``) must match the README status table and its newest
changelog entry.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from core.cli_args import __version__

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestVersionSync(unittest.TestCase):
    """The code, README and OpenAPI contract report the same version."""

    def setUp(self) -> None:
        self.readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    def test_readme_status_table(self) -> None:
        """The README status table shows __version__."""
        match = re.search(r"\| \*\*Version\*\* \| `v([\d.]+)-STABLE` \|", self.readme)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), __version__)

    def test_newest_changelog_entry(self) -> None:
        """The newest README changelog entry is __version__."""
        changelog = self.readme.split("## Changelog", 1)[1]
        match = re.search(r"^### v([\d.]+) ", changelog, re.MULTILINE)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), __version__)

    def test_openapi_contract(self) -> None:
        """openapi.json carries __version__."""
        spec = json.loads((PROJECT_ROOT / "openapi.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["info"]["version"], __version__)


if __name__ == "__main__":
    unittest.main()
