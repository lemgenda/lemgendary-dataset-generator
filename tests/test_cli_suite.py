"""Unit and integration tests for the unified Dataset Compiler CLI.

Covers top-level commands, command groups, flags, exit codes, and help messages.
"""

from __future__ import annotations

import unittest

from typer.testing import CliRunner

from cli import app
from core.cli_args import PROJECT_NAME, __version__

runner = CliRunner()


class TestCompilerCLIBasics(unittest.TestCase):
    """Test top-level CLI flags, version commands, and general discovery."""

    def test_version_subcommand(self) -> None:
        """'version' subcommand prints project name and version."""
        result = runner.invoke(app, ["version"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(__version__, result.output)
        self.assertIn(PROJECT_NAME, result.output)

    def test_version_flag_long(self) -> None:
        """'--version' flag prints version and exits 0."""
        result = runner.invoke(app, ["--version"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(__version__, result.output)
        self.assertIn(PROJECT_NAME, result.output)

    def test_version_flag_short(self) -> None:
        """'-V' flag prints version and exits 0."""
        result = runner.invoke(app, ["-V"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(__version__, result.output)
        self.assertIn(PROJECT_NAME, result.output)

    def test_root_help(self) -> None:
        """'--help' displays top-level commands and command groups."""
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("LemGendary Dataset Compiler Suite", result.output)
        expected_cmds = (
            "compile", "verify", "sync", "server", "presets", "repair", "forex", "llm"
        )
        for expected in expected_cmds:
            self.assertIn(expected, result.output)

    def test_no_args_shows_help(self) -> None:
        """Invoking without arguments shows usage help."""
        result = runner.invoke(app, [])
        self.assertIn("Usage:", result.output)


class TestCompilerCLICommands(unittest.TestCase):
    """Test command group dispatch and help trees."""

    def test_presets_list_help(self) -> None:
        """'presets list --help' shows preset command help."""
        result = runner.invoke(app, ["presets", "list", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--help", result.output)

    def test_server_status_help(self) -> None:
        """'server status --help' displays sidecar daemon probe options."""
        result = runner.invoke(app, ["server", "status", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--json", result.output)

    def test_sync_push_help(self) -> None:
        """'sync push --help' displays Kaggle upload flags."""
        result = runner.invoke(app, ["sync", "push", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--force", result.output)
        self.assertIn("--notes", result.output)

    def test_sync_pull_help(self) -> None:
        """'sync pull --help' displays Kaggle download flags."""
        result = runner.invoke(app, ["sync", "pull", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--version", result.output)

    def test_verify_help(self) -> None:
        """'verify --help' displays manifold verification options."""
        result = runner.invoke(app, ["verify", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--model", result.output)
        self.assertIn("--all", result.output)


if __name__ == "__main__":
    unittest.main()
