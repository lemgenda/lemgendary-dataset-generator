"""Tests for Prompt 11: SUITE_PAT secret resolution and notebook/document generation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from sources.gh import clone_repository

from core.secrets import get_suite_pat, load_secrets_records


def test_suite_pat_loaded_from_github_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm SUITE_PAT is loaded from .secrets.yaml under github-primary."""
    fake_secrets = {
        "secrets": [
            {
                "id": "github-primary",
                "service": "github",
                "label": "GitHub Personal Access Token",
                "secret_value": "suite_secret_test_token_12345",
                "is_default": True,
            }
        ]
    }
    secrets_file = tmp_path / ".secrets.yaml"
    secrets_file.write_text(yaml.safe_dump(fake_secrets), encoding="utf-8")

    monkeypatch.delenv("SUITE_PAT", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)

    with patch("core.secrets.find_secrets_yaml", return_value=secrets_file):
        records = load_secrets_records()
        assert len(records) == 1
        assert records[0]["id"] == "github-primary"

        token = get_suite_pat()
        assert token == "suite_secret_test_token_12345"


def test_suite_pat_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm SUITE_PAT in environment overrides .secrets.yaml value."""
    fake_secrets = {
        "secrets": [
            {
                "id": "github-primary",
                "service": "github",
                "secret_value": "vault_token",
            }
        ]
    }
    secrets_file = tmp_path / ".secrets.yaml"
    secrets_file.write_text(yaml.safe_dump(fake_secrets), encoding="utf-8")

    monkeypatch.setenv("SUITE_PAT", "override_suite_token")
    with patch("core.secrets.find_secrets_yaml", return_value=secrets_file):
        token = get_suite_pat()
        assert token == "override_suite_token"


def test_legacy_github_pat_emits_warning_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Confirm legacy GITHUB_PAT triggers warning and is not used as active token."""
    empty_secrets: dict[str, list[dict[str, str]]] = {"secrets": []}
    secrets_file = tmp_path / ".secrets.yaml"
    secrets_file.write_text(yaml.safe_dump(empty_secrets), encoding="utf-8")

    monkeypatch.delenv("SUITE_PAT", raising=False)
    monkeypatch.setenv("GITHUB_PAT", "legacy_token_attempt")

    with patch("core.secrets.find_secrets_yaml", return_value=secrets_file):
        token = get_suite_pat()
        assert token is None
        assert "GITHUB_PAT is deprecated and unsupported" in caplog.text


def test_gh_source_clone_uses_suite_pat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm sources.gh applies SUITE_PAT via GIT_ASKPASS environment."""
    monkeypatch.setenv("SUITE_PAT", "test_suite_pat_token")

    captured_env: dict[str, str] = {}

    def fake_subprocess_run(_cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return SimpleNamespace(returncode=0)

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        ret = clone_repository("https://github.com/lemgenda/fake-repo.git", str(tmp_path / "out"))
        assert ret == 0
        assert captured_env.get("LEMGENDARY_GIT_TOKEN") == "test_suite_pat_token"
        assert "GIT_ASKPASS" in captured_env
