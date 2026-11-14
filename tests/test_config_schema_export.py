"""Tests for unified_data.yaml canonical schema, JSON Schema export, and error messages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config_schema import UnifiedData, UnifiedDataError, load_unified_data

_ROOT = Path(__file__).resolve().parent.parent


def test_unified_data_yaml_validates() -> None:
    """Verify the repository's unified_data.yaml passes canonical validation."""
    cfg = load_unified_data(_ROOT / "unified_data.yaml")
    assert len(cfg.datasets) >= 21


def test_unified_data_json_schema_matches_model() -> None:
    """Verify unified_data.schema.json stays in sync with UnifiedDataConfig."""
    schema_path = _ROOT / "unified_data.schema.json"
    assert schema_path.exists()
    on_disk = json.loads(schema_path.read_text(encoding="utf-8"))
    assert on_disk == UnifiedData.model_json_schema()


def test_invalid_unified_data_fails_with_clear_error(tmp_path: Path) -> None:
    """Verify invalid configs raise UnifiedDataError or ValidationError."""
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(UnifiedDataError):
        load_unified_data(empty_file)

    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text(
        "_registry_metadata:\n  version: 1.0.0\ndatasets:\n  x:\n    name: X\n"
        "    task: quality\n    val_split: 1.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_unified_data(bad_file)
