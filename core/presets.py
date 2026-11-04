"""Compiler presets loader and validator module.

Loads, validates, and provides structured access to canonical compilation
profiles defined in presets.yaml for CLI, API, and Desktop GUI clients.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

_log = logging.getLogger(__name__)

_ROOT = (
    Path(__file__).resolve().parent.parent
    if Path(__file__).parent.name == "core"
    else Path(__file__).resolve().parent
)
DEFAULT_PRESETS_FILE = _ROOT / "presets.yaml"
if not DEFAULT_PRESETS_FILE.exists():
    DEFAULT_PRESETS_FILE = Path(__file__).resolve().parent / "presets.yaml"


class PresetsFileError(ValueError):
    """``presets.yaml`` is unreadable or not shaped as a presets document."""


class CompilerPreset(BaseModel):
    """Structured configuration for a compiler preset profile."""

    name: str = Field(..., description="Unique machine identifier for preset")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed explanation of profile rationale")
    image_format: str = Field(default="webp", description="Target image format (webp, jpeg, png)")
    image_quality: int = Field(default=92, ge=1, le=100, description="Target image quality (1-100)")
    target_quality: int | None = Field(
        default=95, ge=1, le=100, description="Target quality for ground truth pairs"
    )
    mask_format: str | None = Field(default="webp-lossless", description="Target mask format")
    min_resolution: int | None = Field(
        default=256, ge=64, description="Minimum resolution floor in pixels"
    )
    resampling: str | None = Field(
        default="lanczos", description="Resampling filter (lanczos, bilinear, bicubic)"
    )
    vetting_enabled: bool = Field(
        default=False, description="Whether NIMA aesthetic vetting gate is active"
    )
    labeling_enabled: bool = Field(
        default=False, description="Whether YOLO auto-labeling is active"
    )
    hardlink_gate: bool = Field(
        default=True, description="Whether NTFS hardlinking is enforced for identical targets"
    )
    containers: list[str] = Field(
        default_factory=list, description="Target container formats (parquet, mds, litdata)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert preset model to dictionary representation."""
        return self.model_dump()


def _read_presets_document(config_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Read presets.yaml and return its path and raw root mapping."""
    target_path = config_path or DEFAULT_PRESETS_FILE
    if not target_path.exists():
        _log.error("Presets configuration file not found: %s", target_path)
        raise FileNotFoundError(f"Presets file not found at {target_path}")

    try:
        raw_text = target_path.read_text(encoding="utf-8")
        raw_data = yaml.safe_load(raw_text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        _log.error("Failed to parse YAML from %s: %s", target_path, exc)
        raise PresetsFileError(f"Invalid YAML in {target_path}: {exc}") from exc

    if not isinstance(raw_data, dict) or "presets" not in raw_data:
        _log.error("Missing 'presets' root key in %s", target_path)
        raise PresetsFileError(f"Presets file {target_path} must contain a 'presets' root key")
    return target_path, raw_data


def load_presets(config_path: Path | None = None) -> dict[str, CompilerPreset]:
    """Load and parse compiler presets from YAML file.

    Args:
        config_path: Optional custom path to presets.yaml. Defaults to DEFAULT_PRESETS_FILE.

    Returns:
        Dictionary mapping preset identifiers to CompilerPreset models.

    Raises:
        FileNotFoundError: If the presets configuration file does not exist.
        ValueError: If the file format or schema validation fails.
    """
    target_path, raw_data = _read_presets_document(config_path)
    loaded: dict[str, CompilerPreset] = {}
    presets_dict = raw_data["presets"]
    if not isinstance(presets_dict, dict):
        raise PresetsFileError("The 'presets' key must be a dictionary of preset configurations")

    for key, data in presets_dict.items():
        if not isinstance(data, dict):
            _log.warning("Skipping invalid preset entry for key '%s'", key)
            continue
        try:
            preset_model = CompilerPreset.model_validate(data)
            loaded[key] = preset_model
        except ValidationError as exc:
            _log.error("Validation failed for preset '%s': %s", key, exc)
            raise ValueError(f"Preset '{key}' schema validation failed: {exc}") from exc

    _log.debug("Successfully loaded %d compiler presets from %s", len(loaded), target_path)
    return loaded


def get_preset(name: str, config_path: Path | None = None) -> CompilerPreset:
    """Retrieve a specific compiler preset by identifier.

    Args:
        name: The preset identifier string (e.g. 'quality-vision').
        config_path: Optional custom path to presets.yaml.

    Returns:
        CompilerPreset instance.

    Raises:
        KeyError: If the requested preset is not defined.
    """
    all_presets = load_presets(config_path)
    if name not in all_presets:
        available = ", ".join(sorted(all_presets.keys()))
        raise KeyError(f"Unknown compiler preset '{name}'. Available presets: {available}")
    return all_presets[name]


def get_default_preset_name(config_path: Path | None = None) -> str:
    """Return the preset named by the top-level `default_preset` key of presets.yaml.

    presets.yaml is the single source of truth for the default, so API request
    models never hardcode a preset name that can drift out of the inventory.

    Raises:
        ValueError: If `default_preset` is missing or does not name a defined preset.
    """
    target_path, raw_data = _read_presets_document(config_path)
    default_name = raw_data.get("default_preset")
    if not isinstance(default_name, str) or not default_name:
        raise ValueError(f"Presets file {target_path} must define a non-empty 'default_preset' key")
    available = load_presets(config_path)
    if default_name not in available:
        raise ValueError(
            f"default_preset '{default_name}' in {target_path} is not a defined preset. "
            f"Available presets: {', '.join(sorted(available))}"
        )
    return default_name


def list_presets(config_path: Path | None = None) -> dict[str, CompilerPreset]:
    """Retrieve all available compiler presets."""
    return load_presets(config_path)


def validate_preset(preset_data: dict[str, Any]) -> bool:
    """Validate raw dictionary data against the CompilerPreset schema without raising."""
    try:
        CompilerPreset.model_validate(preset_data)
        return True
    except ValidationError as exc:
        _log.debug("Preset dictionary failed validation: %s", exc)
        return False
