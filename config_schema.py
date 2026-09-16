"""
LemGendary Dataset Compiler — Configuration Schema

Pydantic v2 schema for unified_data.yaml. Validates every field that
manifold_compile.py, manifold_reduce.py, manifold_sync.py, and
modernize_manifold.py consume — catching typos and constraint violations
before any filesystem mutation.

Phase 1.1 of the 2026 modernization roadmap.

Usage:
    from config_schema import load_unified_data
    data = load_unified_data("./unified_data.yaml")

    # Access metadata
    print(data.registry_metadata.name_prefix)

    # Access a dataset entry
    nima = data.datasets["nima_aesthetic"]
    print(nima.name, nima.val_split)

    # Legacy-shape dict for downstream code expecting the original keys
    legacy = data.to_legacy_dict()

CLI:
    python config_schema.py         # validates ./unified_data.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─── Ref-level ──────────────────────────────────────────────────────────────
class SourceRef(BaseModel):
    """A single source dataset reference inside a manifold definition."""

    # Extra allowed: future refs may carry additional metadata (sha, size, etc.)
    model_config = ConfigDict(extra="allow")

    ref: str
    tag: Literal[
        "sfw", "nsfw", "anime_nsfw", "general_nsfw", "sfw_baseline"
    ] = "sfw"


# ─── Registry-level ─────────────────────────────────────────────────────────
class GlobalConstraints(BaseModel):
    """Global size bounds applied to every non-forex manifold."""

    model_config = ConfigDict(extra="allow")

    min_size_gb: float = Field(default=6.0, ge=0.0)
    max_size_gb: float = Field(default=190.0, gt=0.0)


class RegistryMetadata(BaseModel):
    """The `_registry_metadata` block at the top of unified_data.yaml."""

    model_config = ConfigDict(extra="allow")

    version: str
    name_prefix: str
    name_suffix: str
    output_folder_name: str
    specialization: str | None = None
    global_constraints: GlobalConstraints = Field(default_factory=GlobalConstraints)


# ─── Per-dataset policies ───────────────────────────────────────────────────
class ImageFormatPolicy(BaseModel):
    """Image transcoding policy (Phase 3 consumer)."""

    model_config = ConfigDict(extra="allow")

    format: Literal["webp", "jpeg", "png", "keep"] = "webp"
    quality: int = Field(default=92, ge=1, le=100)
    mask_format: Literal["webp-lossless", "png"] = "webp-lossless"
    target_quality: int = Field(default=95, ge=1, le=100)


class ContainerPolicy(BaseModel):
    """Container format policy (Phase 4 consumer)."""

    model_config = ConfigDict(extra="allow")

    primary: Literal[
        "directory", "mds", "litdata", "webdataset", "parquet"
    ] = "directory"
    extra: list[str] = Field(default_factory=list)
    preserve_hardlinks: bool = True


# ─── Dataset entry ──────────────────────────────────────────────────────────
class DatasetEntry(BaseModel):
    """A single manifold definition inside `datasets:`."""

    # Extra forbidden here so typos in the YAML are caught immediately.
    # All legitimate fields are enumerated below.
    model_config = ConfigDict(extra="forbid")

    # Identity & packaging
    name: str
    kaggle_ref: str | None = None
    kaggle_dataset_urls: list[str] = Field(default_factory=list)

    # Sampling
    val_split: float = Field(default=0.12, ge=0.0, lt=1.0)
    nsfw_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # Pipeline toggles
    labeling: bool = True
    vetting: bool = True

    # Task routing
    task_override: str | None = None
    task: str | None = None

    # Domain classification
    dataset_type: Literal["vision", "forex", "diffusion", "vlm"] = "vision"
    acquisition_mode: Literal[
        "hf", "kaggle", "gh", "gdrive", "mt5_terminal", "local"
    ] = "kaggle"

    # Sources
    refs: list[SourceRef] = Field(default_factory=list)

    # Policies (Phase 3 / Phase 4 consumers — optional; defaults apply)
    image_format: ImageFormatPolicy = Field(default_factory=ImageFormatPolicy)
    container: ContainerPolicy = Field(default_factory=ContainerPolicy)

    # Forex-specific (only meaningful when dataset_type == "forex")
    pairs: list[str] | None = None
    timeframe_rungs: list[int] | None = None
    start_date: str | None = None
    lookback_bars: int | None = Field(default=None, ge=1)
    format: str | None = None
    compression: str | None = None
    category: str | None = None
    storage_size_approx: str | None = None
    description: str | None = None


# ─── Root document ──────────────────────────────────────────────────────────
class UnifiedData(BaseModel):
    """Root model for unified_data.yaml."""

    model_config = ConfigDict(extra="forbid")

    registry_metadata: RegistryMetadata
    datasets: dict[str, DatasetEntry] = Field(default_factory=dict)

    # `_registry_metadata` is a valid Python identifier (leading underscore is
    # reserved by pydantic), so remap it here before validation.
    @model_validator(mode="before")
    @classmethod
    def _remap_registry_metadata(cls, data: Any) -> Any:
        if isinstance(data, dict) and "_registry_metadata" in data:
            data = dict(data)
            data["registry_metadata"] = data.pop("_registry_metadata")
        return data

    # Cross-field rules
    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> "UnifiedData":
        for key, entry in self.datasets.items():
            if entry.dataset_type == "forex":
                if not entry.pairs:
                    raise ValueError(
                        f"Dataset '{key}': dataset_type='forex' requires 'pairs'."
                    )
                if not entry.timeframe_rungs:
                    raise ValueError(
                        f"Dataset '{key}': dataset_type='forex' requires 'timeframe_rungs'."
                    )
            if entry.acquisition_mode == "mt5_terminal" and entry.dataset_type != "forex":
                raise ValueError(
                    f"Dataset '{key}': acquisition_mode='mt5_terminal' "
                    f"requires dataset_type='forex' (got '{entry.dataset_type}')."
                )
        return self

    # ─── Backward-compatibility helper ──────────────────────────────────────
    def to_legacy_dict(self) -> dict:
        """
        Return a plain dict with the original `_registry_metadata` key shape.

        Downstream code (compiler_core.py, manifold_compile.py) reads
        YAML_DATA["_registry_metadata"] directly. This helper preserves that
        access pattern while routing everything through pydantic.
        """
        d = self.model_dump(mode="python")
        d["_registry_metadata"] = d.pop("registry_metadata")
        return d


# ─── Loader ─────────────────────────────────────────────────────────────────
def load_unified_data(path: Path | str) -> UnifiedData:
    """
    Load and validate unified_data.yaml.

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the file is empty or fails schema validation
        pydantic.ValidationError: for schema violations (with clear field paths)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"unified_data.yaml not found at {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"unified_data.yaml is empty: {p}")
    if not isinstance(raw, dict):
        raise ValueError(
            f"unified_data.yaml root must be a mapping, got {type(raw).__name__}"
        )
    return UnifiedData.model_validate(raw)


# ─── CLI ────────────────────────────────────────────────────────────────────
def _cli_validate() -> int:
    """CLI entry point for `lemgendary config validate`."""
    here = Path(__file__).parent
    candidates = [
        here / "unified_data.yaml",
        here.parent / "lemgendary-datasets" / "unified_data.yaml",
    ]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        print("[ERROR] unified_data.yaml not found in any of:")
        for c in candidates:
            print(f"  -> {c}")
        return 1

    try:
        data = load_unified_data(target)
    except Exception as e:
        print(f"[ERROR] Configuration validation failed for {target}:")
        print(f"  {type(e).__name__}: {e}")
        return 1

    rm = data.registry_metadata
    print(f"[OK] Configuration validated: {target}")
    print(f"     version={rm.version}  prefix='{rm.name_prefix}'  suffix='{rm.name_suffix}'")
    print(f"     output_folder='{rm.output_folder_name}'")
    print(f"     global_constraints: min={rm.global_constraints.min_size_gb}GB  max={rm.global_constraints.max_size_gb}GB")
    print(f"     datasets: {len(data.datasets)}")
    print()
    for key, entry in sorted(data.datasets.items()):
        task = entry.task or entry.task_override or "(inferred)"
        type_ = entry.dataset_type
        refs = len(entry.refs)
        print(f"       - {key:<42} type={type_:<10} task={task:<24} refs={refs}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli_validate())