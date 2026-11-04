"""Resolution rules feeding ``resolve_manifold_context``.

Determines the authoritative task of a manifold, its task profile (category,
architecture, objective, description, model profile) and the per-source sample
tally used by the lineage tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from core.docs.context_types import SpecInputs, TaskProfile
from core.docs.metadata import (
    MODELS_META,
    TASK_ARCH_BASE,
    TASK_META,
    format_source,
)
from core.docs.scanner import scan_forex_manifold
from core.task_taxonomy import taxonomy

logger = logging.getLogger(__name__)

SourceTally = dict[str, dict[str, Any]]

# Ordered (name keywords, task) rules used when no explicit task is known.
_NAME_TASK_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("aihelper", "ai_helper"), "ai_helper"),
    (("forex",), "forex"),
    (("authenticity",), "authenticity"),
    (
        ("restoration", "ffanet", "mirnet", "mprnet", "nafnet", "film", "codeformer"),
        "restoration",
    ),
    (("ultrazoom", "superresolution"), "super-resolution"),
    (("parsenet", "segmentation"), "segmentation"),
    (("retinaface", "yolo", "detection"), "detection"),
    (("upn", "parameter"), "parameter_prediction"),
    (("classification", "nsfw"), "classification"),
)

# Ordered (name keywords, objective, description) refinements for restoration.
_RESTORATION_COPY: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("dehazing", "indoor", "outdoor"),
        (
            "Remove atmospheric haze and smoke to recover high-contrast ground truth scene"
            " radiance."
        ),
        "Standardized paired dataset for single-image dehazing and transmission recovery.",
    ),
    (
        ("deraining",),
        (
            "Remove dense rain streaks and rain accumulation while preserving background"
            " details."
        ),
        "Standardized paired dataset for single-image deraining and streak removal.",
    ),
    (
        ("deblurring", "debluring"),
        "Remove motion and optical blur to restore structural sharpness and fine edges.",
        "Standardized paired dataset for single-image deblurring and sharpness recovery.",
    ),
    (
        ("denoising",),
        (
            "Eliminate real sensor noise and synthetic additive noise while preserving"
            " high-frequency textures."
        ),
        "Standardized paired dataset for single-image denoising and noise-level estimation.",
    ),
    (
        ("exposure", "lowlight"),
        (
            "Correct extreme under/over-exposure and enhance visual fidelity in low-light"
            " environments."
        ),
        "Standardized paired dataset for exposure correction and low-light enhancement.",
    ),
    (
        ("film",),
        (
            "Restore degraded vintage film frames by removing scratches, color fading, and"
            " archival dust."
        ),
        (
            "Standardized paired dataset for vintage film restoration and historical frame"
            " recovery."
        ),
    ),
    (
        ("codeformer",),
        (
            "Restore severely degraded facial portraits using codebook lookup and transformer"
            " priors."
        ),
        (
            "High-fidelity blind face restoration manifold with synthetic degradation and"
            " aligned targets."
        ),
    ),
)

_LEGACY_SOURCE_TAGS = ["old", "unknown", "none", "legacy"]


def _cached_yaml_task(output_root: Path) -> str | None:
    """Return the non-'quality' task recorded in an existing dataset_info.yaml."""
    yaml_path = output_root / "dataset_info.yaml"
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            info_data = yaml.safe_load(f) or {}
            cached_t = info_data.get("task", info_data.get("dataset_type"))
            if cached_t and cached_t != "quality":
                return str(cached_t)
    except (yaml.YAMLError, OSError) as exc:
        logger.debug("Could not read task from dataset_info.yaml: %s", exc)
    return None


def _explicit_task(
    manifold_name: str,
    unified_entry: dict[str, Any],
    overrides: dict[str, Any] | None,
    final_index: list[dict[str, Any]] | None,
) -> str | None:
    """Return the task stated by overrides, the index, the task map or unified_data.yaml."""
    candidates: list[Any] = []
    if overrides:
        candidates += [overrides.get("dataset_type"), overrides.get("task")]
    if final_index:
        candidates.append(final_index[0].get("task"))
    for candidate in candidates:
        if candidate:
            return str(candidate)
    documentation_task = taxonomy().documentation_tasks.get(manifold_name)
    if documentation_task:
        return documentation_task
    if unified_entry.get("task"):
        return str(unified_entry["task"])
    return None


def _task_from_name(manifold_name: str) -> str:
    """Infer the task from keywords in the manifold name ('quality' when nothing matches)."""
    name_lower = manifold_name.lower()
    for keywords, task in _NAME_TASK_RULES:
        if any(k in name_lower for k in keywords):
            return task
    return "quality"


def resolve_task(
    manifold_name: str,
    output_root: Path,
    unified_entry: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    final_index: list[dict[str, Any]] | None = None,
) -> str:
    """Determine authoritative task classification for the manifold."""
    explicit = _explicit_task(manifold_name, unified_entry, overrides, final_index)
    if explicit is not None:
        return explicit
    return _cached_yaml_task(output_root) or _task_from_name(manifold_name)


def resolve_model_profile(manifold_name: str) -> dict[str, Any] | None:
    """Find the model profile bound to ``manifold_name`` in models_metadata.yaml."""
    for m_info in MODELS_META.values():
        if isinstance(m_info, dict) and manifold_name in m_info.get("datasets", []):
            return m_info
    stripped = manifold_name.lower().replace("lemgendized", "").replace("large", "")
    for m_key, m_info in MODELS_META.items():
        if isinstance(m_info, dict) and m_key.lower() in stripped:
            return m_info
    return None


def resolve_task_profile(manifold_name: str, task: str, spec: SpecInputs) -> TaskProfile:
    """Resolve category, architecture, objective, description and model profile."""
    task_meta_entry = TASK_META.get(task, {})
    category = spec.pick_truthy(
        "category", task_meta_entry.get("category", "General Machine Learning Dataset")
    )
    arch_base = TASK_ARCH_BASE.get(task, "Deep Convolutional / Transformer Architecture")
    objective = task_meta_entry.get("obj", "Train machine learning models with high precision.")
    desc = task_meta_entry.get("desc", f"High-fidelity dataset manifold for {category}.")

    model_profile = resolve_model_profile(manifold_name)
    if model_profile and model_profile.get("arch"):
        arch_base = str(model_profile["arch"])

    if task == "restoration":
        name_lower = manifold_name.lower()
        for keywords, refined_obj, refined_desc in _RESTORATION_COPY:
            if any(k in name_lower for k in keywords):
                objective, desc = refined_obj, refined_desc
                break

    return TaskProfile(
        category=category,
        arch_base=arch_base,
        objective=objective,
        description=desc,
        model_profile=model_profile,
    )


def read_yaml_count(output_root: Path) -> int:
    """Return the ``count`` recorded in dataset_info.yaml, or 0 when unavailable."""
    yaml_path = output_root / "dataset_info.yaml"
    if not yaml_path.exists():
        return 0
    try:
        with open(yaml_path, "r", encoding="utf-8") as yf:
            yinfo = yaml.safe_load(yf) or {}
            return int(yinfo.get("count", 0))
    except (yaml.YAMLError, OSError, ValueError) as exc:
        logger.debug("Could not read count from dataset_info.yaml: %s", exc)
    return 0


def _count_lines_or_zero(path: Path) -> int:
    """Count lines of ``path``; 0 when it is missing or unreadable."""
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _tally_index_items(items: list[dict[str, Any]], derive_legacy: bool) -> SourceTally:
    """Count train/val/total samples per formatted source over index items."""
    sources: SourceTally = {}
    for item in items:
        raw_src = item.get("source", "")
        if derive_legacy and (not raw_src or raw_src.lower() in _LEGACY_SOURCE_TAGS):
            parts = item.get("name", "").split("_")
            raw_src = "_".join(parts[1:-1]) if len(parts) >= 3 else "Unknown"
        bucket = sources.setdefault(format_source(raw_src), {"train": 0, "val": 0, "total": 0})
        bucket["total"] += 1
        sp = item.get("split", "train")
        if sp in ["train", "val"]:
            bucket[sp] += 1
    return sources


def _tally_forex(output_root: Path) -> tuple[SourceTally, int]:
    """Tally Forex samples from the shard scan (falling back to dataset_info.yaml)."""
    forex_data = scan_forex_manifold(output_root)
    total_samples = sum(forex_data.get("samples_per_year", {}).values())
    if total_samples == 0:
        total_samples = read_yaml_count(output_root)
    sources = {
        "MetaTrader 5 Terminal Native Cache (2019-2026)": {
            "train": int(total_samples * 0.88) if total_samples > 0 else 0,
            "val": int(total_samples * 0.12) if total_samples > 0 else 0,
            "total": total_samples,
        }
    }
    return sources, total_samples


def _tally_ai_helper(output_root: Path, unified_entry: dict[str, Any]) -> tuple[SourceTally, int]:
    """Tally instruction pairs of the AI helper JSONL splits."""
    train_file = output_root / "data" / "ai_helper_train.jsonl"
    val_file = output_root / "data" / "ai_helper_val.jsonl"
    if not train_file.exists():
        train_file = output_root / "ai_helper_train.jsonl"
        val_file = output_root / "ai_helper_val.jsonl"
    tr_cnt = _count_lines_or_zero(train_file)
    vl_cnt = _count_lines_or_zero(val_file)
    total_samples = tr_cnt + vl_cnt
    if total_samples == 0:
        total_samples = read_yaml_count(output_root)
    sources: SourceTally = {}
    for p in unified_entry.get("provenance_sources", ["LemGendary AI Documentation Hub"]):
        sources[format_source(p)] = {
            "train": tr_cnt or int(total_samples * 0.85),
            "val": vl_cnt or int(total_samples * 0.15),
            "total": total_samples,
        }
    return sources, total_samples


def _tally_index_json(output_root: Path) -> tuple[SourceTally, int]:
    """Tally samples recorded in an existing index.json."""
    try:
        with open(output_root / "index.json", "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        return _tally_index_items(idx_data, derive_legacy=False), len(idx_data)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Could not parse index.json for sample counts: %s", exc)
    return {}, 0


def _redistribute_provenance(
    sources: SourceTally, total_samples: int, manifold_name: str, unified_entry: dict[str, Any]
) -> SourceTally:
    """Spread samples over unified_data provenance when sources are empty or self-referential."""
    has_only_self = bool(
        sources and all("manifold" in k.lower() or k == manifold_name for k in sources)
    )
    if (sources and not has_only_self) or not unified_entry.get("provenance_sources"):
        return sources
    prov_list = unified_entry["provenance_sources"]
    per_src = total_samples // len(prov_list) if total_samples and prov_list else 0
    val_c = int(per_src * 0.12)
    train_c = per_src - val_c
    return {format_source(p): {"train": train_c, "val": val_c, "total": per_src} for p in prov_list}


def tally_sources(
    output_root: Path,
    task: str,
    manifold_name: str,
    unified_entry: dict[str, Any],
    final_index: list[dict[str, Any]] | None,
) -> tuple[SourceTally, int]:
    """Return the per-source sample tally and the total sample count of the manifold."""
    sources: SourceTally = {}
    total_samples = 0
    if final_index:
        sources = _tally_index_items(final_index, derive_legacy=True)
        total_samples = len(final_index)
    elif task == "forex":
        sources, total_samples = _tally_forex(output_root)
    elif task == "ai_helper":
        sources, total_samples = _tally_ai_helper(output_root, unified_entry)
    elif (output_root / "index.json").exists():
        sources, total_samples = _tally_index_json(output_root)

    if total_samples == 0:
        total_samples = read_yaml_count(output_root)

    sources = _redistribute_provenance(sources, total_samples, manifold_name, unified_entry)
    return sources, total_samples
