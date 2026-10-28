"""LemGendary Dataset Documentation Context Resolver.

Extracts, standardizes, and unifies metadata, file inventories, task types,
and model training profiles for any vision or temporal dataset manifold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any
import yaml

from core.docs.metadata import (
    MANIFOLD_TASK_MAP,
    MODELS_META,
    TASK_ARCH_BASE,
    TASK_META,
    UNIFIED_DATA,
    format_source,
)
from core.docs.scanner import scan_forex_manifold

logger = logging.getLogger(__name__)


@dataclass
class ManifoldContext:
    """Encapsulates all resolved metadata and on-disk assets for a dataset manifold."""

    manifold_path: Path
    manifold_name: str
    slug: str
    task: str
    category: str
    arch_base: str
    objective: str
    description: str
    title: str
    subtitle: str
    license_name: str
    keywords: list[str]
    is_private: bool
    update_frequency: str
    author_name: str
    author_bio: str
    cover_image_name: str | None
    cover_image_path: Path | None
    provenance_sources: list[str]
    collection_methodology: str
    citations: list[str]
    coverage: dict[str, Any]
    unified_entry: dict[str, Any]
    model_profile: dict[str, Any] | None
    total_samples: int
    sources: dict[str, dict[str, Any]]
    root_files: list[Path] = field(default_factory=list)
    root_dirs: list[Path] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)


def find_unified_entry(manifold_name: str) -> dict[str, Any]:
    """Find matching dataset specification from unified_data.yaml by exact or normalized name."""
    datasets = UNIFIED_DATA.get("datasets", {})
    if not isinstance(datasets, dict):
        return {}

    name_lower = manifold_name.lower()
    norm_target = name_lower.replace("_", "").replace("large", "")

    # 1. Exact modernized_folder match (SSOT)
    for _d_key, d_info in datasets.items():
        if not isinstance(d_info, dict):
            continue
        mod_folder = str(d_info.get("modernized_folder", "")).lower()
        if mod_folder == name_lower:
            return d_info

    # 2. Exact name match
    for d_key, d_info in datasets.items():
        if not isinstance(d_info, dict):
            continue
        d_name = str(d_info.get("name", d_key)).lower()
        if f"lemgendized{d_name}" == name_lower or f"lemgendized{d_name}large" == name_lower:
            return d_info

    # 3. Normalized key or name match
    for d_key, d_info in datasets.items():
        if not isinstance(d_info, dict):
            continue
        norm_key = d_key.lower().replace("_", "")
        d_name = str(d_info.get("name", d_key)).lower().replace("_", "")
        if norm_key == norm_target or d_name == norm_target:
            return d_info
        if norm_key in norm_target or norm_target in norm_key:
            return d_info

    return {}


def _resolve_task(
    manifold_name: str,
    output_root: Path,
    unified_entry: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    final_index: list[dict[str, Any]] | None = None,
) -> str:
    """Determine authoritative task classification for the manifold."""
    if overrides:
        if overrides.get("dataset_type"):
            return str(overrides["dataset_type"])
        if overrides.get("task"):
            return str(overrides["task"])

    if final_index and len(final_index) > 0 and final_index[0].get("task"):
        return str(final_index[0]["task"])

    if manifold_name in MANIFOLD_TASK_MAP:
        return MANIFOLD_TASK_MAP[manifold_name]

    if unified_entry.get("task"):
        return str(unified_entry["task"])

    yaml_path = output_root / "dataset_info.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                info_data = yaml.safe_load(f) or {}
                cached_t = info_data.get("task", info_data.get("dataset_type"))
                if cached_t and cached_t != "quality":
                    return str(cached_t)
        except (yaml.YAMLError, OSError) as exc:
            logger.debug("Could not read task from dataset_info.yaml: %s", exc)

    name_lower = manifold_name.lower()
    if "aihelper" in name_lower or "ai_helper" in name_lower:
        return "ai_helper"
    if "forex" in name_lower:
        return "forex"
    if "authenticity" in name_lower:
        return "authenticity"
    if "restoration" in name_lower or any(
        k in name_lower for k in ["ffanet", "mirnet", "mprnet", "nafnet", "film", "codeformer"]
    ):
        return "restoration"
    if "ultrazoom" in name_lower or "superresolution" in name_lower:
        return "super-resolution"
    if "parsenet" in name_lower or "segmentation" in name_lower:
        return "segmentation"
    if "retinaface" in name_lower or "yolo" in name_lower or "detection" in name_lower:
        return "detection"
    if "upn" in name_lower or "parameter" in name_lower:
        return "parameter_prediction"
    if "classification" in name_lower or "nsfw" in name_lower:
        return "classification"

    return "quality"


def _format_subtitle(raw_subtitle: str, fallback_cat: str) -> str:
    """Format and guarantee a subtitle length strictly between 20 and 80 characters for Kaggle API."""
    sub = raw_subtitle.strip().replace("\uFFFD", "-").replace("—", "-").replace("–", "-")
    if not sub:
        sub = f"High-Fidelity Training Manifold for {fallback_cat}"

    if len(sub) > 80:
        truncated = sub[:77]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        sub = truncated + "..."
        if len(sub) > 80:
            sub = sub[:80]

    if len(sub) < 20:
        sub = f"High-Fidelity {fallback_cat} Training Manifold"
        if len(sub) > 80:
            sub = sub[:80]

    return sub


def resolve_manifold_context(
    output_root: str | Path,
    pascal_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    final_index: list[dict[str, Any]] | None = None,
) -> ManifoldContext:
    """Resolve a unified ManifoldContext with full metadata and physical asset inventory."""
    output_root = Path(output_root).resolve()
    manifold_name = pascal_name or output_root.name
    clean_overrides = dict(overrides) if overrides else {}

    unified_entry = find_unified_entry(manifold_name)
    task = _resolve_task(manifold_name, output_root, unified_entry, clean_overrides, final_index)

    # 1. Physical inventory
    root_files: list[Path] = []
    root_dirs: list[Path] = []
    cover_image_path: Path | None = None
    cover_image_name: str | None = None

    if output_root.exists() and output_root.is_dir():
        for item in sorted(output_root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                root_dirs.append(item)
            else:
                root_files.append(item)
                if item.suffix.lower() in [".jpg", ".jpeg", ".png"] and cover_image_path is None:
                    cover_image_path = item
                    cover_image_name = item.name

    if cover_image_name is None:
        expected_img_stem = manifold_name.replace("LemGendized", "").replace("Large", "")
        if not expected_img_stem:
            expected_img_stem = manifold_name
        cover_image_name = f"{expected_img_stem}.jpg"
        cover_image_path = output_root / cover_image_name

    # 2. Task metadata and descriptions
    task_meta_entry = TASK_META.get(task, {})
    category = (
        clean_overrides.get("category")
        or unified_entry.get("category")
        or task_meta_entry.get("category", "General Machine Learning Dataset")
    )
    arch_base = TASK_ARCH_BASE.get(task, "Deep Convolutional / Transformer Architecture")
    objective = task_meta_entry.get("obj", "Train machine learning models with high precision.")
    desc = task_meta_entry.get("desc", f"High-fidelity dataset manifold for {category}.")

    # Model training profile lookup
    model_profile: dict[str, Any] | None = None
    for _m_key, m_info in MODELS_META.items():
        if isinstance(m_info, dict) and manifold_name in m_info.get("datasets", []):
            model_profile = m_info
            break
    if not model_profile:
        for m_key, m_info in MODELS_META.items():
            if isinstance(m_info, dict) and m_key.lower() in manifold_name.lower().replace("lemgendized", "").replace("large", ""):
                model_profile = m_info
                break

    if model_profile and model_profile.get("arch"):
        arch_base = str(model_profile["arch"])

    # Refine task-specific descriptions for restoration subcategories
    if task == "restoration":
        name_lower = manifold_name.lower()
        if "dehazing" in name_lower or "indoor" in name_lower or "outdoor" in name_lower:
            objective = "Remove atmospheric haze and smoke to recover high-contrast ground truth scene radiance."
            desc = "Standardized paired dataset for single-image dehazing and transmission recovery."
        elif "deraining" in name_lower:
            objective = "Remove dense rain streaks and rain accumulation while preserving background details."
            desc = "Standardized paired dataset for single-image deraining and streak removal."
        elif "deblurring" in name_lower or "debluring" in name_lower:
            objective = "Remove motion and optical blur to restore structural sharpness and fine edges."
            desc = "Standardized paired dataset for single-image deblurring and sharpness recovery."
        elif "denoising" in name_lower:
            objective = "Eliminate real sensor noise and synthetic additive noise while preserving high-frequency textures."
            desc = "Standardized paired dataset for single-image denoising and noise-level estimation."
        elif "exposure" in name_lower or "lowlight" in name_lower:
            objective = "Correct extreme under/over-exposure and enhance visual fidelity in low-light environments."
            desc = "Standardized paired dataset for exposure correction and low-light enhancement."
        elif "film" in name_lower:
            objective = "Restore degraded vintage film frames by removing scratches, color fading, and archival dust."
            desc = "Standardized paired dataset for vintage film restoration and historical frame recovery."
        elif "codeformer" in name_lower:
            objective = "Restore severely degraded facial portraits using codebook lookup and transformer priors."
            desc = "High-fidelity blind face restoration manifold with synthetic degradation and aligned targets."

    # 3. Title and Subtitle
    raw_title = (
        clean_overrides.get("title")
        or unified_entry.get("title")
        or manifold_name.replace("Large", "").replace("LemGendized", "LemGendized ")
    )
    title = str(raw_title).strip().replace("\uFFFD", "-")

    raw_sub = (
        clean_overrides.get("subtitle")
        or unified_entry.get("subtitle")
        or f"High-fidelity manifold for {category} machine learning models"
    )
    subtitle = _format_subtitle(str(raw_sub), category)

    slug = manifold_name.lower().replace("_", "")

    # 4. Provenance sources and sample tally
    sources: dict[str, dict[str, Any]] = {}
    total_samples: int = 0

    if final_index and len(final_index) > 0:
        total_samples = len(final_index)
        for item in final_index:
            raw_src = item.get("source", "")
            if not raw_src or raw_src.lower() in ["old", "unknown", "none", "legacy"]:
                parts = item.get("name", "").split("_")
                raw_src = "_".join(parts[1:-1]) if len(parts) >= 3 else "Unknown"
            s_fmt = format_source(raw_src)
            if s_fmt not in sources:
                sources[s_fmt] = {"train": 0, "val": 0, "total": 0}
            sources[s_fmt]["total"] += 1
            sp = item.get("split", "train")
            if sp in ["train", "val"]:
                sources[s_fmt][sp] += 1
    elif task == "forex":
        forex_data = scan_forex_manifold(output_root)
        total_samples = sum(forex_data.get("samples_per_year", {}).values())
        if total_samples == 0:
            yaml_path = output_root / "dataset_info.yaml"
            if yaml_path.exists():
                try:
                    with open(yaml_path, "r", encoding="utf-8") as yf:
                        yinfo = yaml.safe_load(yf) or {}
                        total_samples = int(yinfo.get("count", 0))
                except (yaml.YAMLError, OSError, ValueError) as exc:
                    logger.debug("Could not read count from dataset_info.yaml: %s", exc)
        sources = {
            "MetaTrader 5 Terminal Native Cache (2019-2026)": {
                "train": int(total_samples * 0.88) if total_samples > 0 else 0,
                "val": int(total_samples * 0.12) if total_samples > 0 else 0,
                "total": total_samples,
            }
        }
    elif task == "ai_helper":
        train_file = output_root / "ai_helper_train.jsonl"
        val_file = output_root / "ai_helper_val.jsonl"
        tr_cnt = 0
        vl_cnt = 0
        if train_file.exists():
            try:
                tr_cnt = sum(1 for _ in train_file.open(encoding="utf-8"))
            except OSError:
                tr_cnt = 0
        if val_file.exists():
            try:
                vl_cnt = sum(1 for _ in val_file.open(encoding="utf-8"))
            except OSError:
                vl_cnt = 0
        total_samples = tr_cnt + vl_cnt
        if total_samples == 0 and (output_root / "dataset_info.yaml").exists():
            try:
                with open(output_root / "dataset_info.yaml", "r", encoding="utf-8") as yf:
                    yinfo = yaml.safe_load(yf) or {}
                    total_samples = int(yinfo.get("count", 0))
            except (yaml.YAMLError, OSError, ValueError) as exc:
                logger.debug("Could not read count from dataset_info.yaml: %s", exc)
        prov = unified_entry.get("provenance_sources", ["LemGendary AI Documentation Hub"])
        for p in prov:
            sources[format_source(p)] = {
                "train": tr_cnt or int(total_samples * 0.85),
                "val": vl_cnt or int(total_samples * 0.15),
                "total": total_samples,
            }
    elif (output_root / "index.json").exists():
        try:
            with open(output_root / "index.json", "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            total_samples = len(idx_data)
            for item in idx_data:
                raw_src = item.get("source", "")
                s_fmt = format_source(raw_src)
                if s_fmt not in sources:
                    sources[s_fmt] = {"train": 0, "val": 0, "total": 0}
                sources[s_fmt]["total"] += 1
                sp = item.get("split", "train")
                if sp in ["train", "val"]:
                    sources[s_fmt][sp] += 1
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Could not parse index.json for sample counts: %s", exc)

    if total_samples == 0 and (output_root / "dataset_info.yaml").exists():
        try:
            with open(output_root / "dataset_info.yaml", "r", encoding="utf-8") as yf:
                yinfo = yaml.safe_load(yf) or {}
                total_samples = int(yinfo.get("count", 0))
        except (yaml.YAMLError, OSError, ValueError) as exc:
            logger.debug("Could not read count from dataset_info.yaml: %s", exc)

    # If sources is empty or only contains the self-referential compiled manifold name
    has_only_self = bool(sources and all("manifold" in k.lower() or k == manifold_name for k in sources.keys()))
    if (not sources or has_only_self) and unified_entry.get("provenance_sources"):
        prov_list = unified_entry["provenance_sources"]
        sources = {}
        per_src = total_samples // len(prov_list) if total_samples and prov_list else 0
        val_c = int(per_src * 0.12)
        train_c = per_src - val_c
        for p in prov_list:
            sources[format_source(p)] = {"train": train_c, "val": val_c, "total": per_src}

    # 5. Metadata fields
    license_name = str(clean_overrides.get("license") or unified_entry.get("license", "CC-BY-NC-4.0"))
    keywords = list(clean_overrides.get("keywords") or unified_entry.get("keywords") or [
        "computer-vision", "deep-learning", "image", "benchmark", "artificial-intelligence"
    ])
    is_private = bool(clean_overrides.get("is_private") or unified_entry.get("is_private", False))
    update_frequency = str(
        clean_overrides.get("expected_update_frequency") or unified_entry.get("expected_update_frequency", "never")
    )
    author_name = str(clean_overrides.get("author") or unified_entry.get("author", "Lem Treursic"))
    author_bio = str(
        clean_overrides.get("author_bio")
        or unified_entry.get("author_bio", "Lead AI Architect & Creator of the LemGendary AI Ecosystem")
    )
    provenance_sources = [
        str(s) for s in (clean_overrides.get("provenance_sources") or unified_entry.get("provenance_sources") or list(sources.keys()))
    ]
    collection_methodology = str(
        clean_overrides.get("collection_methodology")
        or unified_entry.get(
            "collection_methodology",
            "Curated multi-source ingestion synthesized from public benchmark corpora, processed with deterministic degradation pipelines (noise, motion blur, haze, downsampling) paired with ground truth targets, validated via structural SSIM/PSNR gates, and compiled into high-velocity streaming containers (MDS / WebDataset) with pre-indexed splits."
        )
    )
    citations = list(clean_overrides.get("citations") or unified_entry.get("citations") or [
        "https://github.com/lemtreursi/model-training"
    ])
    coverage = dict(clean_overrides.get("coverage") or unified_entry.get("coverage") or {
        "temporal": {"start": "2020-01-01", "end": "2026-10-10"},
        "geospatial": "Global Web-Scale Visual Media",
    })

    return ManifoldContext(
        manifold_path=output_root,
        manifold_name=manifold_name,
        slug=slug,
        task=task,
        category=category,
        arch_base=arch_base,
        objective=objective,
        description=desc,
        title=title,
        subtitle=subtitle,
        license_name=license_name,
        keywords=keywords,
        is_private=is_private,
        update_frequency=update_frequency,
        author_name=author_name,
        author_bio=author_bio,
        cover_image_name=cover_image_name,
        cover_image_path=cover_image_path,
        provenance_sources=provenance_sources,
        collection_methodology=collection_methodology,
        citations=citations,
        coverage=coverage,
        unified_entry=unified_entry,
        model_profile=model_profile,
        total_samples=total_samples,
        sources=sources,
        root_files=root_files,
        root_dirs=root_dirs,
        overrides=clean_overrides,
    )
