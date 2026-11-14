"""LemGendary Dataset Documentation Context Resolver.

Extracts, standardizes, and unifies metadata, file inventories, task types,
and model training profiles for any vision or temporal dataset manifold.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from core.docs.context_resolvers import resolve_task, resolve_task_profile
from core.docs.context_types import (
    AuthorInfo,
    Inventory,
    Lineage,
    ListingMeta,
    ManifoldContext,
    ManifoldIdentity,
    SpecInputs,
    TaskProfile,
)
from core.docs.lineage_tally import TallyResult, tally_sources
from core.docs.metadata import UNIFIED_DATA

__all__ = [
    "AuthorInfo",
    "Inventory",
    "Lineage",
    "ListingMeta",
    "ManifoldContext",
    "ManifoldIdentity",
    "SpecInputs",
    "TaskProfile",
    "find_unified_entry",
    "resolve_manifold_context",
]

_COVER_SUFFIXES = [".webp", ".jpg", ".jpeg", ".png"]

_DEFAULT_KEYWORDS = [
    "computer-vision", "deep-learning", "image", "benchmark", "artificial-intelligence"
]
_DEFAULT_AUTHOR_BIO = "Lead AI Architect & Creator of the LemGendary AI Ecosystem"
_DEFAULT_METHODOLOGY = (
    "Curated multi-source ingestion synthesized from public benchmark corpora, processed"
    " with deterministic degradation pipelines (noise, motion blur, haze, downsampling)"
    " paired with ground truth targets, validated via structural SSIM/PSNR gates, and"
    " compiled into high-velocity streaming containers (MDS / WebDataset) with pre-indexed"
    " splits."
)
_DEFAULT_COVERAGE = {
    "temporal": {"start": "2020-01-01", "end": "2026-10-10"},
    "geospatial": "Global Web-Scale Visual Media",
}


def _dataset_entries() -> list[tuple[str, dict[str, Any]]]:
    """Return the (key, info) pairs of unified_data.yaml whose info is a mapping."""
    datasets = UNIFIED_DATA.get("datasets", {})
    if not isinstance(datasets, dict):
        return []
    return [(key, info) for key, info in datasets.items() if isinstance(info, dict)]


def _matches_normalized(d_key: str, d_info: dict[str, Any], norm_target: str) -> bool:
    """True when the normalized key/name of an entry matches ``norm_target``."""
    norm_key = d_key.lower().replace("_", "")
    d_name = str(d_info.get("name", d_key)).lower().replace("_", "")
    return norm_target in (norm_key, d_name) or norm_key in norm_target or norm_target in norm_key


def find_unified_entry(manifold_name: str) -> dict[str, Any]:
    """Find matching dataset specification from unified_data.yaml by exact or normalized name."""
    entries = _dataset_entries()
    name_lower = manifold_name.lower()
    norm_target = name_lower.replace("_", "").replace("large", "")

    # 1. Exact modernized_folder match (SSOT)
    for _d_key, d_info in entries:
        if str(d_info.get("modernized_folder", "")).lower() == name_lower:
            return d_info

    # 2. Exact name match
    for d_key, d_info in entries:
        d_name = str(d_info.get("name", d_key)).lower()
        if name_lower in (f"lemgendized{d_name}", f"lemgendized{d_name}large"):
            return d_info

    # 3. Normalized key or name match
    for d_key, d_info in entries:
        if _matches_normalized(d_key, d_info, norm_target):
            return d_info

    return {}


def _format_subtitle(raw_subtitle: str, fallback_cat: str) -> str:
    """Return ``raw_subtitle`` cut or replaced to fit the Kaggle 20-80 character limit."""
    sub = raw_subtitle.strip().replace("�", "-").replace("—", "-").replace("–", "-")
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


def _scan_inventory(output_root: Path, manifold_name: str) -> Inventory:
    """List root-level files/folders and locate (or name) the cover image."""
    inventory = Inventory(cover_image_name=None, cover_image_path=None)
    if output_root.exists() and output_root.is_dir():
        for item in sorted(output_root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                inventory.root_dirs.append(item)
                continue
            inventory.root_files.append(item)
            if item.suffix.lower() in _COVER_SUFFIXES and inventory.cover_image_path is None:
                inventory.cover_image_path = item
                inventory.cover_image_name = item.name

    if inventory.cover_image_name is None:
        expected_img_stem = manifold_name.replace("LemGendized", "").replace("Large", "")
        inventory.cover_image_name = f"{expected_img_stem or manifold_name}.webp"
        inventory.cover_image_path = output_root / inventory.cover_image_name
    return inventory


def _resolve_listing(manifold_name: str, category: str, spec: SpecInputs) -> ListingMeta:
    """Resolve the Kaggle listing fields from overrides, unified_data.yaml and defaults."""
    raw_title = spec.pick_truthy(
        "title", manifold_name.replace("Large", "").replace("LemGendized", "LemGendized ")
    )
    raw_sub = spec.pick_truthy(
        "subtitle", f"High-fidelity manifold for {category} machine learning models"
    )
    return ListingMeta(
        title=str(raw_title).strip().replace("�", "-"),
        subtitle=_format_subtitle(str(raw_sub), category),
        license_name=str(spec.pick("license", "CC-BY-NC-4.0")),
        keywords=list(spec.pick_truthy("keywords", _DEFAULT_KEYWORDS)),
        is_private=bool(spec.pick("is_private", False)),
        update_frequency=str(spec.pick("expected_update_frequency", "never")),
        author=AuthorInfo(
            name=str(spec.pick("author", "Lem Treursic")),
            bio=str(spec.pick("author_bio", _DEFAULT_AUTHOR_BIO)),
        ),
    )


def _resolve_citations(spec: SpecInputs) -> list[str]:
    """The registry ``citation_url`` (the training suite), then the entry's own references."""
    registry_meta = UNIFIED_DATA.get("_registry_metadata") or {}
    citation_url = registry_meta.get("citation_url") if isinstance(registry_meta, dict) else None
    listed = [str(c) for c in spec.pick_truthy("citations", [])]
    head = [str(citation_url)] if citation_url else []
    return list(dict.fromkeys(head + listed))


def _resolve_lineage(tally: TallyResult, spec: SpecInputs) -> Lineage:
    """Resolve provenance, methodology, citations and coverage around the sample tally."""
    return Lineage(
        total_samples=tally.total_samples,
        sources=tally.sources,
        splits=tally.splits,
        provenance_sources=[
            str(s) for s in spec.pick_truthy("provenance_sources", list(tally.sources.keys()))
        ],
        collection_methodology=str(spec.pick("collection_methodology", _DEFAULT_METHODOLOGY)),
        citations=_resolve_citations(spec),
        coverage=dict(spec.pick_truthy("coverage", copy.deepcopy(_DEFAULT_COVERAGE))),
    )


def resolve_manifold_context(
    output_root: str | Path,
    pascal_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    final_index: list[dict[str, Any]] | None = None,
) -> ManifoldContext:
    """Resolve a unified ManifoldContext with full metadata and physical asset inventory."""
    output_root = Path(output_root).resolve()
    manifold_name = pascal_name or output_root.name
    spec = SpecInputs(
        unified_entry=find_unified_entry(manifold_name),
        overrides=dict(overrides) if overrides else {},
    )
    task = resolve_task(manifold_name, output_root, spec.unified_entry, spec.overrides, final_index)
    inventory = _scan_inventory(output_root, manifold_name)
    profile = resolve_task_profile(manifold_name, task, spec)
    tally = tally_sources(output_root, task, manifold_name, spec.unified_entry, final_index)
    return ManifoldContext(
        identity=ManifoldIdentity(
            path=output_root,
            name=manifold_name,
            slug=manifold_name.lower().replace("_", ""),
            task=task,
        ),
        profile=profile,
        listing=_resolve_listing(manifold_name, profile.category, spec),
        lineage=_resolve_lineage(tally, spec),
        inventory=inventory,
        spec=spec,
    )
