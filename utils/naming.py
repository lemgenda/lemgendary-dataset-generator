"""Slug and category naming helpers. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations


def clean_slug(slug: str) -> str:
    """Normalize a raw source slug to the canonical registry key."""
    sl = slug.lower()
    # 2026: Only collapse known massive multi-part manifolds
    if "laion" in sl:
        return "laion"
    if "ava" in sl:
        return "ava"
    if "aadb" in sl:
        return "aadb"
    if "ffhq" in sl:
        return "ffhq"
    # Preserve specialized source names
    return slug.replace(".tar.gz", "").replace(".tgz", "").replace(".zip", "")


def map_category(cat_name_or_id, source_name, category_map: dict | None = None) -> int:
    """Map a category name/ID to the local class index (default 0 = Person).

    ``category_map`` must be passed explicitly — the pre-1.5.5 version reached
    into compiler_core's module-global CATEGORY_MAP.
    """
    if isinstance(cat_name_or_id, str):
        name = cat_name_or_id.lower().strip()
        cmap = category_map or {}
        return cmap.get(name, 0)
    return int(cat_name_or_id)