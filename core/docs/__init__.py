"""LemGendary Dataset Documentation Generation Subpackage.

Provides modular metadata resolution, manifold scanning, manifest serialization,
and Markdown documentation builders.
"""

from __future__ import annotations

from core.docs.context import ManifoldContext, find_unified_entry, resolve_manifold_context
from core.docs.generators import (
    generate_category_txt,
    generate_classes_txt,
    generate_cover_image,
    generate_dataset_info_yaml,
    generate_index_json,
    generate_kaggle_metadata,
    generate_readme_md,
)
from core.docs.manifests import (
    extract_kaggle_description,
    write_category_txt,
    write_classes_txt,
    write_dataset_info_yaml,
    write_index_json,
    write_kaggle_metadata,
)
from core.docs.metadata import (
    FOREX_COLUMN_FIELDS,
    MANIFEST_CACHE,
    MANIFOLD_TASK_MAP,
    MODELS_META,
    TASK_ARCH_BASE,
    TASK_META,
    UNIFIED_DATA,
    format_source,
)
from core.docs.orchestrator import (
    generate_dataset_docs,
    main,
    regenerate_all_docs,
    regenerate_all_non_forex,
)
from core.docs.scanner import scan_forex_manifold
from core.docs.templates import (
    build_models_markdown,
    build_structure_text,
    clean_readme,
    render_forex_readme,
    render_standard_readme,
)

__all__ = [
    "FOREX_COLUMN_FIELDS",
    "MANIFEST_CACHE",
    "MANIFOLD_TASK_MAP",
    "MODELS_META",
    "ManifoldContext",
    "TASK_ARCH_BASE",
    "TASK_META",
    "UNIFIED_DATA",
    "build_models_markdown",
    "build_structure_text",
    "clean_readme",
    "extract_kaggle_description",
    "find_unified_entry",
    "format_source",
    "generate_category_txt",
    "generate_classes_txt",
    "generate_cover_image",
    "generate_dataset_docs",
    "generate_dataset_info_yaml",
    "generate_index_json",
    "generate_kaggle_metadata",
    "generate_readme_md",
    "main",
    "regenerate_all_docs",
    "regenerate_all_non_forex",
    "render_forex_readme",
    "render_standard_readme",
    "resolve_manifold_context",
    "scan_forex_manifold",
    "write_category_txt",
    "write_classes_txt",
    "write_dataset_info_yaml",
    "write_index_json",
    "write_kaggle_metadata",
]
