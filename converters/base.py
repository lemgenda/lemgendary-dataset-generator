"""
LemGendary Dataset Compiler — Converter base types and contracts.

Phase 1.4 does NOT change any runtime logic. This module exists to document
the shapes that every converter returns, so Phase 3 (which restructures
process_image around a tagged-union `Annotation`) has a written contract
to conform to.

Contract summary
----------------

1. Parsers (``parse_coco``, ``parse_parquet``, ``parse_xml``, ``parse_yolo``,
   ``parse_matlab``, ``parse_safetensors``) — each takes a format-specific
   path argument and returns a data structure with a shape unique to that
   format. Their return types are documented in each submodule's docstring.
   Callers in ``process_image`` (compiler_core.py) consume those shapes
   directly; Phase 3 will formalize them via a tagged union.

2. Dispatch (``detect_annotations``) — takes a dataset directory and returns
   ``(format_name, path_or_subdir)`` or ``(None, None)``. Format names are:
   ``"coco"``, ``"parquet"``, ``"xml"``, ``"yolo"``, ``"npz"``, ``"matlab"``.

3. Coordinate conventions — all bbox-returning parsers yield the *native*
   representation of their source format. Normalization to YOLO happens in
   ``process_image`` via ``convert_bbox_xywh_to_yolo`` and
   ``normalize_points``, NOT in the parsers. This is deliberate: parsers must
   be pure and format-faithful.

Phase 1.4 preserves all of the above exactly as-is. Phase 3 will formalize.
"""

from __future__ import annotations

from typing import Any

# Placeholder for the Phase 3 tagged union. Kept as an alias so future
# imports compile:
#     from converters.base import Annotation, Converter
# without requiring a rewrite in Phase 3.
Annotation = dict[str, Any]
Converter = Any  # Phase 3 Protocol placeholder

__all__ = ["Annotation", "Converter"]