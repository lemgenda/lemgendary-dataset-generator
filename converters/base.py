"""
LemGendary Dataset Compiler — Converter base types and contracts.

Phase 1.4 does NOT change any runtime logic. This module documents the
shapes that every converter returns so Phase 3 (which restructures
process_image around a tagged-union `Annotation`) has a written contract.

Contract summary
----------------

Parser return shapes (Phase 1.5.7 — annotated):

    parse_coco         -> tuple[dict[int, dict[str, Any]],
                                dict[int, list[dict[str, Any]]]]
    parse_parquet      -> tuple[Path, dict[str, Any], list[str]]
    parse_xml          -> list[dict[str, Any]]
    parse_yolo         -> list[dict[str, Any]]
    parse_matlab       -> tuple[dict[str, Any], str]
    parse_safetensors  -> dict[str, Any]

    detect_annotations -> tuple[str | None, Path | None]

Coordinate conventions
----------------------

All bbox-returning parsers yield the *native* representation of their source
format. Normalization to YOLO happens in `process_image` via
`convert_bbox_xywh_to_yolo` and `normalize_points`, NOT in the parsers. This
is deliberate: parsers must be pure and format-faithful.
"""

from __future__ import annotations

from typing import Any

Annotation = dict[str, Any]
Converter = Any  # Phase 3 Protocol placeholder

__all__ = ["Annotation", "Converter"]