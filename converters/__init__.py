"""
LemGendary Dataset Compiler — Annotation Converters Package

Consolidates the seven format parsers previously inline in compiler_core.py:

    parse_coco        -> converters/coco.py
    parse_parquet     -> converters/parquet.py
    parse_xml         -> converters/xml.py
    parse_yolo        -> converters/yolo.py
    parse_matlab      -> converters/matlab.py
    parse_safetensors -> converters/safetensors.py
    detect_annotations -> converters/dispatch.py

Phase 1.4 of the 2026 modernization roadmap.

Backward compatibility:
    compiler_core.py re-exports these symbols so any `from compiler_core import
    parse_coco` (used by manifold_compile.py) keeps working unchanged.
"""

from __future__ import annotations

__all__ = [
    "detect_annotations",
    "parse_coco",
    "parse_parquet",
    "parse_xml",
    "parse_yolo",
    "parse_matlab",
    "parse_safetensors",
]

from .dispatch import detect_annotations
from .coco import parse_coco
from .parquet import parse_parquet
from .xml import parse_xml
from .yolo import parse_yolo
from .matlab import parse_matlab
from .safetensors import parse_safetensors