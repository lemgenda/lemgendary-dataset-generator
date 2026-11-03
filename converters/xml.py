"""Pascal VOC XML annotation parser. Extracted from compiler_core.py in Phase 1.4."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def _require_text(el: ET.Element) -> str:
    """Return el.text, raising if it is None.

    Element.text is annotated ``str | None`` because XML elements can have
    no text content. In Pascal VOC, the <xmin>/<ymin>/<xmax>/<ymax> elements
    always carry numeric text — the surrounding code only reaches this
    helper after verifying the element exists. This function converts that
    invariant into a runtime check, giving the type checker a concrete
    ``str`` return type without a suppression.
    """
    if el.text is None:
        raise ValueError(f"Missing text content in <{el.tag}>")
    return el.text


def _parse_object(obj: ET.Element) -> dict[str, Any] | None:
    """Convert one VOC <object> element to an annotation dict, or None if incomplete."""
    name_node = obj.find("name")
    cls = name_node.text if name_node is not None and name_node.text else "unknown"

    bndbox = obj.find("bndbox")
    if bndbox is None:
        return None

    nodes = [bndbox.find(tag) for tag in ("xmin", "ymin", "xmax", "ymax")]
    present = [node for node in nodes if node is not None]
    if len(present) != len(nodes):
        return None

    xmin, ymin, xmax, ymax = (float(_require_text(node)) for node in present)
    return {"class": cls, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin]}


def parse_xml(xml_path: str | Path) -> list[dict[str, Any]]:
    """Parse a Pascal VOC XML annotation file.

    Returns a list of dicts, each:
        {"class": <str>, "bbox": [xmin, ymin, width, height]}
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    annotations: list[dict[str, Any]] = []

    for obj in root.findall("object"):
        item = _parse_object(obj)
        if item is not None:
            annotations.append(item)

    return annotations
