"""Pascal VOC XML annotation parser. Copied verbatim from compiler_core.py in Phase 1.4."""

from __future__ import annotations


def parse_xml(xml_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        cls = name_node.text if name_node is not None else "unknown"
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xn, yn = bndbox.find("xmin"), bndbox.find("ymin")
            xmn, ymn = bndbox.find("xmax"), bndbox.find("ymax")
            if xn is not None and yn is not None and xmn is not None and ymn is not None:
                xmin, ymin, xmax, ymax = float(xn.text), float(yn.text), float(xmn.text), float(ymn.text) # type: ignore
            else: continue
            width = xmax - xmin
            height = ymax - ymin
            annotations.append({"class": cls, "bbox": [xmin, ymin, width, height]})

    return annotations