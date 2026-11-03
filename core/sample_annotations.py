"""Turn a sample's source annotations into YOLO-style records and write its label file.

One parser per source annotation format is registered in ``_PARSERS``; adding
a format means adding a parser, not editing the dispatch.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

from converters import parse_xml, parse_yolo
from core.compiler_config import CATEGORY_MAP
from utils.geometry import convert_bbox_xywh_to_yolo, normalize_points
from utils.naming import map_category

logger = logging.getLogger(__name__)


class Annotation(TypedDict):
    """Shape of every annotation record a parser returns."""
    type: str        # 'bbox' | 'segmentation' | 'pose'
    cls: int
    data: list[Any]


@dataclass(frozen=True)
class AnnotationSource:
    """The raw annotation payload of one sample plus what parsing it needs."""

    fmt: str
    ann_data: Any
    prefix: str
    task: str
    width: int
    height: int

    def category(self, raw: Any) -> int:
        """Map a source class id or name to the manifold's class index."""
        return map_category(raw, self.prefix, CATEGORY_MAP)


def _parse_coco(src: AnnotationSource) -> list[Annotation]:
    """COCO records: keypoints become pose, polygons segmentation, the rest bbox."""
    out: list[Annotation] = []
    for a_ in src.ann_data:
        cls = src.category(str(a_["category_id"]))
        if a_.get("keypoints"):
            kpts = normalize_points(a_["keypoints"], src.width, src.height, stride=3)
            out.append({"type": "pose", "cls": cls, "data": a_["bbox"] + kpts})
        elif a_.get("segmentation"):
            seg = a_["segmentation"]
            poly_raw = seg[0] if isinstance(seg, list) and len(seg) > 0 else []
            if poly_raw:
                poly = normalize_points(poly_raw, src.width, src.height, stride=2)
                out.append({"type": "segmentation", "cls": cls, "data": poly})
        else:
            out.append({"type": "bbox", "cls": cls, "data": a_["bbox"]})
    return out


def _parse_parquet_row(src: AnnotationSource, row: Any, mapping: dict[str, Any]) -> Annotation:
    """Convert one annotation row of a Parquet source using its column mapping."""
    cls = src.category(row[mapping.get("class", "class")])
    seg_col, kpt_col = mapping.get("segmentation"), mapping.get("keypoints")
    if row.get(seg_col):
        poly = normalize_points(row[seg_col], src.width, src.height, stride=2)
        return {"type": "segmentation", "cls": cls, "data": poly}
    if row.get(kpt_col):
        kpts = normalize_points(row[kpt_col], src.width, src.height, stride=3)
        return {"type": "pose", "cls": cls, "data": [0, 0, 0, 0] + kpts}
    bbox = [
        row[mapping.get("xmin", "xmin")],
        row[mapping.get("ymin", "ymin")],
        row[mapping.get("width", "width")],
        row[mapping.get("height", "height")],
    ]
    return {"type": "bbox", "cls": cls, "data": bbox}


def _parse_parquet(src: AnnotationSource) -> list[Annotation]:
    """Parquet sources carry a (DataFrame subset, column mapping) pair."""
    if isinstance(src.ann_data, dict):
        return []
    df_subset, mapping = cast("tuple[Any, dict[str, Any]]", src.ann_data)
    return [_parse_parquet_row(src, row, mapping) for _, row in df_subset.iterrows()]


def _parse_matlab(src: AnnotationSource) -> list[Annotation]:
    """MATLAB sources are pre-parsed into ``{"class", "bbox"}`` entries."""
    out: list[Annotation] = []
    for entry in src.ann_data:
        try:
            out.append({"type": "bbox", "cls": src.category(entry["class"]),
                        "data": entry["bbox"]})
        except (KeyError, TypeError) as exc:
            logger.debug("Matlab annotation parsing error for entry: %s", exc)
    return out


def _parse_xml(src: AnnotationSource) -> list[Annotation]:
    """Pascal VOC XML files: one bbox per object."""
    if not isinstance(src.ann_data, (str, Path)):
        return []
    return [
        {"type": "bbox", "cls": src.category(a_["class"]),
         "data": cast("list[Any]", a_["bbox"])}
        for a_ in parse_xml(src.ann_data)
    ]


def _parse_yolo(src: AnnotationSource) -> list[Annotation]:
    """YOLO text files: bboxes, with keypoints for pose datasets."""
    if not isinstance(src.ann_data, (str, Path)):
        return []
    out: list[Annotation] = []
    for a_ in parse_yolo(src.ann_data, src.width, src.height):
        cls = src.category("0" if src.task == "pose" else a_["class"])
        a_bbox = cast("list[Any]", a_["bbox"])
        if a_.get("keypoints"):
            a_kpts = cast("list[Any]", a_["keypoints"])
            out.append({"type": "pose", "cls": cls, "data": a_bbox + a_kpts})
        else:
            out.append({"type": "bbox", "cls": cls, "data": a_bbox})
    return out


def _face_pose_from_landmarks(src: AnnotationSource, landmarks: Any) -> Annotation:
    """Build a padded face bbox plus five keypoints from 110-point face landmarks."""
    pts = landmarks[:68]
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    pad_w = (x_max - x_min) * 0.05
    pad_h = (y_max - y_min) * 0.05
    x_min = max(0, x_min - pad_w)
    y_min = max(0, y_min - pad_h)
    x_max = min(src.width, x_max + pad_w)
    y_max = min(src.height, y_max + pad_h)

    l_eye = landmarks[36:42].mean(axis=0)
    r_eye = landmarks[42:48].mean(axis=0)
    nose, l_mouth, r_mouth = landmarks[30], landmarks[48], landmarks[54]
    kpts = [
        l_eye[0], l_eye[1], r_eye[0], r_eye[1],
        nose[0], nose[1], l_mouth[0], l_mouth[1],
        r_mouth[0], r_mouth[1],
    ]
    return {
        "type": "pose", "cls": src.category("0"),
        "data": [x_min, y_min, x_max - x_min, y_max - y_min] + kpts,
    }


def _parse_npz(src: AnnotationSource) -> list[Annotation]:
    """NumPy archives with a 110x2 ``landmarks`` array (SFHQ faces)."""
    if not isinstance(src.ann_data, (str, Path)):
        return []
    try:
        data = np.load(str(src.ann_data))
        if "landmarks" in data.files and data["landmarks"].shape == (110, 2):
            return [_face_pose_from_landmarks(src, data["landmarks"])]
    except (OSError, KeyError, ValueError) as e:
        print(f"Error parsing NPZ {src.ann_data}: {e}")
    return []


def _safetensors_tags(meta: dict[str, Any]) -> list[str]:
    """Collect training tags from kohya ``ss_tag_frequency`` or ``ss_datasets`` metadata."""
    tags: list[str] = []
    if "ss_tag_frequency" in meta:
        try:
            for bucket in json.loads(str(meta["ss_tag_frequency"])).values():
                tags.extend(bucket.keys())
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.debug("Failed parsing ss_tag_frequency: %s", exc)
    if not tags and "ss_datasets" in meta:
        try:
            for ds in json.loads(str(meta["ss_datasets"])):
                if "tag_frequency" in ds:
                    tags.extend(ds["tag_frequency"].keys())
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.debug("Failed parsing ss_datasets: %s", exc)
    return tags


def _parse_safetensors(src: AnnotationSource) -> list[Annotation]:
    """safetensors metadata: up to 20 unique tags, each as a full-frame bbox."""
    if not isinstance(src.ann_data, dict):
        return []
    unique_tags = list(set(_safetensors_tags(src.ann_data)))[:20]
    return [
        {"type": "bbox", "cls": src.category(tag), "data": [0.0, 0.0, 1.0, 1.0]}
        for tag in unique_tags
    ]


_PARSERS: dict[str, Callable[[AnnotationSource], list[Annotation]]] = {
    "coco": _parse_coco,
    "parquet": _parse_parquet,
    "matlab": _parse_matlab,
    "xml": _parse_xml,
    "yolo": _parse_yolo,
    "npz": _parse_npz,
    "safetensors": _parse_safetensors,
}


def parse_annotations(src: AnnotationSource) -> list[Annotation]:
    """Dispatch to the parser registered for the source format."""
    parser = _PARSERS.get(src.fmt)
    if parser is None:
        return []
    if src.fmt == "coco":
        has_data = src.ann_data is not None
    elif src.fmt == "safetensors":
        has_data = True
    else:
        has_data = bool(src.ann_data)
    return parser(src) if has_data else []


# ─── Label file ─────────────────────────────────────────────────────────────
def _classification_label(ann_data: Any, img_path: Path) -> Any:
    """Resolve the class label from the annotation, else from fake/NSFW path keywords."""
    class_label: Any = 1
    if isinstance(ann_data, dict) and "label" in ann_data:
        class_label = ann_data["label"]
    elif isinstance(ann_data, tuple) and len(ann_data) == 2 and isinstance(ann_data[1], dict):
        df_subset, mapping = cast("tuple[Any, dict[str, Any]]", ann_data)
        lbl_col = mapping.get("label", "label")
        if lbl_col in df_subset.columns:
            class_label = df_subset.iloc[0][lbl_col]
    else:
        p_lower = str(img_path).lower()
        if any(k in p_lower for k in ["fake", "ai", "synthetic", "nsfw", "porn", "explicit"]):
            class_label = 0
    if isinstance(class_label, float) and not pd.isna(class_label):
        class_label = int(class_label)
    return class_label


def _annotation_line(ann: Annotation, width: int, height: int) -> str | None:
    """Render one annotation as a YOLO label line."""
    cls, data = ann["cls"], ann["data"]
    if ann["type"] == "bbox":
        return f"{cls} {' '.join(map(str, convert_bbox_xywh_to_yolo(data, width, height)))}"
    if ann["type"] == "segmentation":
        return f"{cls} {' '.join(map(str, data))}"
    if ann["type"] == "pose":
        yolo_box = convert_bbox_xywh_to_yolo(data[:4], width, height)
        return f"{cls} {' '.join(map(str, yolo_box))} {' '.join(map(str, data[4:]))}"
    return None


@dataclass(frozen=True)
class LabelContent:
    """Everything that decides what a sample's label file contains."""

    task: str
    nima_probs: list[float]
    annotations: list[Annotation]
    ann_data: Any
    img_path: Path


def label_lines(content: LabelContent, width: int, height: int) -> list[str]:
    """Return the label file lines: NIMA probabilities, a class id, or YOLO annotations."""
    if content.task == "quality":
        return [" ".join(f"{p:.6f}" for p in content.nima_probs)]
    if content.task == "classification":
        return [str(_classification_label(content.ann_data, content.img_path))]
    lines = (_annotation_line(ann, width, height) for ann in content.annotations)
    return [line for line in lines if line is not None]


def write_label_file(path: Path, lines: list[str]) -> None:
    """Write label lines, newline-terminated, to ``path`` (an empty list makes an empty file).

    The split folder is created on demand: only label-task manifolds get
    ``labels/<split>`` up front, but restoration, segmentation and diffusion
    manifolds write labels too when auto-labeling finds annotations.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)
