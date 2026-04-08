# ============================================
# YOLO Dataset Compiler (FULL SYSTEM: STEP 0 + 1 + 2)
# ============================================

import os
import json
import shutil
import random
import hashlib
from pathlib import Path
from PIL import Image, ImageStat
import numpy as np

# Optional deps:
# pip install pandas scipy pycocotools

# ---------------- CONFIG ----------------
DEFAULT_CONFIG = {
    "train_split": 0.9,
    "img_min": 256,
    "img_max": 4096,
    "max_per_dataset": 50000,
    "enable_dedup": True,
    "black_threshold": 5,
    "tasks_with_targets": ["super_resolution", "denoise", "deblur"]
}

CONFIG_PATH = Path("./config.json")
CONFIG = {**DEFAULT_CONFIG, **json.load(open(CONFIG_PATH))} if CONFIG_PATH.exists() else DEFAULT_CONFIG

INPUT_ROOT = Path("./raw-sets")
OUTPUT_ROOT = Path("./compiled-yolo")

# ---------------- STRUCTURE ----------------
def create_structure(base):
    for section in ["images", "labels", "targets"]:
        for split in ["train", "val"]:
            (base / section / split).mkdir(parents=True, exist_ok=True)

# ---------------- VALIDATION ----------------
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

# ---------------- ADVANCED VALIDATION ----------------
def is_black_image(img):
    stat = ImageStat.Stat(img)
    return sum(stat.mean) / len(stat.mean) < CONFIG["black_threshold"]


def compute_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()

# ---------------- SRGB ----------------
def ensure_srgb(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# ---------------- NAMING ----------------
def generate_name(prefix, idx):
    return f"{prefix}_{idx:09d}"

# ---------------- YOLO ----------------
def convert_bbox_xywh_to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    return [
        (x + bw/2) / w,
        (y + bh/2) / h,
        bw / w,
        bh / h
    ]

# ---------------- FORMAT CONVERTERS ----------------

def parse_coco(json_path):
    data = json.load(open(json_path))
    images = {img["id"]: img for img in data.get("images", [])}
    anns = {}
    for a in data.get("annotations", []):
        anns.setdefault(a["image_id"], []).append(a)
    return images, anns


def parse_voc(xml_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        b = obj.find("bndbox")
        bbox = [
            int(b.find("xmin").text),
            int(b.find("ymin").text),
            int(b.find("xmax").text) - int(b.find("xmin").text),
            int(b.find("ymax").text) - int(b.find("ymin").text)
        ]
        boxes.append((cls, bbox))
    return boxes


def parse_parquet(path):
    import pandas as pd
    return pd.read_parquet(path)


def parse_matlab(path):
    from scipy.io import loadmat
    return loadmat(path)

# ---------------- AUTO FORMAT DETECTION ----------------

def detect_annotations(dataset_path):
    for f in dataset_path.rglob("*"):
        if f.suffix == ".json":
            return "coco", f
        if f.suffix == ".xml":
            return "voc", None
        if f.suffix == ".parquet":
            return "parquet", f
        if f.suffix == ".mat":
            return "matlab", f
    return None, None

# ---------------- TARGETS ----------------

def create_target(img, task):
    if task in CONFIG["tasks_with_targets"]:
        return img
    return None

# ---------------- MAIN PROCESS ----------------

def process_dataset():

    create_structure(OUTPUT_ROOT)

    index = []
    counters = {}
    seen_hashes = set()

    for model_dir in INPUT_ROOT.iterdir():
        if not model_dir.is_dir(): continue

        for dataset in model_dir.iterdir():
            if not dataset.is_dir(): continue

            prefix = dataset.name.replace("-","_")
            counters.setdefault(prefix, 0)

            fmt, ann_path = detect_annotations(dataset)

            coco_images, coco_anns = None, None
            if fmt == "coco":
                coco_images, coco_anns = parse_coco(ann_path)

            images = list(dataset.rglob("*.jpg")) + list(dataset.rglob("*.png"))

            for img_path in images:

                if counters[prefix] >= CONFIG["max_per_dataset"]:
                    break

                if not is_valid_image(img_path):
                    continue

                img = Image.open(img_path)
                img = ensure_srgb(img)

                if is_black_image(img):
                    continue

                # Deduplication
                if CONFIG["enable_dedup"]:
                    h = compute_hash(img)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                else:
                    h = None

                w, hgt = img.size
                if w < CONFIG["img_min"] or hgt < CONFIG["img_min"]:
                    continue

                idx = counters[prefix]
                name = generate_name(prefix, idx)
                counters[prefix] += 1

                split = "train" if random.random() < CONFIG["train_split"] else "val"

                # SAVE IMAGE
                out_img = OUTPUT_ROOT / "images" / split / f"{name}.jpg"
                img.save(out_img, "JPEG", quality=95)

                # -------- LABELS --------
                label_path = OUTPUT_ROOT / "labels" / split / f"{name}.txt"
                annotations = []

                if fmt == "coco":
                    img_name = img_path.name
                    for img_id, meta in coco_images.items():
                        if meta["file_name"] == img_name:
                            anns = coco_anns.get(img_id, [])
                            for a in anns:
                                annotations.append((a["category_id"], a["bbox"]))

                elif fmt == "voc":
                    xml = img_path.with_suffix(".xml")
                    if xml.exists():
                        annotations = parse_voc(xml)

                with open(label_path, "w") as f:
                    for cls, bbox in annotations:
                        yolo = convert_bbox_xywh_to_yolo(bbox, w, hgt)
                        f.write(f"{cls} {' '.join(map(str,yolo))}\n")

                # -------- TARGETS --------
                target_img = create_target(img, model_dir.name)
                if target_img is not None:
                    tgt_path = OUTPUT_ROOT / "targets" / split / f"{name}.jpg"
                    target_img.save(tgt_path, "JPEG", quality=95)

                index.append({
                    "name": name,
                    "source": prefix,
                    "original_path": str(img_path),
                    "split": split,
                    "format": fmt,
                    "hash": h
                })

    with open(OUTPUT_ROOT / "index.json", "w") as f:
        json.dump(index, f, indent=2)

# ---------------- README ----------------

def generate_readme():

    data = json.load(open(OUTPUT_ROOT / "index.json"))

    train = [x for x in data if x["split"] == "train"]
    val = [x for x in data if x["split"] == "val"]

    sources = {}
    for item in data:
        sources.setdefault(item["source"], 0)
        sources[item["source"]] += 1

    readme = f"""
# YOLO Dataset

## Summary
- Total images: {len(data)}
- Train: {len(train)}
- Validation: {len(val)}

## Sources
"""

    for k,v in sources.items():
        readme += f"- {k}: {v}\n"

    readme += f"""

## Quality Filters
- Deduplication: {CONFIG['enable_dedup']}
- Black threshold: {CONFIG['black_threshold']}
- Min size: {CONFIG['img_min']}

## Models
- YOLOv8 / YOLOv9
- RT-DETR

## Loss
- CIoU
- Focal
- Charbonnier

## Metrics
- mAP50 avg: 0.45
- excellent: 0.65
- sota: 0.75+
"""

    with open(OUTPUT_ROOT / "README.md", "w") as f:
        f.write(readme)


if __name__ == "__main__":
    process_dataset()
    generate_readme()
