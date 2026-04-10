# ============================================
# YOLO Dataset Compiler (FULL SYSTEM COMPLETE)
# Steps 0 + 1 + 2 + 3 — FINAL VERSION
# ============================================

import os
import json
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
    "tasks_with_targets": ["super_resolution", "denoise", "deblur"],
    "nima_threshold": 4.5,
    "auto_label": True,
    "nima_model_path": "c:/Development/python/model-training/lemgendary-training-suite/trained-models/nima_technical/nima_technical_best.pth",
    "yolo_model_path": "yolov8n.pt"
}

CONFIG_PATH = Path("./config.json")
CONFIG = {**DEFAULT_CONFIG, **json.load(open(CONFIG_PATH))} if CONFIG_PATH.exists() else DEFAULT_CONFIG

INPUT_ROOT = Path("./raw-sets")
OUTPUT_ROOT = Path("./compiled-datasets")

# ---------------- STRUCTURE ----------------
def create_structure(base):
    for section in ["images", "labels", "targets"]:
        for split in ["train", "val"]:
            (base / section / split).mkdir(parents=True, exist_ok=True)
    
    # Simple File Lock for Thread-Safety
    lock_file = base / ".lock"
    if lock_file.exists():
        import time
        # If lock exists, wait up to 10 seconds or error out
        print(f"⚠️  [RESILIENCE] Lock file detected at {lock_file}. Waiting...")
        for _ in range(10):
            if not lock_file.exists(): break
            time.sleep(1)
        if lock_file.exists():
            raise RuntimeError(f"Could not acquire lock for {base}. Another process might be running.")
    
    lock_file.touch()
    return lock_file

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
    return [round((x + bw/2)/w, 6), round((y + bh/2)/h, 6), round(bw/w, 6), round(bh/h, 6)]

def normalize_points(points, w, h, stride=2):
    """Normalizes a flat list of points [x1, y1, x2, y2...] or [x1, y1, v1...] to [0, 1]"""
    norm = []
    for i in range(0, len(points), stride):
        norm.append(round(points[i] / w, 6))
        norm.append(round(points[i+1] / h, 6))
        if stride == 3: # Keep visibility as-is
            norm.append(points[i+2])
    return norm

# ---------------- FORMAT PARSERS ----------------

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
    df = pd.read_parquet(path)
    # Greedy discovery for bounding box, segmentation, and pose columns
    cols = df.columns.tolist()
    mapping = {}
    for c in cols:
        cl = c.lower()
        if "id" in cl or "class" in cl or "category" in cl: mapping["class"] = c
        if "xmin" in cl or "x1" in cl or "left" in cl: mapping["xmin"] = c
        if "ymin" in cl or "y1" in cl or "top" in cl: mapping["ymin"] = c
        if "width" in cl or "w" == cl: mapping["width"] = c
        if "height" in cl or "h" == cl: mapping["height"] = c
        if "segment" in cl or "poly" in cl: mapping["segmentation"] = c
        if "keypoint" in cl or "landmark" in cl: mapping["keypoints"] = c
        if "file" in cl or "name" in cl: mapping["file_name"] = c
    
    return df, mapping


def parse_matlab(path):
    from scipy.io import loadmat
    data = loadmat(path)
    # Find the largest array which is likely the annotations
    ann_key = None
    max_len = -1
    for k, v in data.items():
        if isinstance(v, (list, np.ndarray)) and len(v) > max_len:
            max_len = len(v)
            ann_key = k
    return data, ann_key

# ---------------- AUTO FORMAT DETECTION ----------------

def detect_annotations(dataset_path):
    for f in dataset_path.rglob("*"):
        if f.suffix == ".json": return "coco", f
        if f.suffix == ".xml": return "voc", None
        if f.suffix == ".parquet": return "parquet", f
        if f.suffix == ".mat": return "matlab", f
    return None, None

# ---------------- TARGETS ----------------

def create_target(img, task):
    if task in CONFIG["tasks_with_targets"]:
        return img
    return None

# ---------------- TASK DETECTION ----------------

def detect_task(model_dir_name):
    name = model_dir_name.lower()
    if "seg" in name: return "segmentation"
    if "pose" in name or "face" in name: return "pose"
    if "yolo" in name or "detect" in name:
        return "detection"
    if "sr" in name or "zoom" in name:
        return "super_resolution"
    if "denoise" in name:
        return "denoise"
    if "blur" in name:
        return "deblur"
    return "generic"

# ---------------- MAIN PROCESS ----------------

def process_dataset():
    lock = create_structure(OUTPUT_ROOT)
    
    # Initialize Vetting Engine
    from vetting_engine import QualitySentry, AutoLabeler
    sentry = QualitySentry(CONFIG["nima_model_path"]) if CONFIG.get("nima_threshold") else None
    labeler = AutoLabeler(CONFIG["yolo_model_path"]) if CONFIG.get("auto_label") else None
    try:
        index = []
        counters = {}
        seen_hashes = set()

        # Recovery/Global Dedup: Load existing hashes from index.json if present
        existing_index_path = OUTPUT_ROOT / "index.json"
        if existing_index_path.exists():
            try:
                old_data = json.load(open(existing_index_path))
                for item in old_data:
                    if item.get("hash"): seen_hashes.add(item["hash"])
                print(f"📥 [RESILIENCE] Loaded {len(seen_hashes)} existing hashes for global deduplication.")
            except: pass

        for model_dir in INPUT_ROOT.iterdir():
            if not model_dir.is_dir(): continue

            task = detect_task(model_dir.name)

            for dataset in model_dir.iterdir():
                if not dataset.is_dir(): continue

                prefix = dataset.name.replace("-","_")
                counters.setdefault(prefix, 0)

                fmt, ann_path = detect_annotations(dataset)

                coco_images, coco_anns = (None, None)
                df_parquet = None
                mat_data = None

                if fmt == "coco":
                    coco_images, coco_anns = parse_coco(ann_path)
                elif fmt == "parquet":
                    df_parquet = parse_parquet(ann_path)
                elif fmt == "matlab":
                    mat_data = parse_matlab(ann_path)

                images = list(dataset.rglob("*.jpg")) + list(dataset.rglob("*.png"))

                for img_path in images:
                    if counters[prefix] >= CONFIG["max_per_dataset"]:
                        break

                    if not is_valid_image(img_path): continue

                    img = Image.open(img_path)
                    img = ensure_srgb(img)

                    if is_black_image(img): continue

                    # NIMA Scoring (Garbage removal)
                    nima_score = 10.0
                    if sentry:
                        nima_score = sentry.score(img)
                        if nima_score < CONFIG["nima_threshold"]:
                            print(f"🗑️  [SENTRY] Discarding {img_path.name} (Score: {nima_score:.2f})")
                            continue

                    if CONFIG["enable_dedup"]:
                        h = compute_hash(img)
                        if h in seen_hashes: continue
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

                    out_img = OUTPUT_ROOT / "images" / split / f"{name}.jpg"
                    img.save(out_img, "JPEG", quality=95)

                    label_path = OUTPUT_ROOT / "labels" / split / f"{name}.txt"
                    annotations = []

                    if fmt == "coco":
                        for img_id, meta in coco_images.items():
                            if meta["file_name"] == img_path.name:
                                for a in coco_anns.get(img_id, []):
                                    if "keypoints" in a and a["keypoints"]:
                                        # Standard COCO: [x1, y1, v1, x2, y2, v2...] in absolute pixels
                                        kpts = normalize_points(a["keypoints"], w, hgt, stride=3)
                                        annotations.append({"type": "pose", "cls": a["category_id"], "data": a["bbox"] + kpts})
                                    elif "segmentation" in a and a["segmentation"]:
                                        # Standard COCO: polygons [[x1, y1, x2, y2...]] in absolute pixels
                                        poly_raw = a["segmentation"][0] if isinstance(a["segmentation"], list) and len(a["segmentation"]) > 0 else []
                                        if poly_raw:
                                            poly = normalize_points(poly_raw, w, hgt, stride=2)
                                            annotations.append({"type": "segmentation", "cls": a["category_id"], "data": poly})
                                    else:
                                        annotations.append({"type": "bbox", "cls": a["category_id"], "data": a["bbox"]})

                    elif fmt == "voc":
                        xml = img_path.with_suffix(".xml")
                        if xml.exists(): 
                            for cls, bbox in parse_voc(xml):
                                annotations.append({"type": "bbox", "cls": cls, "data": bbox})
                    
                    elif fmt == "parquet" and df_parquet is not None:
                        df_parquet_data, mapping = df_parquet
                        df_subset = df_parquet_data[df_parquet_data[mapping.get("file_name", "file_name")] == img_path.name]
                        for _, row in df_subset.iterrows():
                            cls = row[mapping.get("class", "class")]
                            bbox = [row[mapping.get("xmin", "xmin")], row[mapping.get("ymin", "ymin")], 
                                    row[mapping.get("width", "width")], row[mapping.get("height", "height")]]
                            annotations.append({"type": "bbox", "cls": cls, "data": bbox})

                    elif fmt == "matlab" and mat_data is not None:
                        data, key = mat_data
                        if key in data:
                            for entry in data[key]:
                                try:
                                    if entry["image_name"] == img_path.name or entry["name"] == img_path.name:
                                        annotations.append({"type": "bbox", "cls": entry["class"], "data": entry["bbox"]})
                                except: pass

                    # YOLO Auto-Labeling (Fill missing labels)
                    is_autolabeled = False
                    if not annotations and labeler:
                        # Switch labeler mode based on task if possible
                        if task == "segmentation": labeler.mode = "segmentation"
                        elif task == "pose": labeler.mode = "pose"
                        else: labeler.mode = "detection"
                        
                        annotations = labeler.predict(img)
                        if annotations:
                            is_autolabeled = True
                            print(f"🤖 [AUTO-LABEL] Generated {len(annotations)} {labeler.mode} labels for {img_path.name}")

                    with open(label_path, "w") as f:
                        for ann in annotations:
                            cls = ann["cls"]
                            data = ann["data"]
                            if ann["type"] == "bbox":
                                # Standard Detection
                                yolo = convert_bbox_xywh_to_yolo(data, w, hgt)
                                f.write(f"{cls} {' '.join(map(str,yolo))}\n")
                            elif ann["type"] == "segmentation":
                                # Standard Segmentation (Normalized poly already)
                                f.write(f"{cls} {' '.join(map(str,data))}\n")
                            elif ann["type"] == "pose":
                                # Standard Pose (Normalized xywh + kpts x y v)
                                yolo_box = convert_bbox_xywh_to_yolo(data[:4], w, hgt)
                                kpts = data[4:]
                                f.write(f"{cls} {' '.join(map(str,yolo_box))} {' '.join(map(str,kpts))}\n")

                    target_img = create_target(img, task)
                    if target_img is not None:
                        tgt_path = OUTPUT_ROOT / "targets" / split / f"{name}.jpg"
                        target_img.save(tgt_path, "JPEG", quality=95)

                    index.append({
                        "name": name,
                        "source": prefix,
                        "task": task,
                        "split": split,
                        "format": fmt,
                        "hash": h,
                        "nima_score": round(nima_score, 3),
                        "is_autolabeled": is_autolabeled,
                        "has_segmentation": any(a["type"] == "segmentation" for a in annotations),
                        "has_pose": any(a["type"] == "pose" for a in annotations),
                        "label_path": str(label_path.resolve()),
                        "path": str(out_img.resolve()),
                        "target_path": str(tgt_path.resolve()) if target_img is not None else None
                    })

        with open(OUTPUT_ROOT / "index.json", "w") as f:
            json.dump(index, f, indent=2)

    finally:
        if lock and lock.exists():
            lock.unlink()

# ---------------- DATASET YAML ----------------

def generate_dataset_yaml():
    yaml = f"""
path: {OUTPUT_ROOT.resolve()}
train: images/train
val: images/val

nc: 80
names: []
"""
    with open(OUTPUT_ROOT / "dataset.yaml", "w") as f:
        f.write(yaml)

# ---------------- TRAIN CONFIG ----------------

def generate_train_config():
    cfg = {
        "imgsz": 640,
        "batch": 16,
        "epochs": 100,
        "optimizer": "AdamW",
        "lr": 0.001
    }
    with open(OUTPUT_ROOT / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

# ---------------- README ----------------

def generate_readme():
    data = json.load(open(OUTPUT_ROOT / "index.json"))

    train = [x for x in data if x["split"] == "train"]
    val = [x for x in data if x["split"] == "val"]

    tasks = {}
    sources = {}

    for item in data:
        tasks[item["task"]] = tasks.get(item["task"], 0) + 1
        sources[item["source"]] = sources.get(item["source"], 0) + 1

    readme = f"""
# YOLO Dataset (Compiled)

## Summary
- Total: {len(data)}
- Train: {len(train)}
- Val: {len(val)}

## Tasks
"""
    for k,v in tasks.items():
        readme += f"- {k}: {v}\n"

    readme += "\n## Sources\n"
    for k,v in sources.items():
        readme += f"- {k}: {v}\n"

    readme += f"""

## Quality
- Dedup: {CONFIG['enable_dedup']}
- Black filter: {CONFIG['black_threshold']}

## Recommended Models
- Detection: YOLOv8 / RT-DETR
- SR: SwinIR / ESRGAN
- Restoration: NAFNet / MIRNet

## Loss
- Detection: CIoU + Focal
- SR: L1 + Perceptual + FFT
- Restoration: Charbonnier

## Convergence
- Detection mAP50: avg 0.45 / good 0.6 / sota 0.75+
- SR PSNR: avg 26 / good 30 / sota 32+
- Restoration SSIM: avg 0.75 / good 0.85 / sota 0.9+
"""

    with open(OUTPUT_ROOT / "README.md", "w") as f:
        f.write(readme)

# ---------------- RUN ----------------
if __name__ == "__main__":
    process_dataset()
    generate_dataset_yaml()
    generate_train_config()
    generate_readme()
