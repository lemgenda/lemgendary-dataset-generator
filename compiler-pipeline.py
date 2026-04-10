import os
import json
import random
import hashlib
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image, ImageOps
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import multiprocessing

# ---------------- CONFIG ----------------
CONFIG_PATH = Path("./config.json")
DEFAULT_CONFIG = {
    "train_split": 0.8,
    "img_min": 128,
    "max_per_dataset": 10000,
    "nima_threshold": 4.5,
    "black_threshold": 0.02,
    "enable_dedup": True,
    "enable_multitask": True,
    "num_workers": max(1, multiprocessing.cpu_count() - 2)
}
CONFIG = {**DEFAULT_CONFIG, **json.load(open(CONFIG_PATH))} if CONFIG_PATH.exists() else DEFAULT_CONFIG

INPUT_ROOT = Path("./raw-sets")
OUTPUT_ROOT = Path("./compiled-datasets")
CATEGORY_MAP_PATH = Path("./category_map.json")
CATEGORY_MAP = json.load(open(CATEGORY_MAP_PATH)) if CATEGORY_MAP_PATH.exists() else {}

# ---------------- SHARED RESOURCES ----------------
# Initialized in workers
SENTRY = None
LABELER = None

def init_worker(config):
    global SENTRY, LABELER
    # Local imports to avoid pickling issues
    from vetting_engine import QualitySentry, AutoLabeler
    
    # Optional: Logic to select GPU based on worker id if multiple GPUs available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # NIMA Model (QualitySentry)
    model_path = "c:/Development/python/model-training/lemgendary-training-suite/trained-models/nima_technical/nima_technical_best.pth"
    if os.path.exists(model_path):
        SENTRY = QualitySentry(model_path, device=device)
    
    # YOLO Model (AutoLabeler) - Initialized lazy based on task in process_image
    LABELER = {} # Cache for task-specific labelers

def get_labeler(task, device="cuda"):
    from vetting_engine import AutoLabeler
    if task not in LABELER:
        # Map task to model type
        mode = "detection"
        if task == "segmentation": mode = "segmentation"
        elif task == "pose": mode = "pose"
        LABELER[task] = AutoLabeler(mode=mode, device=device)
    return LABELER[task]

# ---------------- HELPERS ----------------
def ensure_srgb(img):
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def is_black_image(img, threshold=None):
    if threshold is None: threshold = CONFIG["black_threshold"]
    grayscale = img.convert("L")
    stat = np.array(grayscale)
    black_ratio = np.sum(stat < 10) / stat.size
    return black_ratio > (1.0 - threshold)

def compute_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()

def convert_bbox_xywh_to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    return [round((x + bw/2)/w, 6), round((y + bh/2)/h, 6), round(bw/w, 6), round(bh/h, 6)]

def normalize_points(points, w, h, stride=2):
    norm = []
    for i in range(0, len(points), stride):
        norm.append(round(points[i] / w, 6))
        norm.append(round(points[i+1] / h, 6))
        if stride == 3: norm.append(points[i+2])
    return norm

def map_category(cat_name_or_id, source_name):
    # If it's a number, we might need source-specific ID mapping (TODO: load mapping catalogs)
    # For now, we normalize string names or use raw if missing
    if isinstance(cat_name_or_id, str):
        name = cat_name_or_id.lower().strip()
        return CATEGORY_MAP.get(name, 0) # Default to 0 (Person) if unknown
    return int(cat_name_or_id)

# ---------------- PROCESSORS ----------------
def process_image(img_path, prefix, idx, task, fmt, ann_data, split):
    """Worker function for parallel processing"""
    try:
        # Validity
        if not img_path.exists(): return None
        img = Image.open(img_path)
        img = ensure_srgb(img)
        if is_black_image(img): return None
        
        w, hgt = img.size
        if w < CONFIG["img_min"] or hgt < CONFIG["img_min"]: return None

        # NIMA Quality
        nima_score = 10.0
        if SENTRY:
            nima_score = SENTRY.score(img)
            if nima_score < CONFIG["nima_threshold"]: return None

        # Meta Preparation
        h = compute_hash(img) if CONFIG["enable_dedup"] else None
        name = f"{prefix}_{idx:09d}"
        
        # Save Output Image
        out_img_path = OUTPUT_ROOT / "images" / split / f"{name}.jpg"
        img.save(out_img_path, "JPEG", quality=95)
        
        # Annotations
        annotations = []
        if fmt == "coco":
            images_meta, anns_meta = ann_data
            img_id = next((k for k, v in images_meta.items() if v["file_name"] == img_path.name), None)
            if img_id is not None:
                for a in anns_meta.get(img_id, []):
                    cls = map_category(str(a["category_id"]), prefix) # Placeholder for COCO meta names
                    if "keypoints" in a and a["keypoints"]:
                        kpts = normalize_points(a["keypoints"], w, hgt, stride=3)
                        annotations.append({"type": "pose", "cls": cls, "data": a["bbox"] + kpts})
                    elif "segmentation" in a and a["segmentation"]:
                        poly_raw = a["segmentation"][0] if isinstance(a["segmentation"], list) and len(a["segmentation"]) > 0 else []
                        if poly_raw:
                            poly = normalize_points(poly_raw, w, hgt, stride=2)
                            annotations.append({"type": "segmentation", "cls": cls, "data": poly})
                    else:
                        annotations.append({"type": "bbox", "cls": cls, "data": a["bbox"]})

        elif fmt == "parquet" and ann_data:
            df, mapping = ann_data
            df_subset = df[df[mapping.get("file_name", "file_name")] == img_path.name]
            for _, row in df_subset.iterrows():
                cls = map_category(row[mapping.get("class", "class")], prefix)
                # Greedy multitask from Parquet
                if mapping.get("segmentation") in row and row[mapping.get("segmentation")]:
                    poly = normalize_points(row[mapping.get("segmentation")], w, hgt, stride=2)
                    annotations.append({"type": "segmentation", "cls": cls, "data": poly})
                elif mapping.get("keypoints") in row and row[mapping.get("keypoints")]:
                    kpts = normalize_points(row[mapping.get("keypoints")], w, hgt, stride=3)
                    annotations.append({"type": "pose", "cls": cls, "data": [0,0,0,0] + kpts}) # Placeholder box
                else:
                    bbox = [row[mapping.get("xmin", "xmin")], row[mapping.get("ymin", "ymin")], 
                            row[mapping.get("width", "width")], row[mapping.get("height", "height")]]
                    annotations.append({"type": "bbox", "cls": cls, "data": bbox})

        elif fmt == "matlab" and ann_data:
            data, key = ann_data
            if key in data:
                for entry in data[key]:
                    try:
                        if entry["image_name"] == img_path.name or entry["name"] == img_path.name:
                            cls = map_category(entry["class"], prefix)
                            annotations.append({"type": "bbox", "cls": cls, "data": entry["bbox"]})
                    except: pass
        is_autolabeled = False
        if not annotations:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            labeler = get_labeler(task, device)
            annotations = labeler.predict(img)
            if annotations: is_autolabeled = True

        # Write Label File
        label_file_path = OUTPUT_ROOT / "labels" / split / f"{name}.txt"
        with open(label_file_path, "w") as f:
            for ann in annotations:
                cls = ann["cls"]
                data = ann["data"]
                if ann["type"] == "bbox":
                    yolo = convert_bbox_xywh_to_yolo(data, w, hgt)
                    f.write(f"{cls} {' '.join(map(str,yolo))}\n")
                elif ann["type"] == "segmentation":
                    f.write(f"{cls} {' '.join(map(str,data))}\n")
                elif ann["type"] == "pose":
                    yolo_box = convert_bbox_xywh_to_yolo(data[:4], w, hgt)
                    f.write(f"{cls} {' '.join(map(str,yolo_box))} {' '.join(map(str,data[4:]))}\n")

        # Result Meta
        return {
            "name": name, "source": prefix, "task": task, "split": split,
            "hash": h, "nima_score": round(nima_score, 3), "is_autolabeled": is_autolabeled,
            "has_segmentation": any(a["type"] == "segmentation" for a in annotations),
            "has_pose": any(a["type"] == "pose" for a in annotations),
            "label_path": str(label_file_path.resolve()), "path": str(out_img_path.resolve())
        }
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

# ---------------- FORMAT PARSERS ----------------
def parse_coco(json_path):
    data = json.load(open(json_path))
    images = {img["id"]: img for img in data.get("images", [])}
    anns = {}
    for a in data.get("annotations", []):
        anns.setdefault(a["image_id"], []).append(a)
    return images, anns

def parse_parquet(path):
    import pandas as pd
    df = pd.read_parquet(path)
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
    mat = loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys: return None, None
    key = keys[0]
    return mat, key

def detect_task(model_dir_name):
    name = model_dir_name.lower()
    if "seg" in name: return "segmentation"
    if "pose" in name or "face" in name: return "pose"
    return "detection"

def detect_annotations(dataset_path):
    coco = list(dataset_path.glob("*.json"))
    if coco: return "coco", coco[0]
    parquet = list(dataset_path.glob("*.parquet"))
    if parquet: return "parquet", parquet[0]
    mat = list(dataset_path.glob("*.mat"))
    if mat: return "matlab", mat[0]
    return "none", None

# ---------------- ORCHESTRATOR ----------------
def process_dataset():
    if not OUTPUT_ROOT.exists():
        for d in ["images", "labels", "targets"]:
            for s in ["train", "val"]: (OUTPUT_ROOT / d / s).mkdir(parents=True, exist_ok=True)

    index = []
    seen_hashes = set()
    
    print(f"🚀 [SOTA v3.0] Initializing Parallel Pool with {CONFIG['num_workers']} workers...")
    
    with ProcessPoolExecutor(max_workers=CONFIG["num_workers"], initializer=init_worker, initargs=(CONFIG,)) as executor:
        futures = []
        
        for model_dir in INPUT_ROOT.iterdir():
            if not model_dir.is_dir(): continue
            task = detect_task(model_dir.name)
            
            for dataset in model_dir.iterdir():
                if not dataset.is_dir(): continue
                prefix = dataset.name.replace("-","_")
                fmt, ann_path = detect_annotations(dataset)
                
                ann_data = None
                if fmt == "coco": ann_data = parse_coco(ann_path)
                elif fmt == "parquet": 
                    df, mapping = parse_parquet(ann_path)
                    ann_data = (df, mapping)
                elif fmt == "matlab":
                    mat, key = parse_matlab(ann_path)
                    ann_data = (mat, key)
                
                images = list(dataset.rglob("*.jpg")) + list(dataset.rglob("*.png"))
                print(f"📦 [QUEUE] Scheduling {len(images)} images from {prefix}...")
                
                for i, img_path in enumerate(images):
                    if i >= CONFIG["max_per_dataset"]: break
                    split = "train" if random.random() < CONFIG["train_split"] else "val"
                    futures.append(executor.submit(process_image, img_path, prefix, i, task, fmt, ann_data, split))

        for future in as_completed(futures):
            res = future.result()
            if res:
                if CONFIG["enable_dedup"]:
                    if res["hash"] in seen_hashes: continue
                    seen_hashes.add(res["hash"])
                index.append(res)
                if len(index) % 100 == 0: print(f"✅ Compiled {len(index)} images...")

    with open(OUTPUT_ROOT / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    
    generate_dataset_yaml()
    generate_readme()
    print(f"🎉 Synthesis Complete: {len(index)} images compiled. YAML and Manifest generated.")

# ---------------- GENERATORS ----------------
def generate_dataset_yaml():
    yaml = f"""
path: {OUTPUT_ROOT.resolve()}
train: images/train
val: images/val

nc: 80
names: {list(CATEGORY_MAP.keys())[:80]}
"""
    with open(OUTPUT_ROOT / "dataset.yaml", "w") as f:
        f.write(yaml)

def generate_readme():
    if not (OUTPUT_ROOT / "index.json").exists(): return
    data = json.load(open(OUTPUT_ROOT / "index.json"))
    
    train = [x for x in data if x["split"] == "train"]
    val = [x for x in data if x["split"] == "val"]
    
    tasks = {}
    sources = {}
    for item in data:
        tasks[item["task"]] = tasks.get(item["task"], 0) + 1
        sources[item["source"]] = sources.get(item["source"], 0) + 1

    readme = f"""# Compiled Dataset Manifest (SOTA v3.0)
## Summary
- Total: {len(data)}
- Train: {len(train)}
- Val: {len(val)}

## Tasks
"""
    for k, v in tasks.items(): readme += f"- {k}: {v}\n"
    readme += "\n## Sources\n"
    for k, v in sources.items(): readme += f"- {k}: {v}\n"

    with open(OUTPUT_ROOT / "README.md", "w") as f:
        f.write(readme)

if __name__ == "__main__":
    process_dataset()
