import os
import json
import random
import argparse
import hashlib
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image, ImageOps
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import multiprocessing
import io
import sqlite3
import webdataset as wds
import yaml
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from safetensors import safe_open

def get_dir_size(path):
    """Calculate recursive directory size in GB."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except (PermissionError, OSError):
        pass
    return total / (1024**3)

# ---------------- CONFIG ----------------
CONFIG_PATH = Path("./config.json")
DEFAULT_CONFIG = {
    "train_split": 0.8,
    "num_workers": max(1, multiprocessing.cpu_count() - 2),
    "diffusion_size": 512
}
CONFIG = {**DEFAULT_CONFIG, **json.load(open(CONFIG_PATH))} if CONFIG_PATH.exists() else DEFAULT_CONFIG

# Load YAML for dynamic config
YAML_DATA = yaml.safe_load(open(Path("./unified_data.yaml")))
META = YAML_DATA.get("_registry_metadata", {})
VERSION = META.get("version", "4.2.0")

# ---------------- CLI ARGS ----------------
parser = argparse.ArgumentParser(description="LemGendary Dataset Compiler v3.1")
parser.add_argument("--name", type=str, default=META.get("output_folder_name", "sota_synthesis"), help="Name of the compiled dataset")
parser.add_argument("--model", type=str, default=None, help="Specific dataset model to compile")
parser.add_argument("--max_gb", type=float, default=None, help="Override max_size_gb")
parser.add_argument("--suffix", type=str, default=None, help="Override suffix")
parser.add_argument("--workers", type=int, default=DEFAULT_CONFIG["num_workers"], help="Number of parallel workers")
args, unknown = parser.parse_known_args()

INPUT_ROOT = Path("./raw-sets")
OUT_PARENT = Path(META.get("output_folder_name", "compiled-datasets"))
OUTPUT_ROOT = OUT_PARENT / f"v_{VERSION}"
CATEGORY_MAP_PATH = Path("./category_map.json")
CATEGORY_MAP = json.load(open(CATEGORY_MAP_PATH)) if CATEGORY_MAP_PATH.exists() else {}
DATASETS_META = YAML_DATA.get("datasets", {})

# Override workers if specified
if args.workers: CONFIG["num_workers"] = args.workers

# ---------------- SHARED RESOURCES ----------------
# Initialized in workers
SENTRY = None
LABELER = None
CAPTIONER = None
CLIP_MANIFOLD = None

def init_worker(config):
    global SENTRY, LABELER, CAPTIONER
    # Local imports to avoid pickling issues
    from vetting_engine import QualitySentry, AutoLabeler, CaptionSentry
    
    # Optional: Logic to select GPU based on worker id if multiple GPUs available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # NIMA Model (QualitySentry)
    model_path = "c:/Development/python/model-training/lemgendary-training-suite/trained-models/nima_technical/nima_technical_best.pth"
    if os.path.exists(model_path):
        SENTRY = QualitySentry(model_path, device=device)
    
    # BLIP Model (CaptionSentry)
    CAPTIONER = CaptionSentry(device=device)

    # CLIP Model (CLIPManifold)
    from vetting_engine import CLIPManifold
    CLIP_MANIFOLD = CLIPManifold(device=device)

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

# ---------------- REGISTRY ----------------
def initialize_registry(db_path):
    conn = sqlite3.connect(db_path)
    # 2026 Resilience: We store BLOBs for latents and bytes for rapid Pass-2 retrieval
    conn.execute("""
        CREATE TABLE IF NOT EXISTS registry (
            id INTEGER PRIMARY KEY,
            name TEXT, source TEXT, task TEXT, split TEXT,
            hash TEXT, nima_score REAL, caption TEXT,
            style_tag TEXT, clip_latent BLOB,
            img_bytes BLOB, cluster_id INTEGER DEFAULT -1
        )
    """)
    return conn

def map_category(cat_name_or_id, source_name):
    # If it's a number, we might need source-specific ID mapping (TODO: load mapping catalogs)
    # For now, we normalize string names or use raw if missing
    if isinstance(cat_name_or_id, str):
        name = cat_name_or_id.lower().strip()
        return CATEGORY_MAP.get(name, 0) # Default to 0 (Person) if unknown
    return int(cat_name_or_id)

class ShardWriter:
    """Industrial TAR sharding using WebDataset"""
    def __init__(self, output_dir, prefix="data", max_size=1e9):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sink = wds.ShardWriter(str(self.output_dir / f"{prefix}-%05d.tar"), maxsize=max_size)

    def write(self, name, img_bytes, caption):
        self.sink.write({
            "__key__": name,
            "jpg": img_bytes,
            "txt": caption
        })

    def close(self):
        self.sink.close()

# ---------------- FORMAT PARSERS ----------------
def detect_annotations(path):
    path = Path(path)
    # 2026 Resilience: Multi-format annotation discovery
    for f in path.rglob("*.json"):
        if "coco" in f.name.lower() or "instances" in f.name.lower(): return "coco", f
    for f in path.rglob("*.parquet"): return "parquet", f
    for f in path.rglob("*.mat"): return "matlab", f
    for f in path.rglob("*.safetensors"): return "safetensors", f
    return None, None

def detect_task(model_dir_name):
    name = model_dir_name.lower()
    if "diffusion" in name: return "diffusion"
    if any(k in name for k in ["seg", "mask", "parsenet"]): return "segmentation"
    if any(k in name for k in ["pose", "face", "codeformer"]): return "pose"
    if any(k in name for k in ["nima", "aesthetic", "quality"]): return "quality"
    if any(k in name for k in ["sr", "ultrazoom", "x2", "x3", "x4", "x8", "super"]): return "super-resolution"
    if any(k in name for k in ["restorer", "enhance", "upn", "lowlight", "exposure", "deraining", "debluring", "denoising", "haze", "restoration", "ffanet", "mirnet", "mprnet", "nafnet"]): 
        return "restoration"
    return "detection"

def parse_coco(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    images = {x["id"]: x for x in data.get("images", [])}
    anns = {}
    for a in data.get("annotations", []):
        img_id = a["image_id"]
        if img_id not in anns: anns[img_id] = []
        anns[img_id].append(a)
    return images, anns

def parse_parquet(pq_path):
    import pandas as pd
    df = pd.read_parquet(pq_path)
    # Detect common schemas
    mapping = {}
    cols = df.columns.tolist()
    if "image" in cols: mapping["file_name"] = "image"
    if "label" in cols: mapping["class"] = "label"
    # Additional mappings for bbox/seg if needed
    for c in cols:
        cl = c.lower()
        if any(x in cl for x in ["xmin", "x1"]): mapping["xmin"] = c
        if any(x in cl for x in ["ymin", "y1"]): mapping["ymin"] = c
        if any(x in cl for x in ["width", "w"]): mapping["width"] = c
        if any(x in cl for x in ["height", "h"]): mapping["height"] = c
    return df, mapping

def parse_matlab(mat_path):
    import scipy.io as sio
    data = sio.loadmat(mat_path)
    # Heuristic for finding the annotation key
    key = [k for k in data.keys() if not k.startswith("__")][0]
    return data, key

def parse_safetensors(st_path):
    metadata = {}
    try:
        with safe_open(st_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
    except Exception:
        pass
    return metadata

# ---------------- PROCESSORS ----------------
def process_image(img_path, prefix, idx, task, fmt, ann_data, split):
    """Worker function for parallel processing"""
    try:
        # Validity
        is_st = img_path.suffix.lower() == ".safetensors"
        if not img_path.exists(): return None
        
        if is_st:
            # Safetensors representative image logic
            img = None
            for ext in [".jpg", ".png", ".webp"]:
                companion = img_path.with_suffix(ext)
                if companion.exists():
                    img = Image.open(companion)
                    break
            
            if img is None:
                # Create a placeholder image
                img = Image.new("RGB", (512, 512), color=(30, 30, 30))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                draw.text((10, 250), f"MODEL: {img_path.stem}", fill=(200, 200, 200))
        else:
            img = Image.open(img_path)
            
        img = ensure_srgb(img)
        if not is_st and is_black_image(img): return None
        
        w, hgt = img.size
        # For diffusion, we might allow smaller but we'll upscale/standardize later
        min_dim = 128 if task != "diffusion" else 64
        if w < min_dim or hgt < min_dim: return None

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
                    except Exception:
                        pass

        elif fmt == "safetensors" and ann_data:
            metadata = ann_data
            # Extract tags from common metadata keys (Kohya/Civitai style)
            tags = []
            if "ss_tag_frequency" in metadata:
                try:
                    freqs = json.loads(metadata["ss_tag_frequency"])
                    for bucket in freqs.values():
                        tags.extend(bucket.keys())
                except: pass
            
            if not tags and "ss_datasets" in metadata:
                try:
                    ds_info = json.loads(metadata["ss_datasets"])
                    for ds in ds_info:
                        if "tag_frequency" in ds:
                            tags.extend(ds["tag_frequency"].keys())
                except: pass

            if tags:
                # Diffusion YOLO: Convert categories to classification-style labels
                unique_tags = list(set(tags))[:20] # Cap at 20 tags
                for tag in unique_tags:
                    cls = map_category(tag, prefix)
                    annotations.append({"type": "bbox", "cls": cls, "data": [0.0, 0.0, 1.0, 1.0]}) # Whole image
        
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

def process_diffusion(img_path, prefix, idx, split):
    """Specialized Text-Image processor for Diffusion Models"""
    try:
        if not img_path.exists(): return None
        img = Image.open(img_path)
        img = ensure_srgb(img)
        if is_black_image(img): return None
        
        # NIMA Quality Filter
        nima_score = 10.0
        if SENTRY:
            nima_score = SENTRY.score(img)
            if nima_score < CONFIG["nima_threshold"]: return None

        # Harmonize to Target Resolution (512x512 standard)
        size = CONFIG.get("diffusion_size", 512)
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Auto-Captioning
        caption = "a high quality image"
        if CAPTIONER:
            # Check for native captions first (DiffusionDB convention)
            caption_file = img_path.parent / (img_path.stem + ".txt")
            if caption_file.exists():
                caption = caption_file.read_text().strip()
            else:
                caption = CAPTIONER.generate(img)

        # Style & Aesthetic Manifold (v5.0)
        style_tag = "standard"
        clip_latent = None
        if CLIP_MANIFOLD:
            style_tag = CLIP_MANIFOLD.tag_style(img)
            clip_latent = CLIP_MANIFOLD.extract_features(img).cpu().numpy().flatten().tolist()
        
        # Hash-based dedup
        h = compute_hash(img) if CONFIG["enable_dedup"] else None
        name = f"{prefix}_{idx:09d}"
        
        # Convert to bytes for sharding (buffered for Pass 2)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        img_bytes = buffer.getvalue()
        
        return {
            "name": name, "source": prefix, "task": "diffusion", "split": split,
            "hash": h, "nima_score": round(nima_score, 3), 
            "caption": caption, "style_tag": style_tag, "clip_latent": clip_latent,
            "img_bytes": img_bytes, "size": len(img_bytes)
        }
    except Exception as e:
        print(f"❌ Error processing diffusion sample {img_path}: {e}")
        return None

# ---------------- ORCHESTRATOR ----------------
def process_dataset():
    if not OUTPUT_ROOT.exists():
        for d in ["images", "labels", "targets"]:
            for s in ["train", "val"]: (OUTPUT_ROOT / d / s).mkdir(parents=True, exist_ok=True)

    index = []
    seen_hashes = set()
    
    # PASS 1: Extraction & Latent Collection
    db_path = OUTPUT_ROOT / "manifold_registry.db"
    conn = initialize_registry(db_path)
    
    # Metadata already loaded globally
    min_gb = META.get("global_constraints", {}).get("min_size_gb", 0.1)
    max_gb = args.max_gb if args.max_gb is not None else META.get("global_constraints", {}).get("max_size_gb", 50.0)
    prefix_str = META.get("name_prefix", "")
    suffix_str = args.suffix if args.suffix is not None else META.get("name_suffix", "")

    shared_root = INPUT_ROOT
    print(f"🚀 [SOTA v5.0] Commencing PASS 1: Latent Extraction & Vetting...")
    with ProcessPoolExecutor(max_workers=CONFIG["num_workers"], initializer=init_worker, initargs=(CONFIG,)) as executor:
        futures = []
        
        # 2026 Shift: Universal Registry-First Iteration
        for model_key, model_config in DATASETS_META.items():
            if args.model and model_key != args.model: continue
            task = detect_task(model_key)
            pascal_name = model_config.get("name", model_key.replace("_", ""))
            prefix = pascal_name
            
            for ref_entry in model_config.get("refs", []):
                ref = ref_entry["ref"]
                slug = ref.replace('hf://', '').split('/')[-1]
                dataset = shared_root / slug
                
                if not dataset.is_dir(): continue
                
                # Shared Manifold Size Filtering
                d_size = get_dir_size(dataset)
                if d_size < min_gb or d_size > max_gb:
                    print(f"⚠️  [SKIPPED] {slug}: {d_size:.2f}GB outside manifold constraints ({min_gb}-{max_gb}GB)")
                    continue

                fmt, ann_path = detect_annotations(dataset)
                ann_data = None
                if fmt == "coco":
                    ann_data = parse_coco(ann_path)
                elif fmt == "parquet":
                    ann_data = parse_parquet(ann_path)
                elif fmt == "matlab":
                    ann_data = parse_matlab(ann_path)

                images = list(dataset.rglob("*.jpg")) + list(dataset.rglob("*.png"))
                if fmt == "safetensors":
                    st_files = list(dataset.rglob("*.safetensors"))
                    images.extend(st_files)
                
                print(f"[QUEUE] {prefix} ({task}) | {slug} | {len(images)} samples scheduled.")
                for i, img_path in enumerate(images):
                    if i >= CONFIG.get("max_per_dataset", 10000): break
                    split = "train" if random.random() < CONFIG["train_split"] else "val"
                    if task == "diffusion":
                        futures.append(executor.submit(process_diffusion, img_path, prefix, i, split))
                    else:
                        futures.append(executor.submit(process_image, img_path, prefix, i, task, fmt, ann_data, split))

        with tqdm(total=len(futures), desc="[PASS 1] Extraction & Vetting") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res:
                    if CONFIG["enable_dedup"] and res["hash"] in seen_hashes: 
                        pbar.update(1)
                        continue
                    seen_hashes.add(res["hash"])
                    
                    # Commit to SQLite Manifold
                    latent_blob = sqlite3.Binary(np.array(res.get("clip_latent", [])).astype(np.float32).tobytes())
                    conn.execute("""
                        INSERT INTO registry (name, source, task, split, hash, nima_score, caption, style_tag, clip_latent, img_bytes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (res["name"], res["source"], res["task"], res["split"], res["hash"], res["nima_score"], 
                          res.get("caption"), res.get("style_tag"), latent_blob, res.get("img_bytes")))
                    
                    pbar.update(1)
        conn.commit()

    # STEP 2: Style Clustering (v5.0 Global Manifold)
    print(f"[STYLING] Commencing Style Clustering on all extracted latents...")
    cursor = conn.execute("SELECT id, clip_latent FROM registry WHERE clip_latent IS NOT NULL")
    ids, latents = [], []
    for row in cursor:
        ids.append(row[0]); latents.append(np.frombuffer(row[1], dtype=np.float32))
    
    if latents:
        X = np.stack(latents)
        n_clusters = CONFIG.get("n_style_clusters", 16)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X)
        labels = kmeans.labels_
        for i, cid in tqdm(zip(ids, labels), total=len(ids), desc="[STYLING] Updating Clusters"):
            conn.execute("UPDATE registry SET cluster_id = ? WHERE id = ?", (int(cid), i))
        conn.commit()

    # PASS 2: Balanced Interleaving & Sharding per Dataset (as requested)
    print(f"[SHARD] Commencing PASS 2: Multi-Domain Balanced Sharding...")
    shard_dir = OUTPUT_ROOT / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    # We group by source and create separate tars if plural "compiled sets" is desired
    # Or we use the prefix/suffix for the overall manifold name.
    # The user said: "appended to each compiled set name on exporting compiled sets"
    # I'll create a shard per dataset to satisfy "each".
    
    unique_sources = [r[0] for r in conn.execute("SELECT DISTINCT source FROM registry").fetchall()]
    
    for source in unique_sources:
        shard_name = f"{prefix_str}{source}{suffix_str}.tar"
        print(f"[SHARD] Writing {shard_name}...")
        with wds.TarWriter(str(shard_dir / shard_name)) as sink:
            cursor = conn.execute("SELECT * FROM registry WHERE source = ? ORDER BY cluster_id, id", (source,))
            for row in cursor:
                res = {"id": row[0], "name": row[1], "source": row[2], "task": row[3], "split": row[4],
                       "hash": row[5], "nima_score": row[6], "caption": row[7], "style_tag": row[8], "cluster_id": row[11]}
                
                if res["task"] == "diffusion" and row[10]:
                    sink.write({
                        "__key__": res["name"],
                        "jpg": row[10],
                        "txt": res["caption"],
                        "json": json.dumps({"style": res["style_tag"], "cluster": res["cluster_id"], "source": res["source"]})
                    })
                index.append(res)
    
    with open(OUTPUT_ROOT / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    
    generate_dataset_yaml()
    generate_readme()
    print(f"[SUCCESS] v5.0 Ascension Complete: {len(index)} samples compiled.")

# ---------------- GENERATORS ----------------
def generate_dataset_yaml():
    yaml = f"""
# SOTA YOLO Dataset Configuration
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
