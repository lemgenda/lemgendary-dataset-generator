# 2026: Environment Linter Sync (Last Verified: 2026-05-01)
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 2026 Resilience: Force UTF-8 encoding for Windows console support (Prevents UnicodeEncodeError)
if os.name == 'nt':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8') # type: ignore
        sys.stderr.reconfigure(encoding='utf-8') # type: ignore
    os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
    os.environ["FOR_IGNORE_EXCEPTIONS"] = "1"

import json
import pandas as pd
import random
# LemGendary Kaggle Manager (Last Verified: 2026-05-01)
import argparse
import hashlib
import shutil
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image, ImageOps, ImageFile
from doc_generator import generate_dataset_docs
from notebook_generator import generate_training_notebook, generate_colab_training_notebook
ImageFile.LOAD_TRUNCATED_IMAGES = True # type: ignore
Image.MAX_IMAGE_PIXELS = None # Disable DOS limit to allow massive panoramas without warning
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import requests
from multiprocessing import Manager
import multiprocessing
import io
import sqlite3
import webdataset as wds
from datetime import datetime
import json
import pandas as pd
import yaml
import hashlib
from sklearn.cluster import MiniBatchKMeans
import time
from datetime import datetime
from tqdm import tqdm
# 2026 Resilience: Force ASCII progress bars globally to prevent Unicode Mojibake in PowerShell
import functools
from safetensors import safe_open

def get_dir_size(path):
    """Calculate recursive directory size in GB."""
    def _get_bytes(p):
        total = 0
        try:
            for entry in os.scandir(p):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += _get_bytes(entry.path)
        except (PermissionError, OSError):
            pass
        return total
    return _get_bytes(path) / (1024**3)

# ---------------- CONFIG ----------------
CONFIG_PATH = Path("./config.json")
DEFAULT_CONFIG = {
    "train_split": 0.8,
    "num_workers": max(1, multiprocessing.cpu_count() - 2),
    "diffusion_size": 1024,
    "black_threshold": 0.1,
    "nima_threshold": 4.0,
    "enable_dedup": False,
    "strict_ground_truth": False # Allow AI vetting as fallback for sources like LAION
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
parser.add_argument("--reduce", action="store_true", help="Start in Reduce mode")
parser.add_argument("--cleanup", action="store_true", help="Start in Cleanup mode")
parser.add_argument("--finalize", action="store_true", help="Only run sharding/readme for existing registry")
parser.add_argument("--no-vetting", action="store_true", help="Disable NIMA quality gate (Pass-Through mode)")
parser.add_argument("--no-labeling", action="store_true", help="Disable YOLO auto-labeling (High-Speed mode)")
parser.add_argument("--no-hash", action="store_true", help="Disable deduplication hash for maximum I/O speed.")
args = parser.parse_args()

INPUT_ROOT = Path("./raw-sets")
OUT_PARENT = Path(META.get("output_folder_name", "../LemGendaryDatasets"))
CATEGORY_MAP_PATH = Path("./category_map.json")
CATEGORY_MAP = json.load(open(CATEGORY_MAP_PATH)) if CATEGORY_MAP_PATH.exists() else {}
DATASETS_META = YAML_DATA.get("datasets", {})

# Override workers if specified
if args.workers: CONFIG["num_workers"] = args.workers

def get_device_info():
    import torch
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU (No CUDA detected)"

# ---------------- GLOBALS ----------------
SENTRY = None
LABELER = None
CAPTIONER = None
CLIP_MANIFOLD = None
AVA_LOOKUP = {}
AADB_LOOKUP = {}
TID_LOOKUP = {}

def get_gaussian_probs(mean_score, sigma=1.0):
    """SOTA conversion of scalar score to 10-bin distribution."""
    import numpy as np
    x = np.arange(1, 11)
    probs = np.exp(-0.5 * ((x - mean_score) / sigma)**2)
    probs /= probs.sum()
    return probs.tolist()

def load_ground_truth(model_name=""):
    global AVA_LOOKUP, AADB_LOOKUP, TID_LOOKUP
    m_low = model_name.lower()

    # 1. Aesthetic Sources
    ava_csv = Path("./raw-sets/ava-aesthetic-visual-assessment/ground_truth_dataset.csv")
    if ava_csv.exists():
        import pandas as pd
        df = pd.read_csv(ava_csv)
        vote_cols = [f"vote_{i}" for i in range(1, 11)]
        AVA_LOOKUP = df.set_index("image_num")[vote_cols].to_dict("index")
        print(f"[GT] {len(AVA_LOOKUP)} AVA Aesthetic ratings cached.")

    aadb_csv = Path("./raw-sets/aadb-imagedatabase/Dataset.csv")
    if aadb_csv.exists():
        import pandas as pd
        df = pd.read_csv(aadb_csv)
        AADB_LOOKUP = df.set_index("ImageFile")["score"].to_dict()
        print(f"[GT] {len(AADB_LOOKUP)} AADB Aesthetic ratings cached.")

    # 2. Technical Sources (Universal Normalization v7.0)
    # Technical Path Helper (v7.5) - Supports Legacy and Jackpot Mirror paths
    def find_gt_path(base_name, relative_target):
        # Try Jackpot Mirror path first
        p1 = Path("./raw-sets/IQA-PyTorch-Datasets") / base_name / relative_target
        if p1.exists(): return p1
        # Try Legacy top-level path
        p2 = Path("./raw-sets") / base_name / relative_target
        if p2.exists(): return p2
        # Try deep mirror path (sometimes archives have nested names)
        p3 = Path("./raw-sets") / base_name / base_name / relative_target
        if p3.exists(): return p3
        return None

    # KonIQ-10k
    koniq_csv = find_gt_path("koniq-10k-dataset", "koniq10k_distributions_sets.csv")
    if not koniq_csv:
        koniq_csv = find_gt_path("koniq10k", "koniq10k_scores.csv")
    if koniq_csv:
        import pandas as pd
        df = pd.read_csv(koniq_csv)
        for _, row in df.iterrows():
            # Map KonIQ 1-100 scale down to NIMA 1-10 scale
            val = float(row['MOS']) / 10.0
            TID_LOOKUP[str(row['image_name']).lower()] = max(1.0, min(10.0, val))
        print(f"[GT] KonIQ-10k ratings cached.")

    # SPAQ
    spaq_csv = find_gt_path("spaq", "SPAQ/Annotations/MOS_Average.csv")
    if not spaq_csv:
        spaq_csv = find_gt_path("spaq", "Annotations/MOS_Average.csv")
    if spaq_csv:
        import pandas as pd
        df = pd.read_csv(spaq_csv)
        for _, row in df.iterrows():
            TID_LOOKUP[str(row['Image name']).lower()] = 1.0 + float(row['MOS']) * 0.09
        print(f"[GT] SPAQ ratings cached.")

    # TID2013
    tid_txt = find_gt_path("tid2013", "mos_with_names.txt")
    if tid_txt:
        with open(tid_txt, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    TID_LOOKUP[parts[1].strip().lower()] = float(parts[0]) + 1.0
        print(f"[GT] TID2013 ratings cached.")

    # LIVE IQA
    live_csv = find_gt_path("live", "live_scores.csv")
    if live_csv:
        import pandas as pd
        df = pd.read_csv(live_csv)
        for _, row in df.iterrows():
            orig = min(100.0, float(row['dmos']))
            TID_LOOKUP[str(row['image_name']).lower()] = 1.0 + (1.0 - orig/100.0) * 9.0
        print(f"[GT] LIVE IQA ratings cached.")

    # CSIQ
    csiq_csv = find_gt_path("csiq", "csiq_scores.csv")
    if csiq_csv:
        import pandas as pd
        df = pd.read_csv(csiq_csv)
        for _, row in df.iterrows():
            TID_LOOKUP[str(row['image_name']).lower()] = 1.0 + (1.0 - float(row['dmos'])) * 9.0
        print(f"[GT] CSIQ ratings cached.")

    # TAD66K
    tad_labels_dir = find_gt_path("TAD66K_for_Image_Aesthetics_Assessment", "labels/unmerge")
    if not tad_labels_dir or not tad_labels_dir.exists():
        # Hugging Face manager extracts labels.zip into a 'labels' subfolder, creating 'labels/labels/unmerge'
        tad_labels_dir = find_gt_path("TAD66K_for_Image_Aesthetics_Assessment", "labels/labels/unmerge")
    
    if tad_labels_dir and tad_labels_dir.exists():
        import os
        import pandas as pd
        tad_count = 0
        for root, _, files in os.walk(tad_labels_dir):
            for f in files:
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(root, f))
                    for _, row in df.iterrows():
                        if 'image' in row and 'score' in row:
                            TID_LOOKUP[str(row['image']).lower()] = max(1.0, min(10.0, float(row['score'])))
                            tad_count += 1
        print(f"[GT] {tad_count} TAD66K ratings cached.")

def detect_task(model_dir_name):
    if not model_dir_name: return "quality"
    name = str(model_dir_name).lower()
    
    # 2026: Check for explicit task_override in dataset config
    if DATASETS_META:
        for ds_key, ds_cfg in DATASETS_META.items():
            if ds_key.lower() == name or (ds_cfg.get('name', '').lower() == name):
                override = ds_cfg.get('task_override')
                if override:
                    return override
    
    if "diffusion" in name: return "diffusion"
    if any(k in name for k in ["seg", "mask", "parsenet"]): return "segmentation"
    if any(k in name for k in ["pose", "face"]): return "pose"
    if any(k in name for k in ["nima", "aesthetic", "quality"]): return "quality"
    if any(k in name for k in ["classify", "classification", "authentic", "authenticity"]): return "classification"
    if any(k in name for k in ["vlm", "vision_language"]): return "diffusion"
    if any(k in name for k in ["sr", "ultrazoom", "x2", "x3", "x4", "x8", "super"]): return "super-resolution"
    
    # 2026: Surgical Restoration Detection (Purity-First)
    # Note: 'upn' removed -- UPN models use task_override for parameter_prediction
    if any(k in name for k in ["deraining", "debluring", "denoising", "dehazing", "lowlight", "exposure"]):
        return "restoration"
    if any(k in name for k in ["restorer", "enhance", "restoration", "ffanet", "mirnet", "mprnet", "nafnet", "upn", "codeformer"]):
        return "restoration"
    return "detection"

DPED_CACHE = set()
PHYSICAL_INDEX = set()

def init_worker(config, dped_cache=None, physical_index=None):
    global SENTRY, LABELER, CAPTIONER, CLIP_MANIFOLD, DPED_CACHE, PHYSICAL_INDEX
    if dped_cache: DPED_CACHE = dped_cache
    if physical_index: PHYSICAL_INDEX = physical_index
    # 2026 Modular Alignment: Local imports from encapsulated modules
    from models.quality_scorer import QualitySentry # type: ignore
    from models.detection import AutoLabeler # type: ignore
    from models.diffusion import CaptionSentry # type: ignore
    from models.encoder import CLIPManifold # type: ignore

    # 2026 Resilience: Workers ignore SIGINT to prevent traceback noise.
    # ONLY apply to sub-processes.
    if os.name == 'nt' and multiprocessing.current_process().name != 'MainProcess':
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    import torch
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True # type: ignore

    # CRITICAL: Prevent multiprocessing thread thrashing on CPU
    # ONLY apply in ProcessPool mode to avoid ThreadPool deadlocks
    if os.name != 'nt' or multiprocessing.current_process().name != 'MainProcess':
        try:
            import torch
            torch.set_num_threads(1)
        except: pass

    # 2026 Resilience: Multi-processing with PyTorch on Windows CUDA causes severe deadlocks
    # and OOMs if multiple workers allocate GPU memory concurrently on a single 4GB card.
    # However, on Kaggle/Cloud with multiple GPUs (e.g. 2x T4), we must distribute workers across GPUs!
    if os.name == 'nt' or not torch.cuda.is_available():
        device = "cpu"
    else:
        gpu_count = torch.cuda.device_count()
        if physical_index is not None and gpu_count > 0:
            device = f"cuda:{physical_index % gpu_count}"
        else:
            device = "cuda:0"

    # 1. Quality Vetting (NIMA) - Only load if the task requires it
    mission = detect_task(args.model)
    if mission in ["quality", "classification", "diffusion"] and not args.no_vetting:
        # For diffusion, we always prefer the aesthetic model to ensure high-quality taste
        model_type = "aesthetic" if mission == "diffusion" or (args.model and "aesthetic" in args.model) else "technical"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", f"nima_{model_type}_best.pth")

        if os.path.exists(model_path):
            try:
                SENTRY = QualitySentry(model_path, model_name=model_type, device=device)
            except Exception:
                pass

    # 2. Diffusion Specifics (Captions)
    # 3. Ground Truth Support (AVA/AADB/LAION)
    if mission in ["quality", "classification", "diffusion", "restoration"]:
        load_ground_truth(args.model or "")

    # Mission Resolution for Workers
    mission = detect_task(args.model)
    if mission == "diffusion":
        try:
            CAPTIONER = CaptionSentry(device=device)
        except Exception:
            pass

    # 3. Style Manifold (CLIP)
    # Load on CPU if memory is tight, but enable for high-energy manifolds
    if "clip" in str(config):
        try:
            CLIP_MANIFOLD = CLIPManifold(device="cpu")
        except Exception:
            pass

def get_labeler(task, device="cuda"):
    from models.detection import AutoLabeler # type: ignore
    global LABELER
    if LABELER is None: LABELER = {}
    if task not in LABELER:
        if task == "face_detection":
            mode = "face_landmarks"
        else:
            mode = "segmentation" if "seg" in task else "detection"
        LABELER[task] = AutoLabeler(mode=mode, device=device)
    return LABELER[task]

# ---------------- HELPERS ----------------
def ensure_srgb(img):
    if img.mode != "RGB":
        if img.mode in ("RGBA", "P", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
        
        # Now safely convert to RGB, dropping alpha without warning
        img = img.convert("RGB")
        img.was_converted = True
        return img
    return img

def is_black_image(img, threshold=None):
    if threshold is None: threshold = CONFIG["black_threshold"]
    img_thumb = img.resize((64, 64), Image.Resampling.NEAREST) if img.size[0] > 64 else img
    grayscale = img_thumb.convert("L")
    stat = np.array(grayscale)
    black_ratio = np.sum(stat < 10) / stat.size
    return black_ratio > (1.0 - threshold)

def compute_hash(img_or_path):
    if getattr(args, 'no_hash', False): return None
    import hashlib
    from pathlib import Path
    if isinstance(img_or_path, (str, Path)):
        h = hashlib.md5()
        with open(img_or_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
        return h.hexdigest()
    elif isinstance(img_or_path, bytes):
        return hashlib.md5(img_or_path).hexdigest()
    else:
        return hashlib.md5(img_or_path.tobytes()).hexdigest()

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

def download_image(url, dest_path, session=None):
    """SOTA Lazy Downloader with exponential backoff and image validation."""
    if os.path.exists(dest_path): return True
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = (session or requests).get(url, headers=headers, timeout=15, stream=True)
            if r.status_code == 200:
                # 2026 Resilience: Verify it's an actual image, not an HTML error page
                content_type = r.headers.get('Content-Type', '')
                if 'image' not in content_type and 'octet-stream' not in content_type:
                    return False
                
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                return True
            elif r.status_code == 404:
                return False # Don't retry 404s
        except Exception:
            if attempt == max_retries - 1: return False
            time.sleep(2 ** attempt)
    return False

# ---------------- REGISTRY ----------------
def initialize_registry(db_path):
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL") # Balanced for resilience and speed
    conn.execute("PRAGMA cache_size=100000")
    conn.execute("PRAGMA temp_store=MEMORY")
    # 2026 Resilience: We store BLOBs for latents and bytes for rapid Pass-2 retrieval
    conn.execute("""
        CREATE TABLE IF NOT EXISTS registry (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE, source TEXT, task TEXT, split TEXT,
            hash TEXT, nima_score REAL, caption TEXT,
            style_tag TEXT, clip_latent BLOB,
            img_bytes BLOB, cluster_id INTEGER DEFAULT -1
        )
    """)
    return conn

def clean_slug(slug):
    sl = slug.lower()
    # 2026: Only collapse known massive multi-part manifolds
    if "laion" in sl: return "laion"
    if "ava" in sl: return "ava"
    if "aadb" in sl: return "aadb"
    if "ffhq" in sl: return "ffhq"
    # 2026 Resilience: Ensure we don't truncate specialized source names
    return slug.replace(".tar.gz", "").replace(".tgz", "").replace(".zip", "")

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
    # 2026 Optimization: Avoid recursive rglob on 1.4M folders; check common top-level locations
    for f in path.glob("*.json"):
        if "coco" in f.name.lower() or "instances" in f.name.lower(): return "coco", f
    for f in path.glob("*.parquet"): return "parquet", f
    for f in path.glob("*.mat"): return "matlab", f

    # Check one level deeper for common structures (e.g. annotations/instances.json)
    for sub in [path / "annotations", path / "Annotations", path / "labels", path / "metadata", path / "data", path / "landmarks"]:
        if sub.exists():
            for f in sub.glob("*.json"):
                if "coco" in f.name.lower() or "instances" in f.name.lower(): return "coco", f
            for f in sub.glob("*.parquet"): return "parquet", f
            
            # 2026: Directory-level annotation formats (1 file per image)
            if any(sub.glob("*.xml")): return "xml", sub
            if any(sub.glob("*.txt")): return "yolo", sub
            if any(sub.glob("*.npz")): return "npz", sub

    return None, None


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
    import pyarrow.parquet as pq
    schema = pq.read_schema(pq_path)
    cols = schema.names
    # Detect common schemas
    mapping = {}
    if "image" in cols: mapping["file_name"] = "image"
    elif "pixel_values" in cols: mapping["file_name"] = "pixel_values"
    if "url" in cols: mapping["url"] = "url"
    if "key" in cols: mapping["key"] = "key"
    if "label" in cols: mapping["class"] = "label"
    
    # Restoration Targets
    if "target" in cols: mapping["target"] = "target"
    if "sharp" in cols: mapping["target"] = "sharp"
    if "ground_truth" in cols: mapping["target"] = "ground_truth"

    # Additional mappings for bbox/seg if needed
    for c in cols:
        cl = c.lower()
        if any(x in cl for x in ["xmin", "x1"]): mapping["xmin"] = c
        if any(x in cl for x in ["ymin", "y1"]): mapping["ymin"] = c
        if any(x in cl for x in ["width", "w"]): mapping["width"] = c
        if any(x in cl for x in ["height", "h"]): mapping["height"] = c
    return pq_path, mapping, cols

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

def parse_yolo(txt_path, img_w, img_h):
    annotations = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = parts[0]
                    # YOLO is center_x, center_y, width, height (normalized)
                    cx, cy, nw, nh = map(float, parts[1:5])
                    w = nw * img_w
                    h = nh * img_h
                    xmin = (cx * img_w) - (w / 2.0)
                    ymin = (cy * img_h) - (h / 2.0)
                    item = {"class": cls_id, "bbox": [xmin, ymin, w, h]}
                    if len(parts) > 5:
                        kpts = list(map(float, parts[5:]))
                        item["keypoints"] = kpts
                    annotations.append(item)
    except Exception:
        pass
    return annotations

# ---------------- BATCH WORKER ----------------
def process_parquet_shard(pq_path, prefix, c_slug, start_idx, task, fmt, split_fallback, output_root_str, skip_lbl, train_prob, existing_names, existing_on_disk, keep_prob, num_rows):
    import pandas as pd
    import random
    from pathlib import Path
    try:
        df = pd.read_parquet(pq_path)
    except Exception as e:
        print(f"[WARNING] Skipping corrupted virtual parquet shard {pq_path}: {e}")
        return [{"hash": "skipped"}] * num_rows
    
    results = []
    global_idx = start_idx
    for row in df.itertuples():
        current_idx = global_idx
        global_idx += 1
        
        name = f"{prefix}_{c_slug}_{current_idx:09d}"
        
        if keep_prob < 1.0 and random.random() > keep_prob:
            results.append({"name": name, "source": c_slug, "task": task, "split": "skipped", "hash": "dropped", "nima_score": 1.0, "size": 0})
            continue
            
        if name in existing_names or name.lower() in existing_on_disk:
            results.append({"name": name, "source": c_slug, "task": task, "split": "skipped", "hash": "skipped", "nima_score": 1.0, "size": 0})
            continue
            
        img_bytes = getattr(row, "image", getattr(row, "pixel_values", None))
        if img_bytes is None:
            results.append({"name": name, "source": c_slug, "task": task, "split": "skipped", "hash": "skipped", "nima_score": 1.0, "size": 0})
            continue
            
        split = "train" if random.random() < train_prob else "val"
        row_dict = {k: getattr(row, k) for k in df.columns}
        
        if task == "diffusion":
            res = process_diffusion(img_bytes, prefix, c_slug, current_idx, split, output_root_str)
        else:
            res = process_image(img_bytes, prefix, c_slug, current_idx, task, fmt, row_dict, split, output_root_str, skip_lbl)
        
        if res:
            results.append(res)
        else:
            results.append({"name": name, "source": c_slug, "task": task, "split": "skipped", "hash": "failed", "nima_score": 1.0, "size": 0})
            
    return results

def batch_worker(tasks):
    """Executes a list of tasks in a single worker call to reduce IPC overhead."""
    results = []
    for i, (task_func, *args) in enumerate(tasks):
        try:
            res = task_func(*args)
            if task_func.__name__ == "process_parquet_shard":
                results.extend(res)
            else:
                results.append(res)
        except Exception as e:
            results.append(None)

    # 2026 Pulse: Log completion for large batches to confirm worker health
    # Pulse logic removed for type safety

    return results

# ---------------- PROCESSORS ----------------
def process_image(
    img_input, prefix, slug, idx, task, fmt, ann_data, split, output_root_str, skip_labeling=False):
    """
    Worker function for parallel processing.
    img_input can be a Path or raw bytes (for Parquet-embedded datasets).
    """
    # 2026 Resilience: Initialize defaults early to prevent UnboundLocalError during fast-skips
    img_path = "Unknown"
    nima_score = 1.0
    nima_probs = [0.0] * 10
    nima_probs[0] = 1.0
    w, hgt = 0, 0
    img = None
    p_str = ""
    img_data = b""

    try:
        # Validity & Format Handling
        if isinstance(img_input, (bytes, dict)):
            if isinstance(img_input, dict) and "bytes" in img_input:
                img_data = img_input["bytes"]
            else:
                img_data = img_input if isinstance(img_input, bytes) else b""

            if img_data is None: return None

            img = Image.open(io.BytesIO(img_data)) # type: ignore
            # 2026 Resilience: Use a Path object even for virtual, but with a safe Windows-friendly name
            img_path = Path(f"virtual_{slug}_{idx:09d}.jpg")
            is_st = False
        else:
            img_path_str = str(img_input)
            img_path = Path(img_path_str)
            is_st = img_path_str.lower().endswith(".safetensors")

        ext = img_path.suffix.lower() if isinstance(img_input, (str, Path)) else ".jpg"
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".npy"]: ext = ".jpg"

        name = f"{prefix}_{slug}_{idx:09d}"
        
        # 2026 Resilience: Detect structural parent to avoid forcing targets/masks into images/
        source_parent_name = img_path.parent.parent.name.lower() if isinstance(img_input, (str, Path)) and len(img_path.parts) >= 3 else "images"
        out_dir = source_parent_name if source_parent_name in ["targets", "masks"] else "images"
        
        out_img_path = Path(output_root_str) / out_dir / split / f"{name}{ext}"
        
        tgt_dir = "masks" if task == "segmentation" else "targets"
        out_tgt_path = Path(output_root_str) / tgt_dir / split / f"{name}{ext}"

        # 2026 Optimization: High-Speed Skip (Using Worker-Global Physical Index)
        # This eliminates the need for expensive os.path.exists() calls on 1.4M files.
        # We use .lower() to ensure case-insensitive matching regardless of OS.
        if PHYSICAL_INDEX and name.lower() in PHYSICAL_INDEX:
            return {"name": name, "source": slug, "task": task, "split": split, "hash": "skipped", "nima_score": nima_score, "size": 0}

        # Restoration Target Resolver (v5.8 Hardened)
        target_img = None
        target_img_path = None

        if task in ["restoration", "super-resolution", "parameter_prediction", "segmentation"]:
            # 1. Parquet/Virtual Target Detection
            if ann_data:
                row_dict = None
                if isinstance(ann_data, dict): row_dict = ann_data
                elif isinstance(ann_data, tuple) and len(ann_data) == 2:
                    df_sub, _ = ann_data
                    if not df_sub.empty: row_dict = df_sub.iloc[0].to_dict()
                
                if row_dict:
                    for k in ["target", "sharp", "ground_truth", "gt", "clean", "original", "mask", "masks"]:
                        val = row_dict.get(k)
                        if isinstance(val, bytes):
                            target_img = Image.open(io.BytesIO(val))
                            break
                        elif isinstance(val, str) and (val.endswith((".png", ".jpg", ".jpeg"))):
                            # Try absolute or relative to dataset root
                            p = Path(val)
                            if p.exists(): target_img_path = str(p)
                            break

            # 2. Generic Neighbor Resolve (GoPro, RealBlur, HideBlur, SFHQ)
            if not target_img and not target_img_path and not isinstance(img_input, (bytes, dict)):
                blur_keys = ["blur", "blurry", "input", "lowres", "lr", "rain", "hazy", "noisy", "degraded", "distorted", "low", "images"]
                sharp_keys = ["sharp", "gt", "ground_truth", "groundtruth", "clean", "clear", "original", "hr", "highres", "target", "norain", "high", "targets", "mask", "masks", "segmentation", "segmentations"]
                
                p_str = str(img_path).replace("\\", "/")
                parent = img_path.parent
                
                # Strategy A: Sibling Folder (e.g. blur/001.png -> sharp/001.png)
                if any(k in parent.name.lower() for k in blur_keys):
                    try:
                        for sibling in parent.parent.iterdir():
                            if sibling.is_dir() and any(k in sibling.name.lower() for k in sharp_keys):
                                # 1. Exact Match
                                potential = sibling / img_path.name
                                if potential.exists():
                                    target_img_path = str(potential)
                                    break
                                # 2. Fuzzy Prefix Match (e.g. RealBlur style: blur_1.png -> gt_1.png)
                                for b_k in blur_keys:
                                    if img_path.name.lower().startswith(b_k):
                                        for s_k in sharp_keys:
                                            # Try replacing prefix (e.g. blur_ -> gt_)
                                            new_name = img_path.name.lower().replace(b_k, s_k, 1)
                                            p_f = sibling / new_name
                                            if p_f.exists():
                                                target_img_path = str(p_f)
                                                break
                                        if target_img_path: break
                                if target_img_path: break
                    except: pass
                
                # Strategy B: Ancestral Sibling (e.g. train/001.png -> GT/001.png)
                if not target_img_path:
                    for ancestor in [parent, parent.parent]:
                        if any(k in ancestor.name.lower() for k in ["train", "test", "val", "images"]):
                            try:
                                for sibling in ancestor.parent.iterdir():
                                    if sibling.is_dir() and any(k in sibling.name.lower() for k in sharp_keys):
                                        # Try flat filename first
                                        potential = sibling / img_path.name
                                        if potential.exists():
                                            target_img_path = str(potential)
                                            break
                                        # Try relative path preservation
                                        try:
                                            rel = img_path.relative_to(ancestor)
                                            potential_rel = sibling / rel
                                            if potential_rel.exists():
                                                target_img_path = str(potential_rel)
                                                break
                                        except: pass
                            except: pass
                            if target_img_path: break
                
                # Strategy C: Same-Folder Resolution (e.g. rain-001.png -> norain-001.png)
                if not target_img_path and not isinstance(img_input, (bytes, dict)):
                    for b_k in blur_keys:
                        if img_path.name.lower().startswith(b_k):
                            for s_k in sharp_keys:
                                new_name = img_path.name.lower().replace(b_k, s_k, 1)
                                if new_name == img_path.name.lower(): continue
                                p_f = parent / new_name
                                if p_f.exists():
                                    target_img_path = str(p_f)
                                    break
                            if target_img_path: break

            # 3. Legacy DPED Fallback
            if not target_img and not target_img_path and not isinstance(img_input, (bytes, dict)):
                for device_name in ["iphone", "sony", "blackberry"]:
                    needle = f"/{device_name}/"
                    if needle in p_str.lower():
                        tgt_p_str = p_str.lower().replace(needle, "/canon/")
                        if DPED_CACHE and tgt_p_str in DPED_CACHE:
                            target_img_path = tgt_p_str
                        elif os.path.exists(tgt_p_str):
                            target_img_path = tgt_p_str
                        break

        # 2026 High-Velocity Optimization: Defer image loading
        # Only load if we are in a task that requires image stats (vetting/labeling/resizing).
        needs_stats = (task in ["quality", "diffusion"] and not args.no_vetting) or (not skip_labeling)

        if needs_stats:
            if not isinstance(img_input, (bytes, dict)):
                if img_path.suffix.lower() == ".npy":
                    data = np.load(img_path)
                    if data.ndim == 3 and data.shape[0] in [1, 3]: data = data.transpose(1, 2, 0)
                    if data.dtype in [np.float32, np.float64]: data = (data * 255).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(data)
                else:
                    img = Image.open(img_path)
            else:
                img = Image.open(io.BytesIO(img_data)) # type: ignore
            img = ensure_srgb(img)
            w, hgt = img.size
            if task == "quality" and "laion" not in slug and "ava" not in slug:
                if is_black_image(img): return None

            # 2026 High-Fidelity Floor (v16.2.8 Hardened)
            # Never start below the training suite's minimum floor to prevent blur pathologies.
            # Restoration/SR floor at 224px; Diffusion floor at 512px for SOTA parity.
            if task in ["diffusion"]:
                min_dim = 512
            elif task in ["quality", "classification", "restoration", "super-resolution"]:
                min_dim = 128 if "artifact" in slug.lower() else 224
            else:
                min_dim = 128
            
            if w < min_dim or hgt < min_dim:
                return None

        # NIMA Quality Logic: Prioritize Ground Truth over AI Guessing
        nima_score = 1.0
        nima_probs = [0.0] * 10
        nima_probs[0] = 1.0 # default fallback

        # 1. Check for AVA Professional Labels (10-bin)
        if "ava" in slug and AVA_LOOKUP:
            try:
                img_id = int(img_path.stem)
                if img_id in AVA_LOOKUP:
                    votes = AVA_LOOKUP[img_id]
                    nima_probs = [votes[f"vote_{i}"] for i in range(1, 11)]
                    nima_score = sum(p * (i+1) for i, p in enumerate(nima_probs))
            except Exception: pass

        # 2. Check for AADB Human Ratings (Scalar 0-1)
        elif "aadb" in slug and AADB_LOOKUP:
            try:
                raw_score = AADB_LOOKUP.get(img_path.name)
                if raw_score is not None:
                    # Convert 0-1 to 1-10 scale
                    nima_score = (raw_score * 9.0) + 1.0
                    nima_probs = get_gaussian_probs(nima_score)
            except Exception: pass

        # 3. Check for LAION Aesthetic Scores (Scalar 1-10)
        elif "laion" in slug:
            try:
                # Case A: Virtual (ann_data is a dict from row)
                if isinstance(ann_data, dict):
                    val = ann_data.get("aesthetic_score", ann_data.get("score", 6.5))
                    nima_score = float(val) if val is not None else 6.5
                    nima_probs = get_gaussian_probs(nima_score)
                # Case B: Physical (ann_data is (df_subset, mapping))
                elif fmt == "parquet" and ann_data:
                    df_subset, mapping = ann_data
                    col = mapping.get("aesthetic_score", "aesthetic_score")
                    if col in df_subset.columns:
                        nima_score = float(df_subset[col].iloc[0])
                        nima_probs = get_gaussian_probs(nima_score)
            except Exception: pass

        # 4. Check for Universal Technical Scores (Scalar 1-10)
        elif TID_LOOKUP and img_path.name.lower() in TID_LOOKUP:
            try:
                raw_score = TID_LOOKUP.get(img_path.name.lower())
                if raw_score is not None:
                    nima_score = raw_score
                    nima_probs = get_gaussian_probs(nima_score)
            except Exception: pass

        # 4.5. Authenticity Label Override (AI vs Human)
        is_authenticity = "authentic" in prefix.lower()
        if is_authenticity:
            parent_name = img_path.parent.name.lower()
            file_name = img_path.name.lower()
            
            # Prevent root dataset folders from biasing the labels
            if parent_name in ["sut-project", "ai-generated-images-vs-real-images", "real vs fake faces", "raw-sets"]:
                parent_name = ""
                
            path_str = f"{parent_name}/{file_name}".lower()
            
            if any(k in path_str for k in ["sut-project", "midjourney", "diffusion", "ai", "fake", "gan", "generated"]):
                nima_probs = [0.0] * 10
                nima_probs[0] = 1.0
                nima_score = 1.0
            elif any(k in path_str for k in ["ffhq", "div2k", "celebahq", "human", "real", "afhq", "nature"]):
                nima_probs = [0.0] * 10
                nima_probs[9] = 1.0
                nima_score = 10.0

        # 5. AI Vetting Fallback (Only if not in Strict Human mode)
        if nima_probs[0] == 1.0 and task in ["quality", "diffusion"] and not is_authenticity:
            # If we are here, no human ground truth was found.
            # 2026 Strategy: Allow AI fallback for LAION-branded sources if strict mode is disabled
            if CONFIG.get("strict_ground_truth", True):
                # Critical Gate: LAION and other internet-scale sets MUST have labels unless specifically bypassed
                if task == "quality" and "laion" not in slug:
                    return None

            if SENTRY:
                nima_score, nima_probs = SENTRY.score(img, return_probs=True)
                if idx < 10:
                    pass # print(f"[LIVE TRACE] {slug}_{idx:09d} | AI Score: {nima_score:.4f}")

        # 2026 Quality Gate: Enforce higher aesthetic standards for Diffusion manifolds
        current_threshold = 5.5 if task == "diffusion" else CONFIG["nima_threshold"]
        if task in ["quality", "diffusion"] and nima_score < current_threshold and not is_authenticity:
            if idx < 5: print(f"DEBUG: {slug} skipped because nima {nima_score} < {current_threshold}")
            return None

        # Meta Preparation
        hash_target = img_data if isinstance(img_input, (bytes, dict)) else img_path
        h = compute_hash(hash_target) if CONFIG["enable_dedup"] else None

        # Save Output Image & Target
        # 2026 Resilience: Hardlink-First Acceleration (v7.1)
        # Trust PHYSICAL_INDEX to avoid expensive os.path.exists directory lookups on millions of files
        is_already_on_disk = PHYSICAL_INDEX and name.lower() in PHYSICAL_INDEX

        is_clean_only = ("parsenet" in slug.lower() or "codeformer" in slug.lower()) and task == "restoration"
        if is_clean_only and not target_img_path and isinstance(img_input, (str, Path)):
            target_img_path = str(img_path)

        if not is_already_on_disk and not is_clean_only:
            if not img and isinstance(img_input, (str, Path)):
                try:
                    # Attempt Hardlink (Instant, zero I/O)
                    os.link(str(img_path), str(out_img_path))
                except (OSError, AttributeError) as e:
                    try:
                        # Fallback to copy if cross-device or permission denied
                        # 2026: Log cross-device copy only once to avoid flooding
                        shutil.copy2(str(img_path), str(out_img_path))
                    except (shutil.SameFileError, OSError):
                        pass
            elif img:
                save_fmt = "PNG" if ext == ".png" else "JPEG"
                img.save(out_img_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
            elif isinstance(img_input, (bytes, dict)):
                with open(out_img_path, "wb") as f:
                    f.write(img_data)

        if task in ["restoration", "super-resolution", "segmentation"]:
            if target_img_path:
                try:
                    os.link(target_img_path, str(out_tgt_path))
                except (OSError, AttributeError):
                    try: shutil.copy(target_img_path, out_tgt_path)
                    except: pass
            elif target_img:
                save_fmt = "PNG" if ext == ".png" else "JPEG"
                target_img.save(out_tgt_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
            else:
                # Synthetic Mode: Clean image is the target
                # 2026 Warp-Speed: Use hardlink instead of copy for zero-cost synthetic target creation
                if is_clean_only:
                    if isinstance(img_input, (bytes, dict)):
                        with open(out_tgt_path, "wb") as f: f.write(img_data)
                    elif img:
                        save_fmt = "PNG" if ext == ".png" else "JPEG"
                        img.save(out_tgt_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
                else:
                    try:
                        os.link(str(out_img_path), str(out_tgt_path))
                    except (OSError, AttributeError):
                        try: shutil.copy2(out_img_path, out_tgt_path)
                        except: pass

        elif task == "parameter_prediction":
            # 2026: Parameter Prediction datasets store clean source images in targets/
            # The training suite applies on-the-fly degradation during training.
            # Prefer resolved clean target (e.g., DPED Canon) over duplicating the input.
            if target_img_path:
                try:
                    os.link(target_img_path, str(out_tgt_path))
                except (OSError, AttributeError):
                    try: shutil.copy(target_img_path, out_tgt_path)
                    except: pass
            elif target_img:
                save_fmt = "PNG" if ext == ".png" else "JPEG"
                target_img.save(out_tgt_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
            else:
                # No paired clean target found -- input IS the clean source (DIV2K, Flickr2K)
                try:
                    os.link(str(out_img_path), str(out_tgt_path))
                except (OSError, AttributeError):
                    try: shutil.copy2(str(out_img_path), str(out_tgt_path))
                    except: pass

        # Annotations
        annotations = []
        if fmt == "coco" and ann_data is not None:
            for a in ann_data:
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

        elif fmt == "parquet" and ann_data and not isinstance(ann_data, dict):
            df_subset, mapping = ann_data
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
            for entry in ann_data:
                try:
                    cls = map_category(entry["class"], prefix)
                    annotations.append({"type": "bbox", "cls": cls, "data": entry["bbox"]})
                except Exception:
                    pass

        elif fmt == "xml" and ann_data:
            xml_anns = parse_xml(ann_data)
            for a in xml_anns:
                cls = map_category(a["class"], prefix)
                annotations.append({"type": "bbox", "cls": cls, "data": a["bbox"]})

        elif fmt == "yolo" and ann_data:
            yolo_anns = parse_yolo(ann_data, w, hgt)
            for a in yolo_anns:
                if task == "pose":
                    cls = map_category("0", prefix)
                else:
                    cls = map_category(a["class"], prefix)
                if "keypoints" in a and a["keypoints"]:
                    annotations.append({"type": "pose", "cls": cls, "data": a["bbox"] + a["keypoints"]})
                else:
                    annotations.append({"type": "bbox", "cls": cls, "data": a["bbox"]})

        elif fmt == "npz" and ann_data:
            try:
                data = np.load(str(ann_data))
                if "landmarks" in data.files:
                    landmarks = data["landmarks"]
                    if landmarks.shape == (110, 2):
                        # Construct bbox from first 68 points (standard face)
                        pts = landmarks[:68]
                        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
                        # Add 5% padding
                        pad_w = (x_max - x_min) * 0.05
                        pad_h = (y_max - y_min) * 0.05
                        x_min = max(0, x_min - pad_w)
                        y_min = max(0, y_min - pad_h)
                        x_max = min(w, x_max + pad_w)
                        y_max = min(hgt, y_max + pad_h)
                        bbox_w = x_max - x_min
                        bbox_h = y_max - y_min
                        
                        # 5 Keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
                        l_eye = landmarks[36:42].mean(axis=0)
                        r_eye = landmarks[42:48].mean(axis=0)
                        nose = landmarks[30]
                        l_mouth = landmarks[48]
                        r_mouth = landmarks[54]
                        
                        kpts = [
                            l_eye[0], l_eye[1],
                            r_eye[0], r_eye[1],
                            nose[0], nose[1],
                            l_mouth[0], l_mouth[1],
                            r_mouth[0], r_mouth[1]
                        ]
                        
                        cls = map_category("0", prefix)
                        annotations.append({"type": "pose", "cls": cls, "data": [x_min, y_min, bbox_w, bbox_h] + kpts})
            except Exception as e:
                print(f"Error parsing NPZ {ann_data}: {e}")

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
        if not annotations and task not in ["quality", "classification"] and not args.no_labeling and not skip_labeling:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            labeler = get_labeler(task, device)
            annotations = labeler.predict(img)
            if annotations: is_autolabeled = True

        # Write Label File
        # 2026 Optimization: Skip label files for parameter_prediction (labels generated at training time)
        # and skip empty label files for restoration tasks to reduce I/O churn
        label_file_path = Path(output_root_str) / "labels" / split / f"{name}.txt"
        has_annotations = len(annotations) > 0 or task in ["quality", "classification"]

        if not skip_labeling or has_annotations:
            with open(label_file_path, "w", encoding="utf-8") as f:
                if task == "quality":
                    f.write(" ".join(f"{p:.6f}" for p in nima_probs) + "\n")
                elif task == "classification":
                    class_label = 1
                    if isinstance(ann_data, dict) and "label" in ann_data:
                        class_label = ann_data["label"]
                    elif isinstance(ann_data, tuple) and len(ann_data) == 2 and isinstance(ann_data[1], dict):
                        df_subset, mapping = ann_data
                        lbl_col = mapping.get("label", "label")
                        if lbl_col in df_subset.columns:
                            class_label = df_subset.iloc[0][lbl_col]
                    else:
                        p_lower = str(img_path).lower()
                        if any(k in p_lower for k in ['fake', 'ai', 'synthetic', 'nsfw', 'porn', 'explicit']):
                            class_label = 0
                    
                    if isinstance(class_label, float):
                        try:
                            import pandas as pd
                            if not pd.isna(class_label):
                                class_label = int(class_label)
                        except:
                            pass
                        
                    f.write(str(class_label) + "\n")
                elif annotations:
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
        size_bytes = 0
        try:
            if out_img_path.exists(): size_bytes += out_img_path.stat().st_size
        except: pass

        return {
            "name": name, "source": slug, "task": task, "split": split,
            "hash": h, "nima_score": round(nima_score, 3), "is_autolabeled": is_autolabeled,
            "has_segmentation": any(a["type"] == "segmentation" for a in annotations),
            "has_pose": any(a["type"] == "pose" for a in annotations),
            "label_path": str(label_file_path.resolve()), "path": str(out_img_path.resolve()),
            "size": size_bytes,
            "clip_latent": None # Standard tasks don't use clip_latent but registry expects it
        }

    except Exception as e:
        print(f"[ERROR] processing {img_path}: {e}")
        return None

def process_diffusion(
    img_path, prefix, slug, idx, split, output_root_str):
    """Specialized Text-Image processor for Diffusion Models"""
    try:
        # Handle virtual dataset bytes
        if isinstance(img_path, (bytes, dict)):
            img_data = img_path["bytes"] if isinstance(img_path, dict) and "bytes" in img_path else img_path
            if img_data is None: return None
            img = Image.open(io.BytesIO(img_data)) # type: ignore
            is_virtual = True
        else:
            if isinstance(img_path, str): img_path = Path(img_path)
            if not img_path.exists(): return None
            img = Image.open(img_path)
            is_virtual = False

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
            caption_file = img_path.parent / (img_path.stem + ".txt") if not is_virtual else None
            if caption_file and caption_file.exists():
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

        # 2026 Resilience: Pre-serialize for SQLite Bulk Ingestion (v8.1)
        latent_blob = None
        if clip_latent:
            latent_blob = sqlite3.Binary(np.array(clip_latent).astype(np.float32).tobytes())

        return {
            "name": name, "source": slug, "task": "diffusion", "split": split,
            "hash": h, "nima_score": round(nima_score, 3),
            "caption": caption, "style_tag": style_tag, "clip_latent": latent_blob,
            "img_bytes": img_bytes, "size": len(img_bytes)
        }
    except Exception as e:
        safe_path = "virtual_bytes" if isinstance(img_path, (bytes, dict)) else img_path
        print(f"[ERROR] Error processing diffusion sample {safe_path}: {e}")
        return None

def remove_empty_dirs(path):
    for sub in ["images", "labels", "targets", "masks", "shards"]:
        for split in ["train", "val", "test"]:
            p = path / sub / split
            if p.exists() and p.is_dir():
                try: p.rmdir()
                except OSError: pass
        p = path / sub
        if p.exists() and p.is_dir():
            try: p.rmdir()
            except OSError: pass

# ---------------- ORCHESTRATOR ----------------
