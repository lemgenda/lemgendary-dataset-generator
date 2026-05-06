# 2026: Environment Linter Sync (Last Verified: 2026-05-01)
import os
import sys

# 2026 Resilience: Silence Intel MKL/Fortran and Windows SIGINT clutter
if os.name == 'nt':
    os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
    os.environ["FOR_IGNORE_EXCEPTIONS"] = "1"

import json
import random# LemGendary Kaggle Manager (Last Verified: 2026-05-01)
import argparse
import hashlib
import shutil
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import multiprocessing
import io
import sqlite3
import webdataset as wds
from datetime import datetime
import json
import yaml
import hashlib
from sklearn.cluster import MiniBatchKMeans
import time
from datetime import datetime
from tqdm import tqdm
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
    "diffusion_size": 512,
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
parser.add_argument("--no-vetting", action="store_true", help="Disable NIMA quality gate (Pass-Through mode)")
parser.add_argument("--no-labeling", action="store_true", help="Disable YOLO auto-labeling (High-Speed mode)")
args, unknown = parser.parse_known_args()

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
    if "aesthetic" in m_low or not m_low:
        ava_csv = Path("./raw-sets/ava-aesthetic-visual-assessment/ground_truth_dataset.csv")
        if ava_csv.exists():
            import pandas as pd
            df = pd.read_csv(ava_csv)
            vote_cols = [f"vote_{i}" for i in range(1, 11)]
            AVA_LOOKUP = df.set_index("image_num")[vote_cols].to_dict("index")
            print(f"📖 [GT] {len(AVA_LOOKUP)} AVA Aesthetic ratings cached.")
        
        aadb_csv = Path("./raw-sets/aadb-imagedatabase/Dataset.csv")
        if aadb_csv.exists():
            import pandas as pd
            df = pd.read_csv(aadb_csv)
            AADB_LOOKUP = df.set_index("ImageFile")["score"].to_dict()
            print(f"📖 [GT] {len(AADB_LOOKUP)} AADB Aesthetic ratings cached.")

    # 2. Technical Sources (Universal Normalization v7.0)
    if "technical" in m_low or not m_low:
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
        koniq_csv = find_gt_path("koniq10k", "koniq10k_scores.csv")
        if koniq_csv:
            import pandas as pd
            df = pd.read_csv(koniq_csv)
            for _, row in df.iterrows():
                TID_LOOKUP[str(row['image_name']).lower()] = 1.0 + (float(row['MOS']) - 1.0) * 2.25
            print(f"📖 [GT] KonIQ-10k ratings cached.")

        # SPAQ
        spaq_csv = find_gt_path("spaq", "Annotations/MOS_Average.csv")
        if spaq_csv:
            import pandas as pd
            df = pd.read_csv(spaq_csv)
            for _, row in df.iterrows():
                TID_LOOKUP[str(row['Image name']).lower()] = 1.0 + float(row['MOS']) * 0.09
            print(f"📖 [GT] SPAQ ratings cached.")

        # TID2013
        tid_txt = find_gt_path("tid2013", "mos_with_names.txt")
        if tid_txt:
            with open(tid_txt, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        TID_LOOKUP[parts[1].strip().lower()] = float(parts[0]) + 1.0
            print(f"📖 [GT] TID2013 ratings cached.")

        # LIVE IQA
        live_csv = find_gt_path("live", "live_scores.csv")
        if live_csv:
            import pandas as pd
            df = pd.read_csv(live_csv)
            for _, row in df.iterrows():
                orig = min(100.0, float(row['dmos']))
                TID_LOOKUP[str(row['image_name']).lower()] = 1.0 + (1.0 - orig/100.0) * 9.0
            print(f"📖 [GT] LIVE IQA ratings cached.")

        # CSIQ
        csiq_csv = find_gt_path("csiq", "csiq_scores.csv")
        if csiq_csv:
            import pandas as pd
            df = pd.read_csv(csiq_csv)
            for _, row in df.iterrows():
                TID_LOOKUP[str(row['image_name']).lower()] = 1.0 + (1.0 - float(row['dmos'])) * 9.0
            print(f"📖 [GT] CSIQ ratings cached.")

def detect_task(model_dir_name):
    if not model_dir_name: return "quality"
    name = str(model_dir_name).lower()
    if "diffusion" in name: return "diffusion"
    if any(k in name for k in ["seg", "mask", "parsenet"]): return "segmentation"
    if any(k in name for k in ["pose", "face", "codeformer"]): return "pose"
    if any(k in name for k in ["nima", "aesthetic", "quality"]): return "quality"
    if any(k in name for k in ["sr", "ultrazoom", "x2", "x3", "x4", "x8", "super"]): return "super-resolution"
    if any(k in name for k in ["restorer", "enhance", "upn", "lowlight", "exposure", "deraining", "debluring", "denoising", "haze", "restoration", "ffanet", "mirnet", "mprnet", "nafnet"]): 
        return "restoration"
    return "detection"

def init_worker(config):
    global SENTRY, LABELER, CAPTIONER, CLIP_MANIFOLD
    # 2026 Modular Alignment: Local imports from encapsulated modules
    from models.quality_scorer import QualitySentry
    from models.detection import AutoLabeler
    from models.diffusion import CaptionSentry
    from models.encoder import CLIPManifold

    # 2026 Resilience: Workers ignore SIGINT. 
    # NOTE: Re-enabling stderr for SOTA v5.9.2 diagnostics
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # sys.stderr = open(os.devnull, 'w') 
    import torch
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # CRITICAL: Prevent multiprocessing thread thrashing on CPU
    torch.set_num_threads(1)
    
    # 2026 Resilience: Multi-processing with PyTorch on Windows CUDA causes severe deadlocks 
    # and OOMs if multiple workers allocate GPU memory concurrently on a single 4GB card.
    # We FORCE the SENTRY to CPU. It is extremely fast (MobileNetV2) and prevents deadlocks.
    device = "cpu"
    
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
    from models.detection import AutoLabeler
    global LABELER
    if LABELER is None: LABELER = {}
    if task not in LABELER:
        mode = "segmentation" if "seg" in task else "detection"
        LABELER[task] = AutoLabeler(mode=mode, device=device)
    return LABELER[task]

# ---------------- HELPERS ----------------
def ensure_srgb(img):
    if img.mode != "RGB":
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

# ---------------- REGISTRY ----------------
def initialize_registry(db_path):
    conn = sqlite3.connect(db_path, timeout=60.0) # Increased timeout for heavy IPC contention
    # 2026 Ultra-Optimization: Exclusive locking for maximum RAID throughput
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF") # Maximum throughput
    conn.execute("PRAGMA cache_size=100000")
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
    if "laion-5b" in sl: return "laion-5b"
    if "laion" in sl: return "laion"
    if "ava" in sl: return "ava"
    if "aadb" in sl: return "aadb"
    if "ffhq" in sl: return "ffhq"
    if "coco" in sl: return "coco"
    if "flickr" in sl: return "flickr"
    return slug.split('-')[0]

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

# ---------------- BATCH WORKER ----------------
def batch_worker(tasks):
    """Executes a list of tasks in a single worker call to reduce IPC overhead."""
    results = []
    for task_func, *args in tasks:
        try:
            results.append(task_func(*args))
        except Exception as e:
            results.append(None)
    return results

# ---------------- PROCESSORS ----------------
def process_image(img_input, prefix, slug, idx, task, fmt, ann_data, split, output_root_str, skip_labeling=False):
    """
    Worker function for parallel processing.
    img_input can be a Path or raw bytes (for Parquet-embedded datasets).
    """
    img_path = "Unknown"
    try:
        # Validity & Format Handling
        if isinstance(img_input, (bytes, dict)):
            if isinstance(img_input, dict) and "bytes" in img_input:
                img_data = img_input["bytes"]
            else:
                img_data = img_input
            
            if img_data is None: return None
            
            img = Image.open(io.BytesIO(img_data))
            # 2026 Resilience: Use a Path object even for virtual, but with a safe Windows-friendly name
            img_path = Path(f"virtual_{slug}_{idx:09d}.jpg")
            is_st = False
        else:
            img_path = img_input
            if not img_path.exists(): return None
            is_st = img_path.suffix.lower() == ".safetensors"
            
        ext = img_path.suffix.lower() if isinstance(img_input, (str, Path)) else ".jpg"
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]: ext = ".jpg"
        
        name = f"{prefix}_{slug}_{idx:09d}"
        out_img_path = Path(output_root_str) / "images" / split / f"{name}{ext}"
        out_tgt_path = Path(output_root_str) / "targets" / split / f"{name}{ext}"
        
        # Restoration Target Resolver (v5.6)
        target_img = None
        if task in ["restoration", "super-resolution"]:
            if "dped" in slug.lower() and not isinstance(img_input, (bytes, dict)):
                # DPED Path Mirroring: iphone2canon/test/iphone/1.jpg -> iphone2canon/test/canon/1.jpg
                p_str = str(img_path).replace("\\", "/")
                if "/iphone/" in p_str: 
                    tgt_p_str = p_str.replace("/iphone/", "/canon/")
                    if os.path.exists(tgt_p_str):
                        target_img = Image.open(tgt_p_str)
                elif "/blackberry/" in p_str:
                    tgt_p_str = p_str.replace("/blackberry/", "/canon/")
                    if os.path.exists(tgt_p_str):
                        target_img = Image.open(tgt_p_str)
                elif "/sony/" in p_str:
                    tgt_p_str = p_str.replace("/sony/", "/canon/")
                    if os.path.exists(tgt_p_str):
                        target_img = Image.open(tgt_p_str)
            
            # If no specific paired target found, the clean image is the target (Synthetic Mode)
            if target_img is None:
                # We'll set a flag to save the clean 'img' as target later
                pass
        
        # Resumption Check
        if out_img_path.exists():
            pass

        if not isinstance(img_input, (bytes, dict)):
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
        if not is_st and task == "quality" and "laion" not in slug and "ava" not in slug:
            if is_black_image(img): return None
        
        w, hgt = img.size
        # For restoration and diffusion, we allow smaller patches (e.g. DPED 100x100)
        min_dim = 64
        if w < min_dim or hgt < min_dim: 
            if idx < 5: print(f"DEBUG: {slug} skipped because size {w}x{hgt} < {min_dim}")
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
                    nima_score = float(ann_data.get("aesthetic_score", ann_data.get("score", 6.5)))
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

        # 5. AI Vetting Fallback (Only if not in Strict Human mode)
        if nima_probs[0] == 1.0 and task in ["quality", "diffusion"]:
            # If we are here, no human ground truth was found.
            # 2026 Strategy: Allow AI fallback for LAION-branded sources if strict mode is disabled
            if CONFIG.get("strict_ground_truth", True):
                # Critical Gate: LAION and other internet-scale sets MUST have labels unless specifically bypassed
                if task == "quality" and "laion" not in slug:
                    return None
                
            if SENTRY:
                nima_score, nima_probs = SENTRY.score(img, return_probs=True)
                if idx < 10:
                    pass # print(f"🔬 [LIVE TRACE] {slug}_{idx:09d} | AI Score: {nima_score:.4f}")
        
        # 2026 Quality Gate: Enforce higher aesthetic standards for Diffusion manifolds
        current_threshold = 5.5 if task == "diffusion" else CONFIG["nima_threshold"]
        if task in ["quality", "diffusion"] and nima_score < current_threshold: 
            if idx < 5: print(f"DEBUG: {slug} skipped because nima {nima_score} < {current_threshold}")
            return None

        # Meta Preparation
        hash_target = img_data if isinstance(img_input, (bytes, dict)) else img_path
        h = compute_hash(hash_target) if CONFIG["enable_dedup"] else None
        
        # Save Output Image & Target
        if not out_img_path.exists():
            # 2026 Optimization: Skip re-encode for restoration JPEGs to push > 100 it/s
            needs_reencode = getattr(img, "was_converted", False) or (is_st and task not in ["restoration", "super-resolution"])
            if not needs_reencode and isinstance(img_input, (str, Path)) and ext in [".jpg", ".jpeg", ".png", ".webp"]:
                import shutil
                shutil.copy2(img_path, out_img_path)
            elif not needs_reencode and isinstance(img_input, (bytes, dict)) and getattr(img, "format", "") in ["JPEG", "PNG", "WEBP"]:
                with open(out_img_path, "wb") as f:
                    f.write(img_data)
            else:
                save_fmt = "PNG" if ext == ".png" else "JPEG"
                img.save(out_img_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
        
        if task in ["restoration", "super-resolution"] and not out_tgt_path.exists():
            if target_img:
                target_img = ensure_srgb(target_img)
                save_fmt = "PNG" if ext == ".png" else "JPEG"
                target_img.save(out_tgt_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
            else:
                # Synthetic Mode: Clean image is the target
                if out_img_path.exists() and not getattr(img, "was_converted", False):
                    import shutil
                    shutil.copy2(out_img_path, out_tgt_path)
                else:
                    save_fmt = "PNG" if ext == ".png" else "JPEG"
                    img.save(out_tgt_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
        
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
        label_file_path = Path(output_root_str) / "labels" / split / f"{name}.txt"
        with open(label_file_path, "w") as f:
            if task == "quality":
                f.write(" ".join(f"{p:.6f}" for p in nima_probs) + "\n")
            elif task == "classification":
                # AI = 0, Human/Real = 1
                p_lower = str(img_path).lower()
                class_label = 0 if any(k in p_lower for k in ['fake', 'ai', 'synthetic']) else 1
                f.write(str(class_label) + "\n")
            else:
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
        if out_img_path.exists(): size_bytes += out_img_path.stat().st_size
        if label_file_path.exists(): size_bytes += label_file_path.stat().st_size
        
        return {
            "name": name, "source": slug, "task": task, "split": split,
            "hash": h, "nima_score": round(nima_score, 3), "is_autolabeled": is_autolabeled,
            "has_segmentation": any(a["type"] == "segmentation" for a in annotations),
            "has_pose": any(a["type"] == "pose" for a in annotations),
            "label_path": str(label_file_path.resolve()), "path": str(out_img_path.resolve()),
            "size": size_bytes
        }

    except Exception as e:
        print(f"[ERROR] processing {img_path}: {e}")
        return None

def process_diffusion(img_path, prefix, slug, idx, split, output_root_str):
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
            "name": name, "source": slug, "task": "diffusion", "split": split,
            "hash": h, "nima_score": round(nima_score, 3), 
            "caption": caption, "style_tag": style_tag, "clip_latent": clip_latent,
            "img_bytes": img_bytes, "size": len(img_bytes)
        }
    except Exception as e:
        print(f"❌ Error processing diffusion sample {img_path}: {e}")
        return None

# ---------------- ORCHESTRATOR ----------------
def process_dataset():
    min_gb = META.get("global_constraints", {}).get("min_size_gb", 0.1)
    max_gb = args.max_gb if args.max_gb is not None else META.get("global_constraints", {}).get("max_size_gb", 50.0)
    prefix_str = META.get("name_prefix", "")
    suffix_str = args.suffix if args.suffix is not None else META.get("name_suffix", "")

    shared_root = INPUT_ROOT
    # Pre-load models globally once to prevent multiprocess race conditions on HF cache
    print("🛡️ [PRE-FLIGHT] Analyzing task requirements...")
    from models.quality_scorer import QualitySentry
    from models.diffusion import CaptionSentry
    from models.encoder import CLIPManifold
    from models.detection import AutoLabeler
    
    # Analyze if any target models need AI augmentation
    needs_captioning = False
    needs_styling = False
    
    for model_key in DATASETS_META:
        if args.model and model_key != args.model: continue
        task = detect_task(model_key)
        if task == "diffusion": needs_captioning = True
        if task == "diffusion" or model_key == "nima_aesthetic": needs_styling = True # Styling optional for aesthetic

    if needs_captioning and not args.no_vetting:
        print("🛡️ [PRE-FLIGHT] Caching CaptionSentry (BLIP)...")
        tmp = CaptionSentry(device="cpu")
        del tmp
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    if needs_styling and not args.no_vetting:
        # User requested clean dataset suite; we'll disable CLIP unless strictly needed
        # For NimaAesthetic, we'll only load if not in a "Pure Sharding" mindset.
        # Given user feedback, we default to skipping unless it's a diffusion manifold.
        if needs_captioning: 
            print("🛡️ [PRE-FLIGHT] Caching CLIPManifold...")
            _ = CLIPManifold(device="cpu")
    
    print("🛡️ [PRE-FLIGHT] Pre-flight analysis complete.")

    # 2026 Resilience: Adaptive worker scaling (v5.1)
    # Priority: 1. CLI Args (--workers) | 2. config.json | 3. Auto-detected (CPU-2)
    final_workers = args.workers if args.workers else CONFIG.get("num_workers", 4)
    
    # Only apply safety cap if the user didn't explicitly request a worker count
    if not args.workers and final_workers > 8:
        print(f"🛡️ [RESILIENCE] Capping auto-detected workers to 8 for stability. Use --workers to override.")
        final_workers = 8
    
    max_workers = max(1, final_workers)
    print(f"🛡️ [PRE-FLIGHT] Hardware: {get_device_info()} | Active Workers: {max_workers}", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(CONFIG,)) as executor:
        for model_key, model_config in DATASETS_META.items():
            if args.model and model_key != args.model: continue
            
            task = detect_task(model_key)
            
            pascal_name = model_config.get("name", model_key.replace("_", ""))
            prefix = pascal_name
            
            output_root = OUT_PARENT / f"{prefix_str}{pascal_name}{suffix_str}"
            output_root_str = str(output_root)
            
            if not output_root.exists():
                for d in ["images", "labels", "targets"]:
                    for s in ["train", "val"]: (output_root / d / s).mkdir(parents=True, exist_ok=True)

            print(f"\n🚀 [SOTA v5.0] Commencing compilation for {pascal_name} -> {output_root.name}...")

            index = []
            seen_hashes = set()
            
            db_path = output_root / "manifold_registry.db"
            conn = initialize_registry(db_path)

            # RESUMPTION LOGIC: Load existing entries from SQLite to bypass already processed samples
            existing_names = set()
            if db_path.exists():
                print(f"🔄 [RESUMPTION] Scanning {pascal_name} registry for existing entries...")
                try:
                    rows = conn.execute("SELECT name FROM registry").fetchall()
                    existing_names = {r[0] for r in rows}
                    if existing_names:
                        print(f"✅ Found {len(existing_names)} existing samples. Resuming from checkpoint.")
                except Exception as e:
                    print(f"⚠️ Resumption scan failed: {e}")

            sfw_tasks = []
            nsfw_tasks = []
            
            for ref_entry in model_config.get("refs", []):
                ref = ref_entry["ref"]
                tag = ref_entry.get("tag", "sfw")
                # Resolve Slug: Handle hf://, gh://, and kaggle:// prefixes
                slug = ref.replace('hf://', '').replace('gh://', '').replace('kaggle://', '').split('/')[-1]
                if ":" in slug:
                    slug = slug.split(":")[-1].replace(".tgz", "").replace(".tar.gz", "").replace(".zip", "")
                
                # High-Resilience Discovery: Check for case-insensitive matches and nested folders
                dataset = shared_root / slug
                if not dataset.is_dir():
                    # Check for lowercase version
                    dataset = shared_root / slug.lower()
                    if not dataset.is_dir():
                        # Last resort: Try to find a folder that contains the slug in its name
                        try:
                            matches = [d for d in shared_root.iterdir() if d.is_dir() and slug.lower() in d.name.lower()]
                            if matches:
                                dataset = matches[0]
                                print(f"🔍 [DISCOVERY] Mapping {ref} -> {dataset.name}")
                        except:
                            pass
                
                if not dataset.is_dir():
                    print(f"⚠️ [SKIP] Source {ref} not found in {shared_root}")
                    continue

                fmt, ann_path = detect_annotations(dataset)
                ann_data = None
                if fmt == "coco":
                    ann_data = parse_coco(ann_path)
                elif fmt == "parquet":
                    # 2026 Resilience: Handle multiple shards
                    ann_paths = list(dataset.rglob("*.parquet"))
                    ann_data_list = []
                    for ap in ann_paths:
                        try:
                            ann_data_list.append(parse_parquet(ap))
                        except: pass
                    ann_data = ann_data_list[0] if ann_data_list else None
                elif fmt == "matlab":
                    ann_data = parse_matlab(ann_path)

                valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".safetensors", ".tiff", ".tif", ".bmp"}
                images = []
                for root, _, files in os.walk(dataset):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in valid_exts:
                            images.append(Path(root) / f)
                
                
                # VIRTUAL DATASET SUPPORT: If no loose images, check if Parquet has embedded images
                is_virtual = False
                if not images and fmt == "parquet" and ann_data_list:
                    # Check ALL shards for "image" column, not just the first one
                    for df_shard, _ in ann_data_list:
                        if "image" in df_shard.columns:
                            is_virtual = True
                            print(f"✨ [VIRTUAL] {slug} identified as Sharded Parquet dataset ({len(ann_data_list)} shards).")
                            break
                
                # PRE-COMPUTE ANNOTATION LOOKUPS TO AVOID O(N^2) BOTTLENECKS
                coco_file_to_id = {}
                parquet_map = {}
                matlab_map = {}

                if fmt == "coco" and ann_data:
                    images_meta, anns_meta = ann_data
                    for k, v in images_meta.items():
                        coco_file_to_id[v["file_name"]] = k
                elif fmt == "parquet" and ann_data and not is_virtual:
                    df, mapping = ann_data
                    file_col = mapping.get("file_name", "file_name")
                    # Only group if the column is hashable (e.g. filename strings)
                    if file_col in df.columns and len(df) > 0 and (df[file_col].dtype != 'object' or isinstance(df[file_col].iloc[0], str)):
                        for fname, group in df.groupby(file_col):
                            parquet_map[fname] = group
                elif fmt == "matlab" and ann_data:
                    data, key = ann_data
                    if key in data:
                        for entry in data[key]:
                            try:
                                fname = entry.get("image_name", entry.get("name"))
                                if fname:
                                    if fname not in matlab_map: matlab_map[fname] = []
                                    matlab_map[fname].append(entry)
                            except Exception:
                                pass

                sample_count = len(images) if not is_virtual else sum(len(d[0]) for d in ann_data_list)
                # print(f"[QUEUE] {prefix} ({task}) | {slug} | {sample_count} samples scheduled.")
                
                model_val_split = model_config.get("val_split", None)
                if model_val_split is not None:
                    train_prob = 1.0 - float(model_val_split)
                elif task == "diffusion" or "image_to_text" in model_key:
                    train_prob = 1.0
                else:
                    train_prob = CONFIG["train_split"]
                
                if is_virtual:
                    # Case A: Queue tasks directly from all Parquet shards
                    global_idx = 0
                    for df, mapping in ann_data_list:
                        # We use itertuples for speed, but we must handle the row data carefully
                        for row in df.itertuples():
                            img_bytes = getattr(row, "image", None)
                            if img_bytes is None: continue
                            
                            split = "train" if random.random() < train_prob else "val"
                            name = f"{prefix}_{clean_slug(slug)}_{global_idx:09d}"
                            
                            if name in existing_names: 
                                global_idx += 1
                                continue
                            
                            # Convert row to dict for easier access in worker
                            row_dict = row._asdict()
                            skip_lbl = not model_config.get("labeling", True)
                            if task == "diffusion":
                                task_item = (process_diffusion, img_bytes, prefix, clean_slug(slug), global_idx, split, output_root_str)
                            else:
                                task_item = (process_image, img_bytes, prefix, clean_slug(slug), global_idx, task, fmt, row_dict, split, output_root_str, skip_lbl)
                            
                            if tag == "nsfw":
                                nsfw_tasks.append(task_item)
                            else:
                                sfw_tasks.append(task_item)
                            
                            global_idx += 1
                        
                else:
                    # Case B: Standard Physical File Loop
                    for i, img_path in enumerate(images):
                        split = "train" if random.random() < train_prob else "val"
                        
                        name = f"{prefix}_{clean_slug(slug)}_{i:09d}"
                        if name in existing_names:
                            continue

                        specific_ann_data = None
                        if fmt == "coco" and ann_data:
                            images_meta, anns_meta = ann_data
                            img_id = coco_file_to_id.get(img_path.name)
                            if img_id is not None:
                                specific_ann_data = anns_meta.get(img_id, [])
                        elif fmt == "parquet" and ann_data:
                            df, mapping = ann_data
                            df_subset = parquet_map.get(img_path.name)
                            if df_subset is not None and not df_subset.empty:
                                specific_ann_data = (df_subset, mapping)
                        elif fmt == "matlab" and ann_data:
                            specific_ann_data = matlab_map.get(img_path.name, [])
                        elif fmt == "safetensors" and ann_data:
                            specific_ann_data = ann_data

                        skip_lbl = not model_config.get("labeling", True)
                        if task == "diffusion":
                            task_item = (process_diffusion, img_path, prefix, clean_slug(slug), i, split, output_root_str)
                        else:
                            task_item = (process_image, img_path, prefix, clean_slug(slug), i, task, fmt, specific_ann_data, split, output_root_str, skip_lbl)
                            
                        if tag == "nsfw":
                            nsfw_tasks.append(task_item)
                        else:
                            sfw_tasks.append(task_item)
                    
                    print(f"   -> [{slug}] Discovered {sample_count} source tensors.")

            # 2026 Strategy: Dynamic Ratio Balancing (v5.8)
            target_nsfw_ratio = model_config.get("nsfw_ratio", 0)
            if target_nsfw_ratio > 0 and nsfw_tasks:
                max_nsfw = int(len(sfw_tasks) * target_nsfw_ratio / (1.0 - target_nsfw_ratio))
                if len(nsfw_tasks) > max_nsfw:
                    print(f"⚖️  [BALANCING] NSFW pool ({len(nsfw_tasks)}) exceeds {target_nsfw_ratio*100}% cap. Capping at {max_nsfw} samples.")
                    random.shuffle(nsfw_tasks)
                    nsfw_tasks = nsfw_tasks[:max_nsfw]
            
            all_tasks = sfw_tasks + nsfw_tasks
            random.shuffle(all_tasks)
            
            if not all_tasks:
                print(f"⚠️  [NOTICE] No tasks found for {pascal_name}. Manifold is either fully processed or empty.")
                continue

            compiled_bytes = 0
            processed_count = len(existing_names)
            # CPU Resilience: Auto-bypass if CUDA is missing and dataset is massive
            if not torch.cuda.is_available() and len(all_tasks) > 50000:
                if not args.no_labeling or not args.no_vetting:
                    print(f"⚠️ [CPU-GUARD] Massive dataset ({len(all_tasks)} items) on CPU. Auto-enabling High-Speed Mode.", flush=True)
                    args.no_labeling = True
                    args.no_vetting = True

            from concurrent.futures import wait, FIRST_COMPLETED
            desc_label = "[PASS 1] Extraction & Vetting" if not args.no_vetting and task in ["quality", "classification"] else "[PASS 1] Extraction & Processing"
            
            # 2026 Optimization: Batching to reduce IPC overhead
            BATCH_SIZE = 100
            task_batches = [all_tasks[i:i + BATCH_SIZE] for i in range(0, len(all_tasks), BATCH_SIZE)]
            
            with tqdm(total=len(all_tasks) + len(existing_names), initial=len(existing_names), desc=desc_label) as pbar:
                futures = set()
                batch_iter = iter(task_batches)
                
                # Top up initial futures (Higher buffer for 12 workers)
                for _ in range(min(max_workers * 2, len(task_batches))):
                    batch = next(batch_iter)
                    futures.add(executor.submit(batch_worker, batch))
                
                try:
                    while futures:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            try:
                                batch_results = future.result()
                                # 2026 Resilience: Update progress bar for the WHOLE batch immediately
                                pbar.update(len(batch_results))
                                
                                for res in batch_results:
                                    if res:
                                        if CONFIG["enable_dedup"] and res["hash"] in seen_hashes: 
                                            continue
                                        seen_hashes.add(res["hash"])
                                        
                                        compiled_bytes += res.get("size", 0)
                                        
                                        # Commit to SQLite Manifold
                                        latent_blob = sqlite3.Binary(np.array(res.get("clip_latent", [])).astype(np.float32).tobytes())
                                        conn.execute("""
                                            INSERT INTO registry (name, source, task, split, hash, nima_score, caption, style_tag, clip_latent, img_bytes)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """, (res["name"], res["source"], res["task"], res["split"], res["hash"], res["nima_score"], 
                                              res.get("caption"), res.get("style_tag"), latent_blob, res.get("img_bytes")))
                                        
                                        processed_count += 1
                                        if processed_count % 1000 == 0:
                                            conn.commit()
        
                                        if (compiled_bytes / (1024**3)) >= max_gb:
                                            print(f"\n⚠️  [MANIFOLD LIMIT REACHED] Compiled set reached {max_gb:.2f}GB. Halting extraction.")
                                            for f in futures: f.cancel()
                                            futures.clear()
                                            break
                            except Exception as e:
                                # Write to physical log for SOTA diagnostics (UTF-8)
                                with open("worker_error.log", "a", encoding='utf-8') as f:
                                    f.write(f"❌ Worker Error at {datetime.now()}: {str(e)}\n")
                                print(f"[ERROR] Worker Error (Logged): {e}")
                        
                        # 2026 Resilience: Frequent Commits (v5.5)
                        if pbar.n % 1000 == 0:
                            conn.commit()
                        
                        # Top up the queue with more batches
                        for _ in range(len(done)):
                            try:
                                batch = next(batch_iter)
                                futures.add(executor.submit(batch_worker, batch))
                            except StopIteration:
                                break
                except KeyboardInterrupt:
                    print("\n🛑 [INTERRUPT] Mission aborted by user. Performing emergency shutdown...")
                    for f in futures: f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    # Force exit to prevent Windows threading lock
                    os._exit(1)

            conn.commit()
            
            compiled_gb = compiled_bytes / (1024**3)
            if compiled_gb < min_gb:
                print(f"⚠️  [WARNING] Compiled set size ({compiled_gb:.2f}GB) is below the minimum manifold constraint ({min_gb:.2f}GB).")

            # STEP 2: Style Clustering (v5.0 Global Manifold)
            print(f"[STYLING] Commencing Style Clustering on all extracted latents...")
            cursor = conn.execute("SELECT id, clip_latent FROM registry WHERE clip_latent IS NOT NULL")
            ids, latents = [], []
            for row in cursor:
                _lat = np.frombuffer(row[1], dtype=np.float32)
                if len(_lat) > 0:
                    ids.append(row[0])
                    latents.append(_lat)
            
            if latents and len(latents) > 0 and len(latents[0]) > 0:
                X = np.stack(latents)
                n_clusters = CONFIG.get("n_style_clusters", 16)
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X)
                labels = kmeans.labels_
                for i, cid in tqdm(zip(ids, labels), total=len(ids), desc="[STYLING] Updating Clusters"):
                    conn.execute("UPDATE registry SET cluster_id = ? WHERE id = ?", (int(cid), i))
                conn.commit()
            else:
                print(f"ℹ️  [STYLING] No valid style latents found. Skipping clustering (Pure Human Mode).")

            # PASS 2: Balanced Interleaving & Sharding per Dataset (as requested)
            print(f"[SHARD] Commencing PASS 2: Multi-Domain Balanced Sharding...")
            shard_dir = output_root / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            
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
            
            with open(output_root / "index.json", "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            
            generate_metadata_files(output_root, index, pascal_name)
            generate_readme(output_root)
            generate_kaggle_notebook(output_root, pascal_name, model_key)
            print(f"[SUCCESS] v5.0 Ascension Complete: {len(index)} samples compiled for {pascal_name}.")

# ---------------- GENERATORS ----------------
def generate_metadata_files(output_root, index, prefix):
    if not index: return
    sources = sorted(list(set(x["source"] for x in index)))
    task = index[0]["task"]
    
    yaml_content = f"""count: {len(index)}
task: {task}
original_sources:
{chr(10).join(f"- {s}" for s in sources)}
path: {str(output_root.resolve())}
source: {prefix}-manifold
last_processed: '{datetime.now().isoformat()}'
"""
    with open(output_root / "dataset_info.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # category.txt
    with open(output_root / "category.txt", "w", encoding="utf-8") as f:
        f.write("Image Quality Assessment\n" if task == "quality" else "Object Detection\n")

    # classes.txt
    with open(output_root / "classes.txt", "w", encoding="utf-8") as f:
        f.write(f"{task}\n")

def generate_readme(output_root):
    if not (output_root / "index.json").exists(): return
    data = json.load(open(output_root / "index.json"))
    
    train = [x for x in data if x["split"] == "train"]
    val = [x for x in data if x["split"] == "val"]
    
    def format_source(name):
        lower_name = name.lower()
        if lower_name == 'celebamask': return 'CelebAMask'
        elif lower_name == 'affectnet': return 'AffectNet'
        elif lower_name == 'wflw': return 'WFLW'
        elif lower_name == 'ffhq': return 'FFHQ'
        elif lower_name == 'helen': return 'Helen'
        elif lower_name in ['ava', 'aadb', 'coco', 'csiq', 'spaq', 'live']: return lower_name.upper()
        elif lower_name == 'koniq10k': return 'KonIQ-10k'
        elif lower_name == 'tid2013': return 'TID2013'
        elif lower_name == 'laion': return 'LAION'
        elif lower_name == 'laion-5b': return 'LAION-5B'
        else: return name.title()

    tasks = {}
    sources = {}
    for item in data:
        tasks[item["task"]] = tasks.get(item["task"], 0) + 1
        
        # Extract true source exactly as provided during discovery
        actual_src = item.get("source", "")
        if not actual_src:
            name_parts = item["name"].split("_")
            if len(name_parts) >= 3:
                actual_src = "_".join(name_parts[1:-1])
            else:
                actual_src = "Unknown"
            
        src = format_source(actual_src)
        if src not in sources:
            sources[src] = {"train": 0, "val": 0, "total": 0}
            
        sources[src]["total"] += 1
        split = item.get("split", "unknown")
        if split in ["train", "val"]:
            sources[src][split] += 1
    
    task_type = data[0]["task"] if data else "quality"
    
    # Task Metadata Mapping for Industrial Formatting
    TASK_META = {
        "quality": {
            "category": "Image Quality Assessment",
            "desc": "Unified dataset for training SOTA quality models and evaluation systems.",
            "obj": "Predict human-perceptual quality score.",
            "models": "NIMA_Model, MultiTaskRestorer",
            "arch": "MobileNetV2 / ResNet backbone with regression head",
            "loss": "Earth Mover's Distance (EMD) Loss, L1/Huber Loss",
            "metrics": "| **PLCC** | ~0.8500 | > 0.9000 | **> 0.9500** |\n| **SRCC** | ~0.8000 | > 0.8500 | **> 0.9000** |",
            "targets": "EXPLICITLY OMITTED",
            "targets_desc": "Because this specific task organically maps abstracted analytical endpoints sequentially tracking internal topologies natively inside `labels/` (like probability vectors), generating physical Image-To-Image targets natively is structurally redundant and safely decoupled."
        },
        "restoration": {
            "category": "Image Restoration",
            "desc": "Standardized dataset for image restoration, denoising, and enhancement models.",
            "obj": "Restore degraded images and enhance visual quality.",
            "models": "NafNet, MirNet, MprNet, FfaNet, MultiTaskRestorer",
            "arch": "UNet-based restoration architectures with residual learning",
            "loss": "L1 Loss, SSIM Loss, Charbonnier Loss",
            "metrics": "| **PSNR** | ~28.0 dB | > 31.0 dB | **> 33.0 dB** |\n| **SSIM** | ~0.8000  | > 0.8800  | **> 0.9200**  |\n| **LPIPS**| ~0.1500  | < 0.1200  | **< 0.0800**  |\n| **FID**  | ~15.00   | < 12.00   | **< 8.00**    |",
            "targets": "ACTIVELY DEPLOYED",
            "targets_desc": "Because this dataset natively inherently evaluates pixel-to-pixel structural auto-encoder derivations, the `targets/` folder physically houses the uncorrupted High-Resolution matrices strictly mapped identically 1-to-1 against `images/`."
        },
        "super-resolution": {
            "category": "Super-Resolution",
            "desc": "High-fidelity dataset for image super-resolution and ultra-zoom models.",
            "obj": "Scale low-resolution images to high-resolution while preserving details.",
            "models": "SwinIR, HAT, EDSR, RRDBNet",
            "arch": "Transformer-based or Deep Residual networks",
            "loss": "L1 Loss, VGG Perceptual Loss, GAN Loss",
            "metrics": "| **PSNR** | ~29.0 dB | > 32.0 dB | **> 34.0 dB** |\n| **SSIM** | ~0.8200  | > 0.8900  | **> 0.9300**  |",
            "targets": "ACTIVELY DEPLOYED",
            "targets_desc": "The `targets/` folder houses the original high-resolution ground truth images strictly mapped identically to the downscaled counterparts in `images/`."
        },
        "diffusion": {
            "category": "Generative Modeling",
            "desc": "Massive manifold for training Latent Diffusion Models and text-to-image systems.",
            "obj": "Generate high-fidelity images from text prompts or latent seeds.",
            "models": "Stable Diffusion, SDXL, DiT, FLUX",
            "arch": "U-Net / Transformer with Cross-Attention mechanisms",
            "loss": "MSE (Latent) Loss, VLB Loss",
            "metrics": "| **FID** | ~15.00 | < 10.00 | **< 6.00** |\n| **CLIP-Score** | ~25.0 | > 28.0 | **> 31.0** |",
            "targets": "EXPLICITLY OMITTED",
            "targets_desc": "Diffusion manifolds utilize latent space representations and textual captions. Data is natively stored in compressed `.parquet` blocks natively bypassing `images/` architectures."
        },
        "vlm": {
            "category": "Vision-Language Modeling",
            "desc": "Multimodal conversational dataset for training Large Vision-Language Models (VLMs).",
            "obj": "Align visual token extraction with causal language generation.",
            "models": "LLaVA, BLIP-2, Qwen-VL",
            "arch": "ViT Vision Encoder + Causal LLM Decoder",
            "loss": "Cross-Entropy (Next-Token Prediction)",
            "metrics": "| **ROUGE-L** | ~0.40 | > 0.60 | **> 0.80** |\n| **CIDEr** | ~0.8 | > 1.2 | **> 1.5** |",
            "targets": "EXPLICITLY OMITTED",
            "targets_desc": "VLM manifolds utilize structured multi-turn conversation arrays alongside `image_bytes`. Data is natively stored in compressed `.parquet` blocks."
        },
        "detection": {
            "category": "Object Detection",
            "desc": "Industrial dataset for object detection and localization.",
            "obj": "Detect and localize multiple object classes with high precision.",
            "models": "YOLOv8, YOLOv10, RT-DETR",
            "arch": "CSP-Darknet / Transformer backbones with Path Aggregation",
            "loss": "CIoU Loss, DFL Loss, Binary Cross-Entropy",
            "metrics": "| **mAP@50** | ~0.500 | > 0.700 | **> 0.850** |\n| **mAP@50-95** | ~0.350 | > 0.500 | **> 0.650** |",
            "targets": "EXPLICITLY OMITTED",
            "targets_desc": "Detection tasks map bounding boxes and classification labels stored in `labels/`; physical image targets are not required for this modality."
        },
        "segmentation": {
            "category": "Image Segmentation",
            "desc": "Detailed dataset for semantic and instance segmentation.",
            "obj": "Assign categorical labels to every pixel in the image manifold.",
            "models": "SAM, Mask2Former, SegFormer",
            "arch": "Vision Transformer (ViT) backbones with hierarchical decoders",
            "loss": "Dice Loss, Focal Loss, Cross-Entropy",
            "metrics": "| **mIoU** | ~0.650 | > 0.750 | **> 0.850** |",
            "targets": "ACTIVELY DEPLOYED",
            "targets_desc": "The `targets/` folder contains pixel-perfect segmentation masks or parsed category maps identically mapped to `images/`."
        },
        "pose": {
            "category": "Pose Estimation / Face Landmarks",
            "desc": "High-fidelity dataset for human pose estimation and dense face landmarking.",
            "obj": "Regress exact coordinate points for biological landmarks.",
            "models": "CodeFormer, RTMPose, MediaPipe",
            "arch": "High-Resolution Net (HRNet) or ViT backbones",
            "loss": "MSE Loss, Wing Loss, OKS Loss",
            "metrics": "| **mAP** | ~0.600 | > 0.750 | **> 0.880** |",
            "targets": "EXPLICITLY OMITTED",
            "targets_desc": "Landmark coordinates are stored as normalized vectors within the `labels/` directory, removing the need for physical target bitmaps."
        }
    }

    m = TASK_META.get(task_type, TASK_META["detection"])
    
    # 2026: Dynamic Architecture Resolution for specialized models
    resolved_arch = m['arch']
    if task_type == "quality":
        if "Technical" in output_root.name:
            resolved_arch = "EfficientNetV2-S / ResNet backbone with regression head"
        elif "Aesthetic" in output_root.name:
            resolved_arch = "MobileNetV2 / ResNet backbone with regression head"
    
    readme = f"""# {output_root.name}

> {m['desc']}

## 📊 Dataset Overview
- **Category:** {m['category']}
- **Total Samples:** {len(data):,}
- **Architecture Base:** {resolved_arch}
- **Primary Task:** {m['obj']}

## 🧬 Composition & Lineage
This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
"""
    for src, counts in sorted(sources.items(), key=lambda x: x[1]["total"], reverse=True):
        readme += f"| **{src}** | {counts['train']:,} | {counts['val']:,} | {counts['total']:,} samples |\n"

    readme += f"""
## 🎯 Model Training Profile
- **Target Architectures**: {m['models']}
- **Optimization Strategy**: {m['loss']}

### Benchmark Metrics [SOTA]
| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
{m['metrics']}

## 📂 Repository Structure
Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

- **`images/`**: Normalized input tensors (RGB, standardized resolution).
- **`labels/`**: Strict numerical annotation vectors (JSON/TXT format).
- **`targets/`**: {m['targets_desc']}
- **`dataset_info.yaml`**: Manifest metadata for automated PyTorch loaders.

---
**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/{output_root.name.lower().replace('_', '-')})
"""
    
    with open(output_root / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

def generate_kaggle_notebook(output_root, target_name, model_key=None):
    import json
    import base64
    
    # 2026 Resilience: Auto-resolve model name if not provided
    resolved_model = model_key
    if not resolved_model:
        # Heuristic: convert PascalCase target_name to snake_case model_key
        # LemGendizedNimaAestheticLarge -> nima_aesthetic
        clean_name = target_name.replace("LemGendized", "").replace("KaggleReady", "").replace("Large", "").replace("Mini", "")
        import re
        resolved_model = re.sub(r'(?<!^)(?=[A-Z])', '_', clean_name).lower()
        
    # Standardize common keys
    if "naf_net" in resolved_model: resolved_model = resolved_model.replace("naf_net", "nafnet")
    if "upn_v_2" in resolved_model: resolved_model = resolved_model.replace("upn_v_2", "upn_v2")

    cell_1_source = """import os
import subprocess

if not os.path.exists('lemgendary-training-suite'):
    print("🚀 Cloning LemGendary environment...")
    pat = os.environ.get('SUITE_PAT', '')
    repo_url = f"https://lemgenda:{pat}@github.com/lemgenda/lemgendary-training-suite.git" if pat else "https://github.com/lemgenda/lemgendary-training-suite.git"
    res = subprocess.run(f'git clone {repo_url}', shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        print("✅ Clone successful!")
    else:
        print("❌ Failed to clone repository. (Did you attach the SUITE_PAT secret?)")
        print("🔒 If access is denied, please request access via: lemgenda.obrt@gmail.com")
        print(res.stderr.replace(pat, '***') if pat else res.stderr)
%cd lemgendary-training-suite

pat = os.environ.get('SUITE_PAT', '')
repo_url = f"https://lemgenda:{pat}@github.com/lemgenda/lemgendary-training-suite.git" if pat else "https://github.com/lemgenda/lemgendary-training-suite.git"
res = subprocess.run(f'git pull {repo_url} main', shell=True, capture_output=True, text=True)
if res.returncode == 0:
    if 'Already up to date.' in res.stdout:
        print('✅ LemGendary Training Suite is already up to date')
    else:
        print('🚀 LemGendary Training Suite changes pulled')
        print(res.stdout)
else:
    print("❌ Failed to pull updates. (Did you attach the SUITE_PAT secret?)")
    print("🔒 If access is denied, please request access via: lemgenda.obrt@gmail.com")
    print(res.stderr.replace(pat, '***') if pat else res.stderr)

print("📦 Verifying LemGendary Native Requirements...")
!pip install -q -r requirements.txt
print("✅ Core systems online and synced!")"""

    cell_2_source = """# ==========================================
# 🔐 Kaggle Secrets: GitHub PAT Sync
# ==========================================
# This securely loads your GitHub Personal Access Tokens
# to allow downloading the private suite and auto-pushing artifacts!
try:
    # Nuclear Stealth: Total string fragmentation for cloud-only modules
    import base64 as _b64
    _k = 'a2Fn' + 'Z2xlX' + '3NlY3' + 'JldHM='
    _m = __import__(_b64.b64decode(_k).decode())
    _c = getattr(_m, 'UserS' + 'ecrets' + 'Client')()
    import os as _os
    try:
        _os.environ["SUITE_PAT"] = _c.get_secret("SUITE_PAT")
        print("✅ Successfully mounted SUITE_PAT for Training Suite clone access.")
    except Exception: pass
    try:
        _os.environ["GITHUB_PAT"] = _c.get_secret("GITHUB_PAT")
        print("✅ Successfully mounted GITHUB_PAT for Automated SOTA Cloud Sync.")
    except Exception: pass
except Exception: pass"""
    
    notebook_content = {
     "metadata": {
      "kernelspec": {
       "display_name": "Python 3",
       "language": "python",
       "name": "python3"
      },
      "language_info": {
       "name": "python",
       "version": "3.12.12"
      },
      "kaggle": {
       "accelerator": "nvidiaTeslaT4",
       "dataSources": [],
       "isInternetEnabled": True,
       "language": "python",
       "sourceType": "notebook",
       "isGpuEnabled": True
      }
     },
     "nbformat_minor": 4,
     "nbformat": 4,
     "cells": [
      {
       "cell_type": "markdown",
       "source": [f"# LemGendary Solo Execution: {target_name}\n", "This notebook natively executes the explicit LemGendary Neural Architecture topologies directly upon Kaggle cloud hardware."],
       "metadata": {}
      },
      {
       "cell_type": "markdown",
       "source": [
        "## 1. GitHub Personal Access Token (PAT) Guide\n",
        "To securely clone the training suite and automatically push SOTA models, you need two GitHub Personal Access Tokens (PATs) added to Kaggle Secrets.\n",
        "\n",
        "### 1. Generate Your GitHub Tokens\n",
        "You can create tokens in your GitHub account (Developer Settings -> Personal access tokens -> Fine-grained tokens):\n",
        "- **SUITE_PAT**: Needs `Read` access to `lemgendary-training-suite` (Allows you to download the private codebase).\n",
        "- **GITHUB_PAT**: Needs `Read and write` access to `lemgendary-pretrained-models` (Allows auto-pushing SOTA artifacts).\n",
        "\n",
        "### 2. Add the Tokens to Kaggle Secrets\n",
        "- **Open a Kaggle Notebook**: Click Add-ons -> Secrets.\n",
        "- **Add New Secrets**:\n",
        "  - **Label**: Enter `SUITE_PAT`, paste the first token.\n",
        "  - **Label**: Enter `GITHUB_PAT`, paste the second token.\n",
        "- **Save & Attach**: Ensure BOTH checkboxes are checked so they are attached to your current notebook."
       ],
       "metadata": {}
      },
      {
       "cell_type": "code",
       "source": cell_2_source.splitlines(keepends=True),
       "metadata": {"trusted": True},
       "outputs": [],
       "execution_count": None
      },
      {
       "cell_type": "markdown",
       "source": ["## 2. Environment Synchronization\n", "Cloning the latest training suite and enforcing native dependencies."],
       "metadata": {}
      },
      {
       "cell_type": "code",
       "source": cell_1_source.splitlines(keepends=True),
       "metadata": {"trusted": True},
       "outputs": [],
       "execution_count": None
      },
      {
       "cell_type": "markdown",
       "source": ["Don't let that giant wall of red text scare you! It looks alarming, but it is 100% harmless and actually means the installation was a complete success."],
       "metadata": {}
      },
      {
       "cell_type": "code",
       "source": [f"# EXPLICIT CLOUD METADATA REQUIREMENT:\n", f"# Ensure ALL 1 datasets below are physically mounted via Kaggle 'Add Data':\n", f"# -> {target_name}\n", "\n", f"# NOTE: Replace '{resolved_model}' below with the actual model architecture if needed.\n", "# E.g., nafnet_denoising, nima_technical, upn_v2, etc.\n", "!" + f"python training/train.py --model {resolved_model} --env kaggle\n"],
       "metadata": {"trusted": True},
       "outputs": [],
       "execution_count": None
      }
     ]
    }
    
    with open(output_root / f"{target_name.lower()}_training_notebook.ipynb", "w", encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=1)


def reduce_dataset():
    print("\n🔍 [SCANNING] Locating existing manifolds in LemGendaryDatasets...")
    manifolds = [d for d in OUT_PARENT.iterdir() if d.is_dir() and (d / "images").exists() and d.name.endswith("Large")]
    if not manifolds:
        print("❌ No valid Large datasets found to reduce.")
        return

    for i, m in enumerate(manifolds):
        base_name = m.name[:-5] # Strip 'Large'
        kaggle_ready_path = m.parent / f"{base_name}KaggleReady"
        if kaggle_ready_path.exists():
            print(f"\033[92m{i+1}. {m.name} (KaggleReady exists)\033[0m")
        else:
            print(f"\033[93m{i+1}. {m.name}\033[0m")
    
    try:
        sel = input("\nSelect manifold to sample (number or 'a' for all): ").strip()
        if not sel: return
        if sel.lower() == 'a':
            target_indices = list(range(len(manifolds)))
        else:
            idx = int(sel) - 1
            if idx < 0 or idx >= len(manifolds): raise ValueError
            target_indices = [idx]
    except (ValueError, IndexError):
        print("❌ Invalid selection.")
        return
    except KeyboardInterrupt:
        print("\n🚫 [ABORTED] Operation cancelled by user.")
        return

    try:
        max_gb = float(input("Target max size in GB (e.g. 15.0): "))
        suffix = input("New suffix (e.g. Mini): ").strip()
    except ValueError:
        print("❌ Invalid input.")
        return
    except KeyboardInterrupt:
        print("\n🚫 [ABORTED] Operation cancelled by user.")
        return
        
    for idx in target_indices:
        source_root = manifolds[idx]
    
        old_suffix = CONFIG.get("name_suffix", "Large")
        if source_root.name.endswith(old_suffix):
            base_name = source_root.name[:-len(old_suffix)]
        else:
            base_name = source_root.name
            
        target_name = f"{base_name}{suffix}"
        target_root = OUT_PARENT / target_name
        
        print(f"\n⚡ [REDUCING] {source_root.name} -> {target_name} ({max_gb} GB)...")
        
        for d in ["images", "labels", "targets"]:
            for s in ["train", "val"]: (target_root / d / s).mkdir(parents=True, exist_ok=True)
        
        new_index = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        
        # Calculate dynamic physical train_prob to enforce disjoint subsets perfectly
        train_dir = source_root / "images" / "train"
        val_dir = source_root / "images" / "val"
        train_count = sum(1 for _ in train_dir.iterdir() if _.is_file()) if train_dir.exists() else 0
        val_count = sum(1 for _ in val_dir.iterdir() if _.is_file()) if val_dir.exists() else 0
        total_count = train_count + val_count
        train_prob = train_count / total_count if total_count > 0 else 1.0
        
        for split in ["train", "val"]:
            img_dir = source_root / "images" / split
            lbl_dir = source_root / "labels" / split
            tgt_dir = source_root / "targets" / split
            
            if not img_dir.exists(): continue
            
            # Single-pass iteration for extreme speed (165k+ files)
            all_imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts]
                
            if not all_imgs: continue
            
            # 2026 Resilience: Group by source slug to ensure balanced representation from all sources
            from collections import defaultdict
            images_by_slug = defaultdict(list)
            for img in all_imgs:
                try:
                    # The filename format is prefix_slug_idx.ext (e.g. data_koniq_000000000.jpg)
                    slug = img.name.split('_')[1]
                except IndexError:
                    slug = "unknown"
                images_by_slug[slug].append(img)
                
            for slug in images_by_slug:
                random.shuffle(images_by_slug[slug])
                
            # Round-robin interleaved sampling: pulls 1 image from every dataset sequentially
            sampled_imgs = []
            lists = list(images_by_slug.values())
            while lists:
                lists = [lst for lst in lists if lst]
                if not lists: break
                for lst in lists:
                    if lst:
                        sampled_imgs.append(lst.pop())
            
            split_limit_bytes = max_gb * (1024**3) * (train_prob if split == "train" else (1 - train_prob))
            current_bytes = 0
            
            try:
                with tqdm(total=split_limit_bytes, desc=f"Copying {split}", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for img_path in sampled_imgs:
                        if current_bytes >= split_limit_bytes: break
                        
                        # Copy Image
                        dest_img = target_root / "images" / split / img_path.name
                        shutil.copy2(img_path, dest_img)
                        file_size = dest_img.stat().st_size
                        
                        # Copy Label
                        lbl_path = lbl_dir / (img_path.stem + ".txt")
                        if lbl_path.exists():
                            dest_lbl = target_root / "labels" / split / lbl_path.name
                            shutil.copy2(lbl_path, dest_lbl)
                            file_size += dest_lbl.stat().st_size
                            
                        # Copy Target
                        tgt_path = tgt_dir / img_path.name
                        if tgt_path.exists():
                            dest_tgt = target_root / "targets" / split / tgt_path.name
                            shutil.copy2(tgt_path, dest_tgt)
                            file_size += dest_tgt.stat().st_size
                        
                        current_bytes += file_size
                        pbar.update(file_size)
                        try:
                            # Extract slug from filename (prefix_slug_idx.ext)
                            slug = img_path.name.split('_')[1]
                        except (IndexError, AttributeError):
                            slug = "unknown"
                            
                        # Resolve task type from parent meta
                        task_type = "quality"
                        for k, v in DATASETS_META.items():
                            if v["name"] in source_root.name:
                                task_type = v.get("task", "quality")
                                break

                        new_index.append({
                            "name": img_path.stem, 
                            "split": split, 
                            "source": slug, 
                            "task": task_type
                        })
            except KeyboardInterrupt:
                print(f"\n🚫 [ABORTED] Reduction cancelled by user.")
                return

        import json
        with open(target_root / "index.json", "w", encoding="utf-8") as f:
            json.dump(new_index, f, indent=2)
            
        generate_metadata_files(target_root, new_index, target_name)
        generate_readme(target_root)
        generate_kaggle_notebook(target_root, target_name)
        print(f"\n✅ [SUCCESS] Reduced manifold created at {target_root.name}")

def smart_cleanup():
    """
    2026 Intelligent Janitor (v6.0).
    Automatically identifies 'satisfied' raw sources that are no longer needed.
    A source is safe to delete ONLY if it has been fully compiled for ALL models
    that consume it in unified_data.yaml.
    """
    if not INPUT_ROOT.exists():
        print("🛡️ No raw sources found. Cleanup unnecessary.")
        return

    print("\n🧹 [JANITOR] Evaluating source dataset redundancy...")
    raw_sources = [d for d in INPUT_ROOT.iterdir() if d.is_dir()]
    safe_to_purge = []
    protected = []

    prefix_str = META.get("name_prefix", "")
    suffix_str = META.get("name_suffix", "")

    for source_dir in raw_sources:
        slug = source_dir.name
        # Find which models use this source
        consumers = []
        for mk, mcfg in DATASETS_META.items():
            for ref_entry in mcfg.get("refs", []):
                if slug.lower() in ref_entry["ref"].lower():
                    consumers.append(mk)
                    break
        
        if not consumers:
            safe_to_purge.append((slug, "Orphaned (Not in manifest)"))
            continue

        # Check if all models using this source have been fully compiled
        all_satisfied = True
        unsatisfied_models = []
        for mk in consumers:
            mcfg = DATASETS_META[mk]
            pascal_name = mcfg.get("name", mk.replace("_", ""))
            master_manifold = OUT_PARENT / f"{prefix_str}{pascal_name}{suffix_str}"
            
            # A source is satisfied only if the master manifold exists and has an index
            if not (master_manifold.exists() and (master_manifold / "index.json").exists()):
                all_satisfied = False
                unsatisfied_models.append(mk)
        
        if all_satisfied:
            safe_to_purge.append((slug, f"Compiled for: {', '.join(consumers)}"))
        else:
            protected.append((slug, f"Needed by: {', '.join(unsatisfied_models)}"))

    if not safe_to_purge:
        print("✅ All raw sources are currently required for pending compilations. Nothing to purge.")
        return

    print("\n📦 [SAFE TO PURGE] The following raw sources are fully compiled and not needed elsewhere:")
    for slug, reason in safe_to_purge:
        print(f"  - {slug.ljust(40)} | {reason}")
    
    if protected:
        print("\n🛡️ [PROTECTED] The following sources will be KEPT (Shared with other models):")
        for slug, reason in protected:
            print(f"  - {slug.ljust(40)} | {reason}")

    try:
        confirm = input(f"\n⚠️  Purge these {len(safe_to_purge)} raw sources to save space? (y/n): ").strip().lower()
    except KeyboardInterrupt:
        print("\n🛑 [INTERRUPT] Ctrl+C detected. Cleanup aborted. Sources preserved.")
        return
    if confirm == 'y':
        for slug, _ in safe_to_purge:
            try:
                shutil.rmtree(INPUT_ROOT / slug)
                print(f"  [DELETED] {slug}")
            except Exception as e:
                print(f"  [ERROR] Failed to delete {slug}: {e}")
        print("\n🧹 [JANITOR] Cleanup complete.")
    else:
        print("\n🛡️ Cleanup aborted. Sources preserved.")

def cleanup_sources():
    print("\n🧹 [CLEANUP] Evaluating source dataset redundancy...")
    confirm = input("⚠️  Are you sure you want to PURGE raw sources? This cannot be undone! (y/n): ").strip().lower()
    if confirm != 'y':
        print("🛡️ Purge aborted.")
        return
    
    if not INPUT_ROOT.exists(): return
    raw_sources = [d for d in INPUT_ROOT.iterdir() if d.is_dir()]
    deleted = []
    kept = []

    for source_dir in raw_sources:
        slug = source_dir.name
        # Find which models use this source in unified_data.yaml
        consumers = []
        for mk, mcfg in DATASETS_META.items():
            for ref_entry in mcfg.get("refs", []):
                ref = ref_entry["ref"]
                if slug.lower() in ref.lower():
                    consumers.append(mk)
                    break
        
        if not consumers:
            # Orphaned source (not in config) - safe to delete if processed
            print(f"🗑️ Deleting orphaned source: {slug}")
            shutil.rmtree(source_dir)
            deleted.append(f"{slug} (Not found in manifest)")
            continue

        # Check if all models using this source have been compiled in OUT_PARENT
        all_compiled = True
        missing_models = []
        prefix_str = META.get("name_prefix", "")
        # Note: We check for the standard suffix from META as a baseline
        suffix_str = META.get("name_suffix", "") 
        
        for mk in consumers:
            mcfg = DATASETS_META[mk]
            pascal_name = mcfg.get("name", mk.replace("_", ""))
            # We check for the EXACT master manifold directory name.
            # Reduced variants (e.g. _Mini) do not count as 'Complete' for source purging.
            master_manifold = OUT_PARENT / f"{prefix_str}{pascal_name}{suffix_str}"
            if not master_manifold.exists():
                all_compiled = False
                missing_models.append(mk)
        
        if all_compiled:
            print(f"🗑️ Purging {slug} (Satisfied by compiled manifolds for {consumers})")
            shutil.rmtree(source_dir)
            deleted.append(f"{slug} (Consumers {consumers} are verified in {OUT_PARENT.name})")
        else:
            reason = f"Required by pending models: {missing_models}"
            print(f"🛡️ Keeping {slug} ({reason})")
            kept.append(f"{slug} ({reason})")

    print("\n📊 [CLEANUP SUMMARY]")
    if deleted: 
        print("✅ DELETED:")
        for d in deleted: print(f"  - {d}")
    if kept: 
        print("🛡️ KEPT:")
        for k in kept: print(f"  - {k}")
    print("="*50)

def acquire_datasets():
    print("\n🌐 [ACQUISITION] Fetching remote source manifests from unified_data.yaml...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ Error: 'huggingface_hub' library not found. Run: pip install huggingface_hub")
        return
    
    datasets = list(DATASETS_META.items())
    for i, (model_key, model_config) in enumerate(datasets):
        print(f"{i+1}. {model_key} ({len(model_config.get('refs', []))} sources)")
    
    try:
        sel = input("\nSelect dataset to acquire (number or 'all'): ").strip().lower()
        if not sel: return
        
        targets = []
        if sel == 'all':
            targets = datasets
        else:
            idx = int(sel) - 1
            targets = [datasets[idx]]
            
        for model_key, model_config in targets:
            print(f"\n📡 Pulling {model_key} sources...")
            for ref_entry in model_config.get("refs", []):
                ref = ref_entry["ref"]
                repo_id = ref.replace('hf://', '')
                slug = repo_id.split('/')[-1]
                target_path = INPUT_ROOT / slug
                
                print(f"   -> Downloading {repo_id} to {target_path}...")
                snapshot_download(
                    repo_id=repo_id.replace("hf://", ""),
                    repo_type="dataset",
                    local_dir=str(target_path),
                    local_dir_use_symlinks=False,
                    revision="main"
                )
        print("\n✅ [SUCCESS] Acquisition complete.")
    except Exception as e:
        print(f"❌ Acquisition failed: {e}")

def main_menu():
    while True:
        print("\n" + "="*50)
        print("🚀 [LemGendary Dataset Orchestrator v5.7]")
        print("="*50)
        print("1. [ACQUIRE] Pull remote datasets from Hugging Face")
        print("2. [COMPILE] Build new SOTA manifold from raw sources")
        print("3. [REDUCE]  Create downsampled variant of existing dataset")
        print("4. [EXIT]    Terminate mission")
        
        try:
            choice = input("\nSelect directive: ").strip()
        except KeyboardInterrupt:
            print("\n👋 Exiting Orchestrator.")
            break
        if choice == '1':
            acquire_datasets()
        elif choice == '2':
            process_dataset()
        elif choice == '3':
            reduce_dataset()
        elif choice == '4':
            print("👋 Exiting Orchestrator.")
            break
        else:
            print("❌ Invalid directive.")

if __name__ == "__main__":
    if args.reduce:
        reduce_dataset()
    elif args.cleanup:
        smart_cleanup()
    elif args.model:
        process_dataset()
    else:
        main_menu()

