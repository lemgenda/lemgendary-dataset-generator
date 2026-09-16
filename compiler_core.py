# 2026: Environment Linter Sync (Last Verified: 2026-05-01)
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 2026 Phase 1.5.5: Runtime environment bootstrap extracted to runtime/.
# Applies UTF-8 stdio, Fortran env vars, and torch CUDA patches once.
from runtime.environment import bootstrap_runtime, get_device_info  # noqa: E402
bootstrap_runtime()

import json
import pandas as pd
import random
import argparse
import hashlib
import shutil
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image, ImageOps, ImageFile
from doc_generator import generate_dataset_docs
from notebook_generator import generate_training_notebook, generate_colab_training_notebook
# PIL exposes LOAD_TRUNCATED_IMAGES and MAX_IMAGE_PIXELS as module-level
# module attributes but does not declare them in its type stubs. Use
# setattr() to bypass the stub without a # type: ignore.
setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
setattr(Image, "MAX_IMAGE_PIXELS", None)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import requests
from multiprocessing import Manager
import multiprocessing
import io
import sqlite3
import webdataset as wds
from datetime import datetime
import yaml
from sklearn.cluster import MiniBatchKMeans
import time
from tqdm import tqdm
import functools
from safetensors import safe_open
from typing import Any, TypedDict, cast

# 2026 Phase 1.4: Annotation parsers extracted to converters/
from converters import (  # noqa: E402
    detect_annotations,
    parse_coco,
    parse_parquet,
    parse_xml,
    parse_yolo,
    parse_matlab,
    parse_safetensors,
)

# 2026 Phase 1.5.5: Utility helpers extracted to utils/ and registry/
# These re-exports preserve backward compatibility for callers that do
# `from compiler_core import get_dir_size, compute_hash, ...`.
from utils.fs import get_dir_size, remove_empty_dirs  # noqa: E402,F401
from utils.geometry import convert_bbox_xywh_to_yolo, normalize_points  # noqa: E402,F401
from utils.hashing import compute_hash  # noqa: E402,F401
from utils.image import ensure_srgb, is_black_image  # noqa: E402,F401
from utils.math import get_gaussian_probs  # noqa: E402,F401
from utils.naming import clean_slug, map_category  # noqa: E402,F401
from utils.net import download_image  # noqa: E402,F401
from registry import initialize_registry, ensure_registry_schema  # noqa: E402,F401
from audit.ground_truth import GroundTruthCaches  # noqa: E402


# ---------------- CONFIG ----------------
CONFIG_PATH = Path("./config.json")
DEFAULT_CONFIG = {
    "train_split": 0.8,
    "num_workers": max(1, multiprocessing.cpu_count() - 2),
    "diffusion_size": 1024,
    "black_threshold": 0.1,
    "nima_threshold": 4.0,
    "enable_dedup": False,
    "strict_ground_truth": False,
}
CONFIG = {**DEFAULT_CONFIG, **json.load(open(CONFIG_PATH))} if CONFIG_PATH.exists() else DEFAULT_CONFIG

# 2026 Phase 1.1: Route all YAML access through pydantic schema validation.
from config_schema import load_unified_data, UnifiedData

try:
    _UNIFIED: UnifiedData = load_unified_data(Path("./unified_data.yaml"))
    YAML_DATA = _UNIFIED.to_legacy_dict()
except FileNotFoundError as e:
    print(f"[FATAL] {e}")
    sys.exit(3)
except Exception as e:
    print(f"[FATAL] unified_data.yaml failed schema validation:")
    print(f"  {type(e).__name__}: {e}")
    print("[REMEDY] Run `python config_schema.py` to see detailed field diagnostics.")
    sys.exit(2)

META = YAML_DATA.get("_registry_metadata", {})
VERSION = META.get("version", "4.2.0")


# ---------------- CLI ARGS ----------------
# 2026 Phase 1.5: Parser definition moved to cli_args.py (SSOT). Both this
# module and manifold_compile.py build the same parser; argparse does not
# consume sys.argv, so both `parse_args()` calls succeed against the same
# argument vector.
from cli_args import build_parser

parser = build_parser()
args = parser.parse_args()

INPUT_ROOT = Path("./raw-sets")
OUT_PARENT = Path(META.get("output_folder_name", "../LemGendaryDatasets"))
CATEGORY_MAP_PATH = Path("./category_map.json")
CATEGORY_MAP = json.load(open(CATEGORY_MAP_PATH)) if CATEGORY_MAP_PATH.exists() else {}
DATASETS_META = YAML_DATA.get("datasets", {})

if args.workers:
    CONFIG["num_workers"] = args.workers


# ---------------- GLOBALS ----------------
SENTRY = None
LABELER = None
CAPTIONER = None
CLIP_MANIFOLD = None

# 2026 Phase 1.5.5: Ground-truth caches consolidated into a single dataclass.
# Workers populate this in init_worker(); consumers read from _GT_CACHE.{ava,aadb,tid}.
_GT_CACHE = GroundTruthCaches()


class _Annotation(TypedDict):
    """Typed shape of every record appended to `annotations` in `process_image`.

    `data` is widened to ``list[Any]`` because three source families produce
    numeric lists of different underlying types:
      * COCO / YOLO       -> list[float]
      * Parquet / MATLAB  -> list[Any]
      * NPZ               -> list[Any]
    """
    type: str        # 'bbox' | 'segmentation' | 'pose'
    cls: int
    data: list[Any]


def detect_task(model_dir_name):
    """Route a model key to its task class. Compiler-specific routing logic."""
    if not model_dir_name:
        return "quality"
    name = str(model_dir_name).lower()

    if DATASETS_META:
        for ds_key, ds_cfg in DATASETS_META.items():
            if ds_key.lower() == name or (ds_cfg.get("name", "").lower() == name):
                override = ds_cfg.get("task_override")
                if override:
                    return override

    task_patterns = [
        (["diffusion", "vlm", "vision_language"], "diffusion"),
        (["seg", "mask", "parsenet"], "segmentation"),
        (["pose", "face"], "pose"),
        (["nima", "aesthetic", "quality"], "quality"),
        (["classify", "classification", "authentic", "authenticity"], "classification"),
        (["sr", "ultrazoom", "x2", "x3", "x4", "x8", "super"], "super-resolution"),
        ([
            "deraining", "debluring", "denoising", "dehazing", "lowlight", "exposure",
            "restorer", "enhance", "restoration", "ffanet", "mirnet", "mprnet", "nafnet",
            "upn", "codeformer",
        ], "restoration"),
    ]
    for patterns, t_name in task_patterns:
        if any(k in name for k in patterns):
            return t_name
    return "detection"


DPED_CACHE = set()
PHYSICAL_INDEX = set()


def init_worker(config, dped_cache=None, physical_index=None):
    """Per-process worker bootstrap. Loads models, GT caches, and device config."""
    global SENTRY, CAPTIONER, CLIP_MANIFOLD, DPED_CACHE, PHYSICAL_INDEX, _GT_CACHE
    if dped_cache:
        DPED_CACHE = dped_cache
    if physical_index:
        PHYSICAL_INDEX = physical_index

    # 2026 Modular Alignment: Local imports from encapsulated modules.
    # The `models/` package ships with py.typed-style imports resolvable
    # from the project root; no suppression should be required. If Pyrefly
    # still reports an error here, the fix is to add or repair
    # ``models/__init__.py`` — not to re-add a # type: ignore.
    from models.quality_scorer import QualitySentry
    from models.detection import AutoLabeler
    from models.diffusion import CaptionSentry
    from models.encoder import CLIPManifold

    # Workers ignore SIGINT to prevent traceback noise (subprocesses only).
    if os.name == "nt" and multiprocessing.current_process().name != "MainProcess":
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    import torch
    from PIL import ImageFile
    setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)

    # Prevent thread thrashing on CPU in ProcessPool mode.
    if os.name != "nt" or multiprocessing.current_process().name != "MainProcess":
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    # Device selection: CPU on Windows or when no CUDA, else distribute across GPUs.
    if os.name == "nt" or not torch.cuda.is_available():
        device = "cpu"
    else:
        gpu_count = torch.cuda.device_count()
        if physical_index is not None and gpu_count > 0:
            device = f"cuda:{physical_index % gpu_count}"
        else:
            device = "cuda:0"

    # 1. Quality Vetting (NIMA)
    mission = detect_task(args.model)
    if mission in ["quality", "classification", "diffusion"] and not args.no_vetting:
        model_type = "aesthetic" if mission == "diffusion" or (args.model and "aesthetic" in args.model) else "technical"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", f"nima_{model_type}_best.pth")
        if os.path.exists(model_path):
            try:
                SENTRY = QualitySentry(model_path, model_name=model_type, device=device)
            except Exception:
                pass

    # 2. Ground truth caches (AVA / AADB / TID via registry loaders)
    if mission in ["quality", "classification", "diffusion", "restoration"]:
        _GT_CACHE.load(INPUT_ROOT, args.model or "")

    # 3. Diffusion captioning
    if mission == "diffusion":
        try:
            CAPTIONER = CaptionSentry(device=device)
        except Exception:
            pass

    # 4. Style manifold (CLIP)
    if "clip" in str(config):
        try:
            CLIP_MANIFOLD = CLIPManifold(device="cpu")
        except Exception:
            pass


def get_labeler(task, device="cuda"):
    from models.detection import AutoLabeler
    global LABELER
    if LABELER is None:
        LABELER = {}
    if task not in LABELER:
        if task == "face_detection":
            mode = "face_landmarks"
        else:
            mode = "segmentation" if "seg" in task else "detection"
        LABELER[task] = AutoLabeler(mode=mode, device=device)
    return LABELER[task]


# ---------------- BATCH WORKER ----------------
def process_parquet_shard(pq_path, prefix, c_slug, start_idx, task, fmt, split_fallback,
                          output_root_str, skip_lbl, train_prob, existing_names,
                          existing_on_disk, keep_prob, num_rows):
    """Process an entire Parquet shard as one worker task (avoids per-row IPC)."""
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
    """Execute a list of (func, *args) in one worker call to amortize IPC."""
    results = []
    for i, (task_func, *args) in enumerate(tasks):
        try:
            res = task_func(*args)
            if task_func.__name__ == "process_parquet_shard":
                results.extend(res)
            else:
                results.append(res)
        except Exception:
            results.append(None)
    return results


# ---------------- PROCESSORS ----------------
def _is_image_valid_for_dataset(img, w, hgt, task, slug):
    if img and task == "quality" and "laion" not in slug and "ava" not in slug:
        if is_black_image(img, CONFIG["black_threshold"]):
            return False
    if task in ["diffusion"]:
        min_dim = 512
    elif task in ["quality", "classification", "restoration", "super-resolution"]:
        min_dim = 128 if "artifact" in slug.lower() else 224
    else:
        min_dim = 128
    return w >= min_dim and hgt >= min_dim


def _passes_nima_filter(task, slug, is_authenticity, nima_score, nima_probs, current_threshold, idx):
    if nima_probs[0] == 1.0 and task in ["quality", "diffusion"] and not is_authenticity:
        if CONFIG.get("strict_ground_truth", True) and task == "quality" and "laion" not in slug:
            return False
    if task in ["quality", "diffusion"] and nima_score < current_threshold and not is_authenticity:
        if idx < 5:
            print(f"DEBUG: {slug} skipped because nima {nima_score} < {current_threshold}")
        return False
    return True


def process_image(img_input, prefix, slug, idx, task, fmt, ann_data, split,
                  output_root_str, skip_labeling=False):
    """Worker function for parallel processing.

    img_input can be a Path or raw bytes (for Parquet-embedded datasets).
    """
    img_path = "Unknown"
    nima_score = 1.0
    nima_probs = [0.0] * 10
    nima_probs[0] = 1.0
    w, hgt = 0, 0
    img = None
    p_str = ""
    img_data = b""

    try:
        # ── Input normalization ─────────────────────────────────────────
        if isinstance(img_input, (bytes, dict)):
            if isinstance(img_input, dict) and "bytes" in img_input:
                img_data = img_input["bytes"]
            else:
                img_data = img_input if isinstance(img_input, bytes) else b""
            if img_data is None:
                return None
            img = Image.open(io.BytesIO(img_data))
            img_path = Path(f"virtual_{slug}_{idx:09d}.jpg")
            is_st = False
        else:
            img_path_str = str(img_input)
            img_path = Path(img_path_str)
            is_st = img_path_str.lower().endswith(".safetensors")

        ext = img_path.suffix.lower() if isinstance(img_input, (str, Path)) else ".jpg"
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".npy"]:
            ext = ".jpg"

        name = f"{prefix}_{slug}_{idx:09d}"

        source_parent_name = (
            img_path.parent.parent.name.lower()
            if isinstance(img_input, (str, Path)) and len(img_path.parts) >= 3
            else "images"
        )
        out_dir = source_parent_name if source_parent_name in ["targets", "masks"] else "images"
        out_img_path = Path(output_root_str) / out_dir / split / f"{name}{ext}"

        tgt_dir = "masks" if task == "segmentation" else "targets"
        out_tgt_path = Path(output_root_str) / tgt_dir / split / f"{name}{ext}"

        # ── High-speed skip via worker-global physical index ────────────
        if PHYSICAL_INDEX and name.lower() in PHYSICAL_INDEX:
            return {"name": name, "source": slug, "task": task, "split": split, "hash": "skipped", "nima_score": nima_score, "size": 0}

        # ── Restoration target resolution ──────────────────────────────
        target_img = None
        target_img_path = None

        if task in ["restoration", "super-resolution", "parameter_prediction", "segmentation"]:
            # Strategy 1: Parquet/Virtual target detection
            if ann_data:
                row_dict = None
                if isinstance(ann_data, dict):
                    row_dict = ann_data
                elif isinstance(ann_data, tuple) and len(ann_data) == 2:
                    df_sub, _ = ann_data
                    if not df_sub.empty:
                        row_dict = df_sub.iloc[0].to_dict()
                if row_dict:
                    for k in ["target", "sharp", "ground_truth", "gt", "clean", "original", "mask", "masks"]:
                        val = row_dict.get(k)
                        if isinstance(val, bytes):
                            target_img = Image.open(io.BytesIO(val))
                            break
                        elif isinstance(val, str) and val.endswith((".png", ".jpg", ".jpeg")):
                            p = Path(val)
                            if p.exists():
                                target_img_path = str(p)
                            break

            # Strategy 2: Generic neighbor resolve (GoPro, RealBlur, HiDeBlur, SFHQ)
            if not target_img and not target_img_path and not isinstance(img_input, (bytes, dict)):
                blur_keys = ["blur", "blurry", "input", "lowres", "lr", "rain", "hazy", "noisy", "degraded", "distorted", "low", "images"]
                sharp_keys = ["sharp", "gt", "ground_truth", "groundtruth", "clean", "clear", "original", "hr", "highres", "target", "norain", "high", "targets", "mask", "masks", "segmentation", "segmentations"]
                p_str = str(img_path).replace("\\", "/")
                parent = img_path.parent

                # 2a: Sibling folder
                if any(k in parent.name.lower() for k in blur_keys):
                    try:
                        for sibling in parent.parent.iterdir():
                            if sibling.is_dir() and any(k in sibling.name.lower() for k in sharp_keys):
                                potential = sibling / img_path.name
                                if potential.exists():
                                    target_img_path = str(potential)
                                    break
                                for b_k in blur_keys:
                                    if img_path.name.lower().startswith(b_k):
                                        for s_k in sharp_keys:
                                            new_name = img_path.name.lower().replace(b_k, s_k, 1)
                                            p_f = sibling / new_name
                                            if p_f.exists():
                                                target_img_path = str(p_f)
                                                break
                                        if target_img_path:
                                            break
                                if target_img_path:
                                    break
                    except Exception:
                        pass

                # 2b: Ancestral sibling
                if not target_img_path:
                    for ancestor in [parent, parent.parent]:
                        if any(k in ancestor.name.lower() for k in ["train", "test", "val", "images"]):
                            try:
                                for sibling in ancestor.parent.iterdir():
                                    if sibling.is_dir() and any(k in sibling.name.lower() for k in sharp_keys):
                                        potential = sibling / img_path.name
                                        if potential.exists():
                                            target_img_path = str(potential)
                                            break
                                        try:
                                            rel = img_path.relative_to(ancestor)
                                            potential_rel = sibling / rel
                                            if potential_rel.exists():
                                                target_img_path = str(potential_rel)
                                                break
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            if target_img_path:
                                break

                # 2c: Same-folder resolution
                if not target_img_path and not isinstance(img_input, (bytes, dict)):
                    for b_k in blur_keys:
                        if img_path.name.lower().startswith(b_k):
                            for s_k in sharp_keys:
                                new_name = img_path.name.lower().replace(b_k, s_k, 1)
                                if new_name == img_path.name.lower():
                                    continue
                                p_f = parent / new_name
                                if p_f.exists():
                                    target_img_path = str(p_f)
                                    break
                            if target_img_path:
                                break

            # Strategy 3: Legacy DPED fallback
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

        # ── Deferred image loading (only if stats needed) ──────────────
        needs_stats = (task in ["quality", "diffusion"] and not args.no_vetting) or (not skip_labeling)
        if needs_stats:
            if not isinstance(img_input, (bytes, dict)):
                if img_path.suffix.lower() == ".npy":
                    data = np.load(img_path)
                    if data.ndim == 3 and data.shape[0] in [1, 3]:
                        data = data.transpose(1, 2, 0)
                    if data.dtype in [np.float32, np.float64]:
                        data = (data * 255).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(data)
                else:
                    img = Image.open(img_path)
            else:
                img = Image.open(io.BytesIO(img_data))
            img = ensure_srgb(img)
            w, hgt = img.size
            if not _is_image_valid_for_dataset(img, w, hgt, task, slug):
                return None

        # ── NIMA quality gate: GT-first, AI fallback ───────────────────
        nima_score = 1.0
        nima_probs = [0.0] * 10
        nima_probs[0] = 1.0

        # 1. AVA professional labels (10-bin)
        if "ava" in slug and _GT_CACHE.ava:
            try:
                img_id = int(img_path.stem)
                if img_id in _GT_CACHE.ava:
                    votes = _GT_CACHE.ava[img_id]
                    nima_probs = [votes[f"vote_{i}"] for i in range(1, 11)]
                    nima_score = sum(p * (i + 1) for i, p in enumerate(nima_probs))
            except Exception:
                pass

        # 2. AADB human ratings (scalar 0-1)
        elif "aadb" in slug and _GT_CACHE.aadb:
            try:
                raw_score = _GT_CACHE.aadb.get(img_path.name)
                if raw_score is not None:
                    nima_score = (raw_score * 9.0) + 1.0
                    nima_probs = get_gaussian_probs(nima_score)
            except Exception:
                pass

        # 3. LAION aesthetic scores (scalar 1-10)
        elif "laion" in slug:
            try:
                if isinstance(ann_data, dict):
                    val = ann_data.get("aesthetic_score", ann_data.get("score", 6.5))
                    nima_score = float(val) if val is not None else 6.5
                    nima_probs = get_gaussian_probs(nima_score)
                elif fmt == "parquet" and ann_data:
                    df_subset, mapping = ann_data
                    col = mapping.get("aesthetic_score", "aesthetic_score")
                    if col in df_subset.columns:
                        nima_score = float(df_subset[col].iloc[0])
                        nima_probs = get_gaussian_probs(nima_score)
            except Exception:
                pass

        # 4. Universal technical scores (scalar 1-10)
        elif _GT_CACHE.tid and img_path.name.lower() in _GT_CACHE.tid:
            try:
                raw_score = _GT_CACHE.tid.get(img_path.name.lower())
                if raw_score is not None:
                    nima_score = raw_score
                    nima_probs = get_gaussian_probs(nima_score)
            except Exception:
                pass

        # 4.5. Authenticity override (AI vs Human)
        is_authenticity = "authentic" in prefix.lower()
        if is_authenticity:
            parent_name = img_path.parent.name.lower()
            file_name = img_path.name.lower()
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

        # 5. AI vetting fallback
        if nima_probs[0] == 1.0 and task in ["quality", "diffusion"] and not is_authenticity:
            if SENTRY:
                nima_score, nima_probs = SENTRY.score(img, return_probs=True)

        current_threshold = 5.5 if task == "diffusion" else CONFIG["nima_threshold"]
        if not _passes_nima_filter(task, slug, is_authenticity, nima_score, nima_probs, current_threshold, idx):
            return None

        # ── Hash + output writes ───────────────────────────────────────
        hash_target = img_data if isinstance(img_input, (bytes, dict)) else img_path
        h = compute_hash(hash_target, no_hash=args.no_hash) if CONFIG["enable_dedup"] else None

        is_already_on_disk = PHYSICAL_INDEX and name.lower() in PHYSICAL_INDEX
        is_clean_only = ("parsenet" in slug.lower() or "codeformer" in slug.lower()) and task == "restoration"
        if is_clean_only and not target_img_path and isinstance(img_input, (str, Path)):
            target_img_path = str(img_path)

        if not is_already_on_disk and not is_clean_only:
            if not img and isinstance(img_input, (str, Path)):
                try:
                    os.link(str(img_path), str(out_img_path))
                except (OSError, AttributeError):
                    try:
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
                    try:
                        shutil.copy(target_img_path, out_tgt_path)
                    except Exception:
                        pass
            elif target_img:
                save_fmt = "PNG" if ext == ".png" else "JPEG"
                target_img.save(out_tgt_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
            else:
                if is_clean_only:
                    if isinstance(img_input, (bytes, dict)):
                        with open(out_tgt_path, "wb") as f:
                            f.write(img_data)
                    elif img:
                        save_fmt = "PNG" if ext == ".png" else "JPEG"
                        img.save(out_tgt_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
                else:
                    try:
                        os.link(str(out_img_path), str(out_tgt_path))
                    except (OSError, AttributeError):
                        try:
                            shutil.copy2(out_img_path, out_tgt_path)
                        except Exception:
                            pass
        elif task == "parameter_prediction":
            if target_img_path:
                try:
                    os.link(target_img_path, str(out_tgt_path))
                except (OSError, AttributeError):
                    try:
                        shutil.copy(target_img_path, out_tgt_path)
                    except Exception:
                        pass
            elif target_img:
                save_fmt = "PNG" if ext == ".png" else "JPEG"
                target_img.save(out_tgt_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
            else:
                try:
                    os.link(str(out_img_path), str(out_tgt_path))
                except (OSError, AttributeError):
                    try:
                        shutil.copy2(str(out_img_path), str(out_tgt_path))
                    except Exception:
                        pass

        # ── Annotation dispatch ────────────────────────────────────────
        annotations: list[_Annotation] = []

        if fmt == "coco" and ann_data is not None:
            for a in ann_data:
                cls = map_category(str(a["category_id"]), prefix, CATEGORY_MAP)
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
                cls = map_category(row[mapping.get("class", "class")], prefix, CATEGORY_MAP)
                if mapping.get("segmentation") in row and row[mapping.get("segmentation")]:
                    poly = normalize_points(row[mapping.get("segmentation")], w, hgt, stride=2)
                    annotations.append({"type": "segmentation", "cls": cls, "data": poly})
                elif mapping.get("keypoints") in row and row[mapping.get("keypoints")]:
                    kpts = normalize_points(row[mapping.get("keypoints")], w, hgt, stride=3)
                    annotations.append({"type": "pose", "cls": cls, "data": [0, 0, 0, 0] + kpts})
                else:
                    bbox = [row[mapping.get("xmin", "xmin")], row[mapping.get("ymin", "ymin")],
                            row[mapping.get("width", "width")], row[mapping.get("height", "height")]]
                    annotations.append({"type": "bbox", "cls": cls, "data": bbox})

        elif fmt == "matlab" and ann_data:
            for entry in ann_data:
                try:
                    cls = map_category(entry["class"], prefix, CATEGORY_MAP)
                    annotations.append({"type": "bbox", "cls": cls, "data": entry["bbox"]})
                except Exception:
                    pass

        elif fmt == "xml" and ann_data:
            # Contract: when fmt == "xml", manifold_compile.py passes a Path
            # or str to the annotations directory. Narrow explicitly rather
            # than suppressing — the runtime check matches the caller contract.
            if not isinstance(ann_data, (str, Path)):
                return None
            xml_anns = parse_xml(ann_data)

        elif fmt == "yolo" and ann_data:
            if not isinstance(ann_data, (str, Path)):
                return None
            yolo_anns = parse_yolo(ann_data, w, hgt)
            for a in yolo_anns:
                if task == "pose":
                    cls = map_category("0", prefix, CATEGORY_MAP)
                else:
                    cls = map_category(a["class"], prefix, CATEGORY_MAP)
                a_bbox = cast(list[Any], a["bbox"])
                if "keypoints" in a and a["keypoints"]:
                    a_kpts = cast(list[Any], a["keypoints"])
                    annotations.append({"type": "pose", "cls": cls, "data": a_bbox + a_kpts})
                else:
                    annotations.append({"type": "bbox", "cls": cls, "data": a_bbox})

        elif fmt == "npz" and ann_data:
            if not isinstance(ann_data, (str, Path)):
                return None
            try:
                data = np.load(str(ann_data))
                if "landmarks" in data.files:
                    landmarks = data["landmarks"]
                    if landmarks.shape == (110, 2):
                        pts = landmarks[:68]
                        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
                        pad_w = (x_max - x_min) * 0.05
                        pad_h = (y_max - y_min) * 0.05
                        x_min = max(0, x_min - pad_w)
                        y_min = max(0, y_min - pad_h)
                        x_max = min(w, x_max + pad_w)
                        y_max = min(hgt, y_max + pad_h)
                        bbox_w = x_max - x_min
                        bbox_h = y_max - y_min
                        l_eye = landmarks[36:42].mean(axis=0)
                        r_eye = landmarks[42:48].mean(axis=0)
                        nose = landmarks[30]
                        l_mouth = landmarks[48]
                        r_mouth = landmarks[54]
                        kpts = [
                            l_eye[0], l_eye[1], r_eye[0], r_eye[1], nose[0], nose[1],
                            l_mouth[0], l_mouth[1], r_mouth[0], r_mouth[1],
                        ]
                        cls = map_category("0", prefix, CATEGORY_MAP)
                        annotations.append({"type": "pose", "cls": cls, "data": [x_min, y_min, bbox_w, bbox_h] + kpts})
            except Exception as e:
                print(f"Error parsing NPZ {ann_data}: {e}")

        elif fmt == "safetensors" and isinstance(ann_data, dict):
            metadata = ann_data
            tags = []
            if "ss_tag_frequency" in metadata:
                try:
                    freqs = json.loads(str(metadata["ss_tag_frequency"]))
                    for bucket in freqs.values():
                        tags.extend(bucket.keys())
                except Exception:
                    pass
            if not tags and "ss_datasets" in metadata:
                try:
                    ds_info = json.loads(str(metadata["ss_datasets"]))
                    for ds in ds_info:
                        if "tag_frequency" in ds:
                            tags.extend(ds["tag_frequency"].keys())
                except Exception:
                    pass
            if tags:
                unique_tags = list(set(tags))[:20]
                for tag in unique_tags:
                    cls = map_category(tag, prefix, CATEGORY_MAP)
                    annotations.append({"type": "bbox", "cls": cls, "data": [0.0, 0.0, 1.0, 1.0]})

        is_autolabeled = False
        if not annotations and task not in ["quality", "classification"] and not args.no_labeling and not skip_labeling:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            labeler = get_labeler(task, device)
            annotations = cast(list[_Annotation], labeler.predict(img))
            if annotations:
                is_autolabeled = True

        # ── Write label file ───────────────────────────────────────────
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
                        if any(k in p_lower for k in ["fake", "ai", "synthetic", "nsfw", "porn", "explicit"]):
                            class_label = 0
                    if isinstance(class_label, float):
                        try:
                            import pandas as pd
                            if not pd.isna(class_label):
                                class_label = int(class_label)
                        except Exception:
                            pass
                    f.write(str(class_label) + "\n")
                elif annotations:
                    for ann in annotations:
                        cls = ann["cls"]
                        data = ann["data"]
                        if ann["type"] == "bbox":
                            yolo = convert_bbox_xywh_to_yolo(data, w, hgt)
                            f.write(f"{cls} {' '.join(map(str, yolo))}\n")
                        elif ann["type"] == "segmentation":
                            f.write(f"{cls} {' '.join(map(str, data))}\n")
                        elif ann["type"] == "pose":
                            yolo_box = convert_bbox_xywh_to_yolo(data[:4], w, hgt)
                            f.write(f"{cls} {' '.join(map(str, yolo_box))} {' '.join(map(str, data[4:]))}\n")

        size_bytes = 0
        try:
            if out_img_path.exists():
                size_bytes += out_img_path.stat().st_size
        except Exception:
            pass

        return {
            "name": name, "source": slug, "task": task, "split": split,
            "hash": h, "nima_score": round(nima_score, 3), "is_autolabeled": is_autolabeled,
            "has_segmentation": any(a["type"] == "segmentation" for a in annotations),
            "has_pose": any(a["type"] == "pose" for a in annotations),
            "label_path": str(label_file_path.resolve()), "path": str(out_img_path.resolve()),
            "size": size_bytes,
            "clip_latent": None,
        }

    except Exception as e:
        print(f"[ERROR] processing {img_path}: {e}")
        return None


def process_diffusion(img_path, prefix, slug, idx, split, output_root_str):
    """Text-image processor for Diffusion manifolds."""
    try:
        if isinstance(img_path, (bytes, dict)):
            raw = img_path["bytes"] if isinstance(img_path, dict) and "bytes" in img_path else img_path
            # Pyrefly: narrow `raw` to bytes. Replaces the weaker `is None`
            # check — a dict-typed value could theoretically be non-None but
            # not bytes, and BytesIO would then raise at runtime.
            if not isinstance(raw, bytes):
                return None
            img_data = raw
            img = Image.open(io.BytesIO(img_data))
            is_virtual = True
        else:
            if isinstance(img_path, str):
                img_path = Path(img_path)
            if not img_path.exists():
                return None
            img = Image.open(img_path)
            is_virtual = False

        img = ensure_srgb(img)
        if is_black_image(img, CONFIG["black_threshold"]):
            return None

        nima_score = 10.0
        if SENTRY:
            nima_score = SENTRY.score(img)
            if nima_score < CONFIG["nima_threshold"]:
                return None

        # Pyrefly: CONFIG is a dict[str, Any] and .get() returns Any. PIL's
        # resize() requires an int or an int-pair; wrap in int() for a clean
        # narrowing and a runtime guard against a malformed config value.
        size = int(CONFIG.get("diffusion_size", 512))
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        caption = "a high quality image"
        if CAPTIONER:
            caption_file = (
                Path(img_path).parent / (Path(img_path).stem + ".txt")
                if not is_virtual and isinstance(img_path, (str, Path))
                else None
            )
            if caption_file and caption_file.exists():
                caption = caption_file.read_text().strip()
            else:
                caption = CAPTIONER.generate(img)

        style_tag = "standard"
        clip_latent = None
        if CLIP_MANIFOLD:
            style_tag = CLIP_MANIFOLD.tag_style(img)
            clip_latent = CLIP_MANIFOLD.extract_features(img).cpu().numpy().flatten().tolist()

        h = compute_hash(img, no_hash=args.no_hash) if CONFIG["enable_dedup"] else None
        name = f"{prefix}_{idx:09d}"

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        img_bytes = buffer.getvalue()

        latent_blob = None
        if clip_latent:
            latent_blob = sqlite3.Binary(np.array(clip_latent).astype(np.float32).tobytes())

        nima_val = float(nima_score[0]) if isinstance(nima_score, (tuple, list)) else float(nima_score)
        return {
            "name": name, "source": slug, "task": "diffusion", "split": split,
            "hash": h, "nima_score": round(nima_val, 3),
            "caption": caption, "style_tag": style_tag, "clip_latent": latent_blob,
            "img_bytes": img_bytes, "size": len(img_bytes),
        }
    except Exception as e:
        safe_path = "virtual_bytes" if isinstance(img_path, (bytes, dict)) else img_path
        print(f"[ERROR] Error processing diffusion sample {safe_path}: {e}")
        return None


# ---------------- ORCHESTRATOR ----------------