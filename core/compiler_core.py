"""
LemGendary Dataset Compiler — Core Coordinator.

Owns the compile-time pipeline:
  - worker bootstrap (init_worker)
  - parallel batch dispatch (batch_worker / process_parquet_shard)
  - per-sample processing (process_image / process_diffusion)

Re-exports the utility, audit, registry, converter, transcode, format, and
generator symbols extracted in Phases 1.4 through 5.0 so existing call sites
in manifold_compile.py keep working unchanged.

Import-order contract
---------------------
Runtime env vars (TORCH_CUDA_ARCH_LIST, CUDA_FORCE_PTX_JIT,
PYTORCH_CUDA_ALLOC_CONF) must be set before `import torch`, because torch
reads them at module-import time. bootstrap_runtime() therefore runs between
the stdlib imports and the heavy imports. The resulting E402 warnings are
declared once in pyproject.toml — never inline.

CLI contract
------------
This module DOES NOT parse sys.argv on import. The entry point must call::

    import compiler_core as cc
    args = cc.configure()       # parses sys.argv (or a list), applies overrides

before invoking any worker function. Spawned worker processes on Windows
re-import this module with a different argv; the parent MUST forward the
Namespace through the initializer::

    Pool(..., initializer=cc.init_worker,
         initargs=(cc.CONFIG, dped, phys, cc.get_args()))
"""

# ─── Standard library (safe to import anywhere) ─────────────────────────────
import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runtime.environment import bootstrap_runtime

bootstrap_runtime()

# ─── Heavy imports (env vars are now in place) ─────────────────────────────
import io
import json
import multiprocessing
import random
import shutil
import sqlite3
from typing import Any, Literal, TypedDict, cast

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image, ImageFile

setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
Image.MAX_IMAGE_PIXELS = None


# ─── Public surface ─────────────────────────────────────────────────────────
__all__ = [
    # Config constants (defined below)
    "CONFIG", "DEFAULT_CONFIG", "META", "VERSION", "YAML_DATA",
    "INPUT_ROOT", "OUT_PARENT", "CATEGORY_MAP", "DATASETS_META",
    "CATEGORY_MAP_PATH", "CONFIG_PATH",
    # CLI configuration (Phase 1.5 — replaces module-level `args` / `parser`)
    "configure", "get_args", "set_args",
    # Runtime (Phase 1.7)
    "bootstrap_runtime", "get_device_info",
    # Utils (Phase 1.5.5)
    "get_dir_size", "remove_empty_dirs",
    "convert_bbox_xywh_to_yolo", "normalize_points",
    "compute_hash", "ensure_srgb", "is_black_image",
    "get_gaussian_probs", "clean_slug", "map_category", "download_image",
    # Registry (Phase 1.3 / 1.5.5)
    "initialize_registry", "ensure_registry_schema",
    # Ground truth (Phase 1.5.5)
    "GroundTruthCaches",
    # Converters (Phase 1.4)
    "detect_annotations", "parse_coco", "parse_parquet", "parse_xml",
    "parse_yolo", "parse_matlab", "parse_safetensors",
    # Audit (Phase 2)
    "VisionAuditor", "ExactHasher", "PerceptualHasher", "RejectLog",
    # Transcode (Phase 3)
    "ImageTranscoder", "KeepFormatError", "ImageFormatPolicy",
    # Formats (Phase 4)
    "ShardWriter", "WebDatasetWriter", "MDSWriter", "LitDataWriter",
    "ParquetWriter", "DirectoryWriter", "DirectorySampleSource",
    "Sample", "Writer", "make_writer", "parse_also_format",
    # Generators (Phase 5)
    "GenerationResult", "LabelGenerator", "PromptGenerator", "MaskGenerator",
    # Degradation Engine (Phase 6)
    "DegradationProfile", "CompositeProfile", "DynamicDegrader", "composite", "parse_profile",
    # Core API (defined below)
    "detect_task", "get_labeler",
    "init_worker", "batch_worker",
    "process_image", "process_diffusion", "process_parquet_shard",
]


# ─── Config ─────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("./config.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "train_split": 0.8,
    "num_workers": max(1, multiprocessing.cpu_count() - 2),
    "diffusion_size": 1024,
    "black_threshold": 0.1,
    "nima_threshold": 4.0,
    "enable_dedup": False,
    "enable_perceptual_dedup": False,   # Phase 2
    "audit_debug_rejects": False,       # Phase 2
    "strict_ground_truth": False,
    "image_format": "webp",             # Phase 3
    "image_quality": 92,                # Phase 3
    "target_quality": 95,               # Phase 3
    "mask_format": "webp-lossless",     # Phase 3
    "label_strategy": None,             # Phase 5 — None => infer from task
    "prompt_strategy": None,            # Phase 5
    "mask_strategy": None,              # Phase 5 — None => infer from task
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
        CONFIG: dict[str, Any] = {**DEFAULT_CONFIG, **json.load(_f)}
else:
    CONFIG = dict(DEFAULT_CONFIG)


ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "core" else Path(__file__).resolve().parent
from core.config_schema import ImageFormatPolicy, UnifiedData, load_unified_data

try:
    _manifest_path = ROOT / "unified_data.yaml"
    if not _manifest_path.exists():
        _manifest_path = Path("./unified_data.yaml")
    _UNIFIED: UnifiedData = load_unified_data(_manifest_path)
    YAML_DATA: dict[str, Any] = _UNIFIED.to_legacy_dict()
except FileNotFoundError as e:
    print(f"[FATAL] {e}")
    sys.exit(3)
except Exception as e:
    print("[FATAL] unified_data.yaml failed schema validation:")
    print(f"  {type(e).__name__}: {e}")
    print("[REMEDY] Run `python config_schema.py` to see detailed field diagnostics.")
    sys.exit(2)

META: dict[str, Any] = YAML_DATA.get("_registry_metadata", {})
VERSION: str = META.get("version", "4.2.0")


# ─── CLI configuration (Phase 1.5: SSOT in cli_args.py) ─────────────────────
# NOTE: Nothing here runs at import time. The entry point must call
# configure() explicitly before any worker function is invoked.
_args: argparse.Namespace | None = None


def configure(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and apply CONFIG overrides.

    Must be called once from the process entry point (typically
    manifold_compile.py) BEFORE any worker function or `get_args()` use.

    Calling this more than once is safe but overwrites the previous
    Namespace and re-applies CLI overrides to CONFIG (idempotent for the
    same argv).
    """
    global _args
    from core.cli_args import build_parser
    _args = build_parser().parse_args(argv)
    _apply_cli_overrides(_args)
    return _args


def set_args(ns: argparse.Namespace) -> None:
    """Inject a pre-parsed Namespace.

    Used by spawned worker processes (Windows spawn) where sys.argv in the
    child is not the parent's argv. The parent should pass
    `get_args()` as the 4th element of `initargs` to the Pool.
    """
    global _args
    _args = ns


def get_args() -> argparse.Namespace:
    """Return the active Namespace, raising if the module isn't configured."""
    if _args is None:
        raise RuntimeError(
            "compiler_core has not been configured. Call "
            "compiler_core.configure() from your entry point before using "
            "worker functions, or pass args=... to init_worker() in spawned "
            "worker processes."
        )
    return _args


def _apply_cli_overrides(ns: argparse.Namespace) -> None:
    """Apply CLI flags to CONFIG.

    Extracted verbatim from the previously module-level block; only the
    trigger point moved.
    """
    if getattr(ns, "workers", None):
        CONFIG["num_workers"] = ns.workers

    # Phase 3: CLI overrides for transcode policy.
    if getattr(ns, "image_format", None):
        CONFIG["image_format"] = ns.image_format
    if getattr(ns, "image_quality", None) is not None:
        CONFIG["image_quality"] = ns.image_quality
    if getattr(ns, "target_quality", None) is not None:
        CONFIG["target_quality"] = ns.target_quality
    if getattr(ns, "mask_format", None):
        CONFIG["mask_format"] = ns.mask_format

    # Phase 5: CLI overrides for generation strategies.
    if getattr(ns, "label_strategy", None):
        CONFIG["label_strategy"] = ns.label_strategy
    if getattr(ns, "prompt_strategy", None):
        CONFIG["prompt_strategy"] = ns.prompt_strategy
    if getattr(ns, "mask_strategy", None):
        CONFIG["mask_strategy"] = ns.mask_strategy


# Backwards-compat trap: give a clear migration message instead of a cryptic
# AttributeError when old call sites still touch `compiler_core.args`.
def __getattr__(name: str) -> Any:
    if name in ("args", "parser"):
        raise AttributeError(
            f"compiler_core.{name} no longer exists. Call "
            f"compiler_core.configure() in your entry point and use "
            f"compiler_core.get_args() (or pass args= to init_worker() in "
            f"spawned workers)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Static paths. None of these depend on CLI args — they were already
# constants before the refactor.
INPUT_ROOT = Path("./raw-sets")
OUT_PARENT = Path(META.get("output_folder_name", "../LemGendaryDatasets"))
CATEGORY_MAP_PATH = ROOT / "converters" / "category_map.json"
if not CATEGORY_MAP_PATH.exists():
    CATEGORY_MAP_PATH = ROOT / "category_map.json"

if CATEGORY_MAP_PATH.exists():
    with open(CATEGORY_MAP_PATH, "r", encoding="utf-8") as _f:
        CATEGORY_MAP: dict[str, int] = json.load(_f)
else:
    CATEGORY_MAP = {}

DATASETS_META: dict[str, Any] = YAML_DATA.get("datasets", {})


# ─── Package re-exports ────────────────────────────────────────────────────
from converters import (
    detect_annotations,
    parse_coco,
    parse_matlab,
    parse_parquet,
    parse_safetensors,
    parse_xml,
    parse_yolo,
)
from utils.fs import get_dir_size, remove_empty_dirs
from utils.geometry import convert_bbox_xywh_to_yolo, normalize_points
from utils.hashing import compute_hash
from utils.image import ensure_srgb, is_black_image
from utils.math import get_gaussian_probs
from utils.naming import clean_slug, map_category
from utils.net import download_image
from core.registry import ensure_registry_schema, initialize_registry
from audit.ground_truth import GroundTruthCaches
from audit.vision_audit import VisionAuditor
from audit.dedup import ExactHasher, PerceptualHasher
from audit.reject_log import RejectLog
from formats.base import Sample, Writer, make_writer, parse_also_format
from formats.transcode import ImageTranscoder, KeepFormatError
from formats.webdataset import ShardWriter, WebDatasetWriter
from formats.mds import MDSWriter
from formats.litdata import LitDataWriter
from formats.parquet import ParquetWriter
from formats.directory import DirectorySampleSource, DirectoryWriter
from generators import GenerationResult, LabelGenerator, MaskGenerator, PromptGenerator
from degrade import (
    CompositeProfile,
    DegradationProfile,
    DynamicDegrader,
    composite,
    parse_profile,
)
from runtime.environment import get_device_info


# ─── Globals ────────────────────────────────────────────────────────────────
SENTRY = None
LABELER = None
CAPTIONER = None
CLIP_MANIFOLD = None

_GT_CACHE = GroundTruthCaches()

AUDITOR: VisionAuditor | None = None
EXACT_HASHER: ExactHasher | None = None
PERCEPTUAL_HASHER: PerceptualHasher | None = None

TRANSCODER: ImageTranscoder | None = None

# Phase 5: generators instantiated once per worker in init_worker.
LABEL_GEN: LabelGenerator | None = None
PROMPT_GEN: PromptGenerator | None = None
MASK_GEN: MaskGenerator | None = None


class _Annotation(TypedDict):
    """Shape of every record appended to `annotations` in `process_image`."""
    type: str        # 'bbox' | 'segmentation' | 'pose'
    cls: int
    data: list[Any]


# ─── Task routing ───────────────────────────────────────────────────────────
def detect_task(model_dir_name: str | None) -> str:
    """Route a model key to its task class."""
    if not model_dir_name:
        return "quality"
    name = model_dir_name.lower()

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


DPED_CACHE: set[str] = set()
PHYSICAL_INDEX: set[str] = set()


def _infer_label_strategy(task: str) -> str | None:
    """Map a task to its default label-generation strategy."""
    return {
        "diffusion": "blip_caption",
        "classification": "clip_zeroshot",
        "detection": "yolo_detection",
        "segmentation": "parsenet_segmentation",
    }.get(task)


def _infer_mask_strategy(task: str) -> str | None:
    """Map a task to its default mask-generation strategy."""
    return "parsenet" if task == "segmentation" else None


# ─── Worker bootstrap ───────────────────────────────────────────────────────
def init_worker(
    config: dict[str, Any],
    dped_cache: set[str] | None = None,
    physical_index: set[str] | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    """Per-process worker bootstrap. Loads models, GT caches, device config.

    On Windows spawn the child re-imports this module with a different
    sys.argv; the parent MUST pass `get_args()` as the 4th element of
    `initargs`. If `args` is None, the parent's module state is used — this
    only works in the parent process itself, not in spawned children.
    """
    if args is not None:
        set_args(args)
    a = get_args()

    global SENTRY, CAPTIONER, CLIP_MANIFOLD, DPED_CACHE, PHYSICAL_INDEX, _GT_CACHE
    global AUDITOR, EXACT_HASHER, PERCEPTUAL_HASHER, TRANSCODER
    global LABEL_GEN, MASK_GEN

    if dped_cache:
        DPED_CACHE = dped_cache
    if physical_index:
        PHYSICAL_INDEX = physical_index

    from models.quality_scorer import QualitySentry
    from models.detection import AutoLabeler
    from models.diffusion import CaptionSentry
    from models.encoder import CLIPManifold

    # Workers ignore SIGINT to prevent traceback noise (subprocesses only).
    if os.name == "nt" and multiprocessing.current_process().name != "MainProcess":
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)

    if os.name != "nt" or multiprocessing.current_process().name != "MainProcess":
        try:
            torch.set_num_threads(1)
        except Exception as exc:
            logger.debug("Failed setting torch num threads: %s", exc)

    # Device selection.
    if os.name == "nt" or not torch.cuda.is_available():
        device = "cpu"
    else:
        gpu_count = torch.cuda.device_count()
        if physical_index is not None and gpu_count > 0:
            device = f"cuda:{physical_index % gpu_count}"
        else:
            device = "cuda:0"

    # Phase 2 audit primitives.
    AUDITOR = VisionAuditor(CONFIG)
    EXACT_HASHER = ExactHasher(no_hash=a.no_hash)
    PERCEPTUAL_HASHER = PerceptualHasher(no_hash=a.no_hash)

    # Phase 3 transcoder.
    policy = ImageFormatPolicy(
        format=CONFIG.get("image_format", "webp"),
        quality=int(CONFIG.get("image_quality", 92)),
        target_quality=int(CONFIG.get("target_quality", 95)),
        mask_format=CONFIG.get("mask_format", "webp-lossless"),
    )
    TRANSCODER = ImageTranscoder(policy)

    mission = detect_task(a.model)

    # Phase 5: resolve strategies.
    _label_strategy = CONFIG.get("label_strategy") or _infer_label_strategy(mission)
    _mask_strategy = CONFIG.get("mask_strategy") or _infer_mask_strategy(mission)
    _gen_labeler = None
    if _label_strategy in ("yolo_detection", "parsenet_segmentation"):
        seg_mode = _label_strategy == "parsenet_segmentation"
        try:
            _gen_labeler = get_labeler("segmentation" if seg_mode else mission, device)
        except Exception:
            _gen_labeler = None

    # NIMA quality vetting.
    if mission in ["quality", "classification", "diffusion"] and not a.no_vetting:
        model_type = (
            "aesthetic"
            if mission == "diffusion" or (a.model and "aesthetic" in a.model)
            else "technical"
        )
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", f"nima_{model_type}_best.pth")
        if os.path.exists(model_path):
            try:
                SENTRY = QualitySentry(model_path, model_name=model_type, device=device)
            except Exception as exc:
                logger.warning("Worker failed loading QualitySentry (%s): %s", model_path, exc)

    # Ground truth caches.
    if mission in ["quality", "classification", "diffusion", "restoration"]:
        _GT_CACHE.load(INPUT_ROOT, a.model or "")

    # Diffusion captioning.
    if mission == "diffusion":
        try:
            CAPTIONER = CaptionSentry(device=device)
        except Exception as exc:
            logger.warning("Worker failed loading CaptionSentry: %s", exc)

    # Style manifold (CLIP) — loaded on CPU to conserve VRAM.
    if "clip" in str(config):
        try:
            CLIP_MANIFOLD = CLIPManifold(device="cpu")
        except Exception as exc:
            logger.warning("Worker failed loading CLIPManifold: %s", exc)

    # Phase 5 generators (constructed after all model wrappers are loaded).
    if _label_strategy:
        try:
            LABEL_GEN = LabelGenerator(
                _label_strategy,
                captioner=CAPTIONER,
                clip=CLIP_MANIFOLD,
                labeler=_gen_labeler,
                sentry=SENTRY,
            )
        except Exception as e:
            print(f"[WARN] LabelGenerator init failed: {e}")
            LABEL_GEN = None

    if _mask_strategy:
        try:
            MASK_GEN = MaskGenerator(_mask_strategy, labeler=_gen_labeler)
        except Exception as e:
            print(f"[WARN] MaskGenerator init failed: {e}")
            MASK_GEN = None


def get_labeler(task: str, device: str = "cuda"):
    """Singleton labeler per task type (detection / segmentation / face)."""
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


# ─── Batch worker ───────────────────────────────────────────────────────────
def process_parquet_shard(
    pq_path, prefix, c_slug, start_idx, task, fmt, split_fallback,
    output_root_str, skip_lbl, train_prob, existing_names,
    existing_on_disk, keep_prob, num_rows,
):
    """Process an entire Parquet shard as one worker task."""
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
            results.append({"name": name, "source": c_slug, "task": task,
                            "split": "skipped", "hash": "dropped",
                            "nima_score": 1.0, "size": 0})
            continue

        if name in existing_names or name.lower() in existing_on_disk:
            results.append({"name": name, "source": c_slug, "task": task,
                            "split": "skipped", "hash": "skipped",
                            "nima_score": 1.0, "size": 0})
            continue

        img_bytes = getattr(row, "image", getattr(row, "pixel_values", None))
        if img_bytes is None:
            results.append({"name": name, "source": c_slug, "task": task,
                            "split": "skipped", "hash": "skipped",
                            "nima_score": 1.0, "size": 0})
            continue

        split = "train" if random.random() < train_prob else "val"
        row_dict = {k: getattr(row, k) for k in df.columns}

        if task == "diffusion":
            res = process_diffusion(img_bytes, prefix, c_slug, current_idx, split, output_root_str)
        else:
            res = process_image(img_bytes, prefix, c_slug, current_idx, task, fmt,
                                row_dict, split, output_root_str, skip_lbl)

        if res:
            results.append(res)
        else:
            results.append({"name": name, "source": c_slug, "task": task,
                            "split": "skipped", "hash": "failed",
                            "nima_score": 1.0, "size": 0})
    return results


def batch_worker(tasks: list[tuple[Any, ...]]) -> list[Any]:
    """Execute a list of (func, *args) in one worker call to amortize IPC."""
    results: list[Any] = []
    for task_func, *task_args in tasks:
        try:
            res = task_func(*task_args)
            if task_func.__name__ == "process_parquet_shard":
                results.extend(res)
            else:
                results.append(res)
        except Exception as exc:
            logger.warning("Worker task failed: %s with args %s: %s", getattr(task_func, "__name__", str(task_func)), task_args, exc)
            results.append(None)
    return results


# ─── Audit gate shims (Phase 2) ─────────────────────────────────────────────
def _is_image_valid_for_dataset(img, w, hgt, task, slug) -> bool:
    if AUDITOR is None:
        return True
    return AUDITOR.audit_image(img, task, slug).valid


def _passes_nima_filter(
    task, slug, is_authenticity, nima_score, nima_probs,
    current_threshold, idx,
) -> bool:
    if AUDITOR is None:
        return True
    return AUDITOR.passes_nima(
        task, slug, is_authenticity, nima_score,
        nima_probs, current_threshold, idx,
    )


# ─── Phase 3 write helper ───────────────────────────────────────────────────
def _write_image_phase3(
    *,
    loaded_img: Image.Image | None,
    source_bytes: bytes,
    source_path: Path,
    out_path: Path,
    kind: Literal["image", "target", "mask"],
) -> Path:
    """Write a single image to disk, honoring the Phase 3 transcode policy."""
    if TRANSCODER is not None and TRANSCODER.enabled:
        try:
            img = loaded_img
            if img is None:
                if source_bytes:
                    img = Image.open(io.BytesIO(source_bytes))
                else:
                    img = Image.open(source_path)
            img = ensure_srgb(img)

            encoded, fmt = TRANSCODER.encode(img, kind=kind)
            new_ext = TRANSCODER.extension_for(fmt)
            final_path = out_path.with_suffix(new_ext)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            with open(final_path, "wb") as f:
                f.write(encoded)
            return final_path
        except KeepFormatError as kfe:
            logger.debug("Retaining format for %s per policy: %s", source_path.name, kfe)
        except Exception as e:
            print(f"[WARN] Transcode failed for {source_path.name}: {e}; using copy fallback.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if loaded_img is not None:
        save_fmt = "PNG" if out_path.suffix == ".png" else "JPEG"
        loaded_img.save(out_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
        return out_path
    if source_bytes:
        with open(out_path, "wb") as f:
            f.write(source_bytes)
        return out_path
    try:
        os.link(str(source_path), str(out_path))
        return out_path
    except (OSError, AttributeError) as exc:
        logger.debug("Hardlink creation skipped on %s, falling back to copy: %s", out_path.name, exc)
    try:
        shutil.copy2(str(source_path), str(out_path))
    except (shutil.SameFileError, OSError) as exc:
        logger.debug("Copy fallback error for %s: %s", out_path.name, exc)
    return out_path


# ─── Per-sample processors ──────────────────────────────────────────────────
def process_image(
    img_input, prefix, slug, idx, task, fmt, ann_data, split,
    output_root_str, skip_labeling=False,
):
    """Worker function for parallel processing."""
    a = get_args()  # was: module-global `args`

    img_path: Any = "Unknown"
    nima_score = 1.0
    nima_probs = [0.0] * 10
    nima_probs[0] = 1.0
    w, hgt = 0, 0
    img = None
    p_str = ""
    img_data = b""

    try:
        # ── Input normalization ─────────────────────────────────────────────
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

        if PHYSICAL_INDEX and name.lower() in PHYSICAL_INDEX:
            return {"name": name, "source": slug, "task": task, "split": split,
                    "hash": "skipped", "nima_score": nima_score, "size": 0}

        # ── Restoration target resolution ───────────────────────────────────
        target_img = None
        target_img_path = None

        if task in ["restoration", "super-resolution", "parameter_prediction", "segmentation"]:
            if ann_data:
                row_dict = None
                if isinstance(ann_data, dict):
                    row_dict = ann_data
                elif isinstance(ann_data, tuple) and len(ann_data) == 2:
                    df_sub, _ = ann_data
                    if not df_sub.empty:
                        row_dict = df_sub.iloc[0].to_dict()
                if row_dict:
                    for k in ["target", "sharp", "ground_truth", "gt", "clean",
                              "original", "mask", "masks"]:
                        val = row_dict.get(k)
                        if isinstance(val, bytes):
                            target_img = Image.open(io.BytesIO(val))
                            break
                        elif isinstance(val, str) and val.endswith((".png", ".jpg", ".jpeg")):
                            p = Path(val)
                            if p.exists():
                                target_img_path = str(p)
                            break

            if not target_img and not target_img_path and not isinstance(img_input, (bytes, dict)):
                blur_keys = ["blur", "blurry", "input", "lowres", "lr", "rain",
                             "hazy", "noisy", "degraded", "distorted", "low", "images"]
                sharp_keys = ["sharp", "gt", "ground_truth", "groundtruth", "clean",
                              "clear", "original", "hr", "highres", "target", "norain",
                              "high", "targets", "mask", "masks", "segmentation",
                              "segmentations"]
                p_str = str(img_path).replace("\\", "/")
                parent = img_path.parent

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
                    except OSError as exc:
                        logger.debug("Target sibling search OSError on %s: %s", parent, exc)

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
                                        except ValueError as exc:
                                            logger.debug("Relative path computation failed: %s", exc)
                            except OSError as exc:
                                logger.debug("Ancestor sibling search OSError on %s: %s", ancestor, exc)
                            if target_img_path:
                                break

                if not target_img_path:
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

        # ── Deferred image loading ──────────────────────────────────────────
        needs_stats = (
            (task in ["quality", "diffusion"] and not a.no_vetting)
            or (not skip_labeling)
        )
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
            if AUDITOR is not None:
                audit = AUDITOR.audit_image(img, task, slug)
                if not audit.valid:
                    return None

        # ── NIMA quality gate ───────────────────────────────────────────────
        nima_score = 1.0
        nima_probs = [0.0] * 10
        nima_probs[0] = 1.0

        if "ava" in slug and _GT_CACHE.ava:
            try:
                img_id = int(img_path.stem)
                if img_id in _GT_CACHE.ava:
                    votes = _GT_CACHE.ava[img_id]
                    nima_probs = [votes[f"vote_{i}"] for i in range(1, 11)]
                    nima_score = sum(p * (i + 1) for i, p in enumerate(nima_probs))
            except (ValueError, KeyError) as exc:
                logger.debug("AVA score extraction fallback for %s: %s", img_path.name, exc)

        elif "aadb" in slug and _GT_CACHE.aadb:
            try:
                raw_score = _GT_CACHE.aadb.get(img_path.name)
                if raw_score is not None:
                    nima_score = (raw_score * 9.0) + 1.0
                    nima_probs = get_gaussian_probs(nima_score)
            except (TypeError, ValueError) as exc:
                logger.debug("AADB score extraction fallback for %s: %s", img_path.name, exc)

        elif "laion" in slug:
            try:
                if isinstance(ann_data, dict):
                    val = ann_data.get("aesthetic_score", ann_data.get("score", 6.5))
                    nima_score = float(val) if val is not None else 6.5
                    nima_probs = get_gaussian_probs(nima_score)
                elif fmt == "parquet" and ann_data:
                    df_subset, mapping = cast("tuple[Any, dict[str, Any]]", ann_data)
                    col = mapping.get("aesthetic_score", "aesthetic_score")
                    if col in df_subset.columns:
                        nima_score = float(df_subset[col].iloc[0])
                        nima_probs = get_gaussian_probs(nima_score)
            except (TypeError, ValueError, KeyError) as exc:
                logger.debug("LAION score extraction fallback for %s: %s", img_path.name, exc)

        elif _GT_CACHE.tid and img_path.name.lower() in _GT_CACHE.tid:
            try:
                raw_score = _GT_CACHE.tid.get(img_path.name.lower())
                if raw_score is not None:
                    nima_score = raw_score
                    nima_probs = get_gaussian_probs(nima_score)
            except (TypeError, ValueError) as exc:
                logger.debug("TID score extraction fallback for %s: %s", img_path.name, exc)

        is_authenticity = "authentic" in prefix.lower()
        if is_authenticity:
            parent_name = img_path.parent.name.lower()
            file_name = img_path.name.lower()
            if parent_name in ["sut-project", "ai-generated-images-vs-real-images",
                               "real vs fake faces", "raw-sets"]:
                parent_name = ""
            path_str = f"{parent_name}/{file_name}".lower()
            if any(k in path_str for k in ["sut-project", "midjourney", "diffusion",
                                           "ai", "fake", "gan", "generated"]):
                nima_probs = [0.0] * 10
                nima_probs[0] = 1.0
                nima_score = 1.0
            elif any(k in path_str for k in ["ffhq", "div2k", "celebahq", "human",
                                             "real", "afhq", "nature"]):
                nima_probs = [0.0] * 10
                nima_probs[9] = 1.0
                nima_score = 10.0

        if nima_probs[0] == 1.0 and task in ["quality", "diffusion"] and not is_authenticity:
            if SENTRY:
                nima_score, nima_probs = SENTRY.score(img, return_probs=True)

        current_threshold = 5.5 if task == "diffusion" else CONFIG["nima_threshold"]
        if not _passes_nima_filter(task, slug, is_authenticity, nima_score,
                                   nima_probs, current_threshold, idx):
            return None

        # ── Hash + output writes ────────────────────────────────────────────
        hash_target = img_data if isinstance(img_input, (bytes, dict)) else img_path
        h = EXACT_HASHER.hash(hash_target) if (CONFIG["enable_dedup"] and EXACT_HASHER) else None

        ph: str | None = None
        if CONFIG.get("enable_perceptual_dedup") and PERCEPTUAL_HASHER and img is not None:
            p_hashes = PERCEPTUAL_HASHER.hash(img)
            if p_hashes is not None:
                ph = f"{p_hashes[0]}:{p_hashes[1]}"

        is_already_on_disk = PHYSICAL_INDEX and name.lower() in PHYSICAL_INDEX
        is_clean_only = ("parsenet" in slug.lower() or "codeformer" in slug.lower()) and task == "restoration"
        if is_clean_only and not target_img_path and isinstance(img_input, (str, Path)):
            target_img_path = str(img_path)

        if not is_already_on_disk and not is_clean_only:
            out_img_path = _write_image_phase3(
                loaded_img=img,
                source_bytes=img_data,
                source_path=img_path,
                out_path=out_img_path,
                kind="image",
            )

        if task in ["restoration", "super-resolution", "segmentation"]:
            if target_img_path:
                out_tgt_path = _write_image_phase3(
                    loaded_img=None,
                    source_bytes=b"",
                    source_path=Path(target_img_path),
                    out_path=out_tgt_path,
                    kind="target",
                )
            elif target_img:
                out_tgt_path = _write_image_phase3(
                    loaded_img=target_img,
                    source_bytes=b"",
                    source_path=img_path,
                    out_path=out_tgt_path,
                    kind="target",
                )
            else:
                if is_clean_only:
                    out_tgt_path = _write_image_phase3(
                        loaded_img=img,
                        source_bytes=img_data,
                        source_path=img_path,
                        out_path=out_tgt_path,
                        kind="image",
                    )
                elif task == "segmentation" and MASK_GEN is not None and img is not None:
                    mask_result = MASK_GEN.generate(img, context={"task": task, "slug": slug})
                    if mask_result.value is not None:
                        out_tgt_path = _write_image_phase3(
                            loaded_img=mask_result.value,
                            source_bytes=b"",
                            source_path=img_path,
                            out_path=out_tgt_path,
                            kind="mask",
                        )
                else:
                    try:
                        os.link(str(out_img_path), str(out_tgt_path))
                    except (OSError, AttributeError):
                        try:
                            shutil.copy2(out_img_path, out_tgt_path)
                        except OSError as exc:
                            logger.debug("Target copy fallback error for %s: %s", out_tgt_path.name, exc)

        elif task == "parameter_prediction":
            if target_img_path:
                out_tgt_path = _write_image_phase3(
                    loaded_img=None,
                    source_bytes=b"",
                    source_path=Path(target_img_path),
                    out_path=out_tgt_path,
                    kind="target",
                )
            elif target_img:
                out_tgt_path = _write_image_phase3(
                    loaded_img=target_img,
                    source_bytes=b"",
                    source_path=img_path,
                    out_path=out_tgt_path,
                    kind="target",
                )
            else:
                try:
                    os.link(str(out_img_path), str(out_tgt_path))
                except (OSError, AttributeError):
                    try:
                        shutil.copy2(str(out_img_path), str(out_tgt_path))
                    except OSError as exc:
                        logger.debug("Param prediction copy fallback error for %s: %s", out_tgt_path.name, exc)

        # ── Annotation dispatch ─────────────────────────────────────────────
        annotations: list[_Annotation] = []

        if fmt == "coco" and ann_data is not None:
            for a_ in ann_data:
                cls = map_category(str(a_["category_id"]), prefix, CATEGORY_MAP)
                if "keypoints" in a_ and a_["keypoints"]:
                    kpts = normalize_points(a_["keypoints"], w, hgt, stride=3)
                    annotations.append({"type": "pose", "cls": cls, "data": a_["bbox"] + kpts})
                elif "segmentation" in a_ and a_["segmentation"]:
                    poly_raw = (a_["segmentation"][0]
                                if isinstance(a_["segmentation"], list) and len(a_["segmentation"]) > 0
                                else [])
                    if poly_raw:
                        poly = normalize_points(poly_raw, w, hgt, stride=2)
                        annotations.append({"type": "segmentation", "cls": cls, "data": poly})
                else:
                    annotations.append({"type": "bbox", "cls": cls, "data": a_["bbox"]})

        elif fmt == "parquet" and ann_data and not isinstance(ann_data, dict):
            df_subset, mapping = cast("tuple[Any, dict[str, Any]]", ann_data)
            for _, row in df_subset.iterrows():
                cls = map_category(row[mapping.get("class", "class")], prefix, CATEGORY_MAP)
                if mapping.get("segmentation") in row and row[mapping.get("segmentation")]:
                    poly = normalize_points(row[mapping.get("segmentation")], w, hgt, stride=2)
                    annotations.append({"type": "segmentation", "cls": cls, "data": poly})
                elif mapping.get("keypoints") in row and row[mapping.get("keypoints")]:
                    kpts = normalize_points(row[mapping.get("keypoints")], w, hgt, stride=3)
                    annotations.append({"type": "pose", "cls": cls, "data": [0, 0, 0, 0] + kpts})
                else:
                    bbox = [
                        row[mapping.get("xmin", "xmin")],
                        row[mapping.get("ymin", "ymin")],
                        row[mapping.get("width", "width")],
                        row[mapping.get("height", "height")],
                    ]
                    annotations.append({"type": "bbox", "cls": cls, "data": bbox})

        elif fmt == "matlab" and ann_data:
            for entry in ann_data:
                try:
                    cls = map_category(entry["class"], prefix, CATEGORY_MAP)
                    annotations.append({"type": "bbox", "cls": cls, "data": entry["bbox"]})
                except (KeyError, TypeError) as exc:
                    logger.debug("Matlab annotation parsing error for entry: %s", exc)

        elif fmt == "xml" and ann_data:
            if isinstance(ann_data, (str, Path)):
                xml_anns = parse_xml(ann_data)
                for a_ in xml_anns:
                    cls = map_category(a_["class"], prefix, CATEGORY_MAP)
                    a_bbox = cast("list[Any]", a_["bbox"])
                    annotations.append({"type": "bbox", "cls": cls, "data": a_bbox})

        elif fmt == "yolo" and ann_data:
            if isinstance(ann_data, (str, Path)):
                yolo_anns = parse_yolo(ann_data, w, hgt)
                for a_ in yolo_anns:
                    if task == "pose":
                        cls = map_category("0", prefix, CATEGORY_MAP)
                    else:
                        cls = map_category(a_["class"], prefix, CATEGORY_MAP)
                    a_bbox = cast("list[Any]", a_["bbox"])
                    if "keypoints" in a_ and a_["keypoints"]:
                        a_kpts = cast("list[Any]", a_["keypoints"])
                        annotations.append({"type": "pose", "cls": cls, "data": a_bbox + a_kpts})
                    else:
                        annotations.append({"type": "bbox", "cls": cls, "data": a_bbox})

        elif fmt == "npz" and ann_data:
            if isinstance(ann_data, (str, Path)):
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
                                l_eye[0], l_eye[1], r_eye[0], r_eye[1],
                                nose[0], nose[1], l_mouth[0], l_mouth[1],
                                r_mouth[0], r_mouth[1],
                            ]
                            cls = map_category("0", prefix, CATEGORY_MAP)
                            annotations.append({
                                "type": "pose", "cls": cls,
                                "data": [x_min, y_min, bbox_w, bbox_h] + kpts,
                            })
                except (OSError, KeyError, ValueError) as e:
                    print(f"Error parsing NPZ {ann_data}: {e}")

        elif fmt == "safetensors" and isinstance(ann_data, dict):
            tags: list[str] = []
            if "ss_tag_frequency" in ann_data:
                try:
                    freqs = json.loads(str(ann_data["ss_tag_frequency"]))
                    for bucket in freqs.values():
                        tags.extend(bucket.keys())
                except (json.JSONDecodeError, AttributeError) as exc:
                    logger.debug("Failed parsing ss_tag_frequency: %s", exc)
            if not tags and "ss_datasets" in ann_data:
                try:
                    ds_info = json.loads(str(ann_data["ss_datasets"]))
                    for ds in ds_info:
                        if "tag_frequency" in ds:
                            tags.extend(ds["tag_frequency"].keys())
                except (json.JSONDecodeError, AttributeError) as exc:
                    logger.debug("Failed parsing ss_datasets: %s", exc)
            if tags:
                unique_tags = list(set(tags))[:20]
                for tag in unique_tags:
                    cls = map_category(tag, prefix, CATEGORY_MAP)
                    annotations.append({
                        "type": "bbox", "cls": cls,
                        "data": [0.0, 0.0, 1.0, 1.0],
                    })

        # ── Auto-labeling fallback (Phase 5 dispatch) ───────────────────────
        is_autolabeled = False
        if (not annotations
                and task not in ["quality", "classification"]
                and not a.no_labeling
                and not skip_labeling
                and img is not None):
            if LABEL_GEN is not None:
                gen_result = LABEL_GEN.generate(
                    img, context={"task": task, "slug": slug, "path": str(img_path)}
                )
                if gen_result.kind == "label" and isinstance(gen_result.value, dict):
                    gen_annotations = gen_result.value.get("annotations")
                    if gen_annotations:
                        annotations = cast("list[_Annotation]", gen_annotations)
                        is_autolabeled = True
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                labeler = get_labeler(task, device)
                annotations = cast("list[_Annotation]", labeler.predict(img))
                if annotations:
                    is_autolabeled = True

        # ── Write label file ────────────────────────────────────────────────
        label_file_path = Path(output_root_str) / "labels" / split / f"{name}.txt"
        has_annotations = len(annotations) > 0 or task in ["quality", "classification"]

        if not skip_labeling or has_annotations:
            with open(label_file_path, "w", encoding="utf-8") as f:
                if task == "quality":
                    f.write(" ".join(f"{p:.6f}" for p in nima_probs) + "\n")
                elif task == "classification":
                    class_label: Any = 1
                    if isinstance(ann_data, dict) and "label" in ann_data:
                        class_label = ann_data["label"]
                    elif (isinstance(ann_data, tuple) and len(ann_data) == 2
                          and isinstance(ann_data[1], dict)):
                        df_subset, mapping = cast("tuple[Any, dict[str, Any]]", ann_data)
                        lbl_col = mapping.get("label", "label")
                        if lbl_col in df_subset.columns:
                            class_label = df_subset.iloc[0][lbl_col]
                    else:
                        p_lower = str(img_path).lower()
                        if any(k in p_lower for k in ["fake", "ai", "synthetic",
                                                      "nsfw", "porn", "explicit"]):
                            class_label = 0
                    if isinstance(class_label, float):
                        if not pd.isna(class_label):
                            class_label = int(class_label)
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
                            f.write(f"{cls} {' '.join(map(str, yolo_box))} "
                                    f"{' '.join(map(str, data[4:]))}\n")

        size_bytes = 0
        try:
            if out_img_path.exists():
                size_bytes += out_img_path.stat().st_size
        except OSError as exc:
            logger.debug("Failed reading file size for %s: %s", out_img_path, exc)

        return {
            "name": name, "source": slug, "task": task, "split": split,
            "hash": h, "perceptual_hash": ph,
            "nima_score": round(nima_score, 3), "is_autolabeled": is_autolabeled,
            "has_segmentation": any(a_["type"] == "segmentation" for a_ in annotations),
            "has_pose": any(a_["type"] == "pose" for a_ in annotations),
            "label_path": str(label_file_path.resolve()),
            "path": str(out_img_path.resolve()),
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

        nima_score: Any = 10.0
        if SENTRY:
            nima_score = SENTRY.score(img)
            if nima_score < CONFIG["nima_threshold"]:
                return None

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

        h = EXACT_HASHER.hash(img) if (CONFIG["enable_dedup"] and EXACT_HASHER) else None
        ph: str | None = None
        if CONFIG.get("enable_perceptual_dedup") and PERCEPTUAL_HASHER:
            p_hashes = PERCEPTUAL_HASHER.hash(img)
            if p_hashes is not None:
                ph = f"{p_hashes[0]}:{p_hashes[1]}"

        name = f"{prefix}_{idx:09d}"

        if TRANSCODER is not None and TRANSCODER.enabled:
            img_bytes, _fmt = TRANSCODER.encode(img, kind="image")
        else:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            img_bytes = buffer.getvalue()

        latent_blob = None
        if clip_latent:
            latent_blob = sqlite3.Binary(np.array(clip_latent).astype(np.float32).tobytes())

        nima_val = float(nima_score[0]) if isinstance(nima_score, (tuple, list)) else float(nima_score)
        return {
            "name": name, "source": slug, "task": "diffusion", "split": split,
            "hash": h, "perceptual_hash": ph,
            "nima_score": round(nima_val, 3),
            "caption": caption, "style_tag": style_tag, "clip_latent": latent_blob,
            "img_bytes": img_bytes, "size": len(img_bytes),
        }
    except Exception as e:
        safe_path = "virtual_bytes" if isinstance(img_path, (bytes, dict)) else img_path
        print(f"[ERROR] Error processing diffusion sample {safe_path}: {e}")
        return None


# ─── ORCHESTRATOR ───────────────────────────────────────────────────────────
# The orchestration body lives in manifold_compile.py.