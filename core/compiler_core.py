"""
LemGendary Dataset Compiler — Core Coordinator.

Public facade of the compile-time pipeline. The implementation lives in:
  - core.compiler_config   CONFIG, unified_data.yaml metadata, CLI state
  - core.compiler_worker   worker bootstrap (init_worker), task routing,
                           per-process model state and image writes
  - core.sample_jobs       ImageJob / ShardJob task payloads
  - core.sample_processor  per-image processing (process_image)
  - core.sample_diffusion  text-image processing (process_diffusion)
  - core.sample_batch      batch dispatch (batch_worker / process_parquet_shard)
  - core.sample_targets, core.sample_scoring, core.sample_annotations
                           target lookup, NIMA scores, annotation parsing

Re-exports the utility, audit, registry, converter, transcode, format, and
generator symbols extracted in Phases 1.4 through 5.0 so existing call sites
in manifold_compile.py keep working unchanged.

Import-order contract
---------------------
Runtime env vars (TORCH_CUDA_ARCH_LIST, CUDA_FORCE_PTX_JIT,
PYTORCH_CUDA_ALLOC_CONF) must be set before torch is imported, because torch
reads them when it loads. This module therefore never imports torch
directly: it takes it from runtime.environment.load_torch(), which applies
the environment first.

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

from typing import Any

from audit.dedup import ExactHasher, PerceptualHasher
from audit.ground_truth import GroundTruthCaches
from audit.reject_log import RejectLog
from audit.vision_audit import VisionAuditor
from converters import (
    detect_annotations,
    parse_coco,
    parse_matlab,
    parse_parquet,
    parse_safetensors,
    parse_xml,
    parse_yolo,
)
from core.compiler_config import (
    CATEGORY_MAP,
    CATEGORY_MAP_PATH,
    CONFIG,
    CONFIG_PATH,
    DATASETS_META,
    DEFAULT_CONFIG,
    INPUT_ROOT,
    META,
    OUT_PARENT,
    VERSION,
    YAML_DATA,
    configure,
    get_args,
    set_args,
)
from core.compiler_worker import detect_task, get_labeler, init_worker
from core.config_schema import ImageFormatPolicy
from core.registry import ensure_registry_schema, initialize_registry
from core.sample_batch import batch_worker, process_parquet_shard
from core.sample_diffusion import process_diffusion
from core.sample_jobs import (
    ExistingSamples,
    ImageJob,
    SampleKey,
    ShardJob,
    SourceContext,
)
from core.sample_processor import process_image
from degrade import (
    CompositeProfile,
    DegradationProfile,
    DynamicDegrader,
    composite,
    parse_profile,
)
from formats.base import Sample, Writer, parse_also_format
from formats.directory import DirectorySampleSource, DirectoryWriter
from formats.factory import make_writer
from formats.litdata import LitDataWriter
from formats.mds import MDSWriter
from formats.parquet import ParquetWriter
from formats.transcode import ImageTranscoder, KeepFormatError
from formats.webdataset import ShardWriter, WebDatasetWriter
from generators import GenerationResult, LabelGenerator, MaskGenerator, PromptGenerator
from runtime.environment import bootstrap_runtime, get_device_info
from utils.fs import get_dir_size, remove_empty_dirs
from utils.geometry import convert_bbox_xywh_to_yolo, normalize_points
from utils.hashing import compute_hash
from utils.image import ensure_srgb, is_black_image
from utils.math import get_gaussian_probs
from utils.naming import clean_slug, map_category
from utils.net import download_image

# ─── Public surface ─────────────────────────────────────────────────────────
__all__ = [
    # Config constants (core.compiler_config)
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
    # Core API (core.compiler_worker / core.sample_*)
    "detect_task", "get_labeler",
    "init_worker", "batch_worker",
    "process_image", "process_diffusion", "process_parquet_shard",
    # Task payloads (core.sample_jobs)
    "ImageJob", "ShardJob", "SampleKey", "SourceContext", "ExistingSamples",
]



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
