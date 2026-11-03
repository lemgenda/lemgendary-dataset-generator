"""Per-process worker state for the compiler: model loading, audit gates and image writes.

``init_worker`` runs once in every pool process (and in the parent for
in-process runs). It fills the module-level ``WORKER`` state that the
per-sample processors in ``core.sample_processor`` read.
"""

import argparse
import io
import logging
import multiprocessing
import os
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageFile

from audit.dedup import ExactHasher, PerceptualHasher
from audit.ground_truth import GroundTruthCaches
from audit.vision_audit import VisionAuditor
from core.compiler_config import CONFIG, DATASETS_META, INPUT_ROOT, get_args, set_args
from core.config_schema import ImageFormatPolicy
from core.error_types import MODEL_LOAD_ERRORS, SAMPLE_PROCESSING_ERRORS
from core.lazy_imports import load_model_module
from formats.transcode import ImageTranscoder, KeepFormatError
from generators import LabelGenerator, MaskGenerator
from runtime.environment import load_torch
from utils.image import ensure_srgb

torch = load_torch()

logger = logging.getLogger(__name__)

setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
Image.MAX_IMAGE_PIXELS = None

GT_CACHE = GroundTruthCaches()

ImageKind = Literal["image", "target", "mask"]


@dataclass
class WorkerModels:
    """The torch-backed model wrappers a worker loads for vetting and labeling."""

    sentry: Any = None
    captioner: Any = None
    clip_manifold: Any = None
    labelers: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditTools:
    """The Phase 2 auditor and hashers and the Phase 3 transcoder."""

    auditor: VisionAuditor | None = None
    exact_hasher: ExactHasher | None = None
    perceptual_hasher: PerceptualHasher | None = None
    transcoder: ImageTranscoder | None = None


@dataclass
class Generators:
    """The Phase 5 label and mask generators."""

    label_gen: LabelGenerator | None = None
    mask_gen: MaskGenerator | None = None


@dataclass
class WorkerState:
    """Models, audit primitives and caches that init_worker loads once per process."""

    models: WorkerModels = field(default_factory=WorkerModels)
    audit: AuditTools = field(default_factory=AuditTools)
    generators: Generators = field(default_factory=Generators)
    dped_cache: set[str] = field(default_factory=set)
    physical_index: set[str] = field(default_factory=set)


WORKER = WorkerState()

_TASK_PATTERNS: list[tuple[list[str], str]] = [
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


# ─── Task routing ───────────────────────────────────────────────────────────
def _task_override(name: str) -> str | None:
    """Return the registry's ``task_override`` for a dataset key or name, if any."""
    for ds_key, ds_cfg in DATASETS_META.items():
        if ds_key.lower() == name or (ds_cfg.get("name", "").lower() == name):
            override = ds_cfg.get("task_override")
            if override:
                return override
    return None


def detect_task(model_dir_name: str | None) -> str:
    """Route a model key to its task class."""
    if not model_dir_name:
        return "quality"
    name = model_dir_name.lower()

    override = _task_override(name)
    if override:
        return override

    for patterns, t_name in _TASK_PATTERNS:
        if any(k in name for k in patterns):
            return t_name
    return "detection"


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
def _configure_process() -> str:
    """Silence SIGINT in children, pin torch threads and return the worker's device."""
    is_main = multiprocessing.current_process().name == "MainProcess"
    # Workers ignore SIGINT to prevent traceback noise (subprocesses only).
    if os.name == "nt" and not is_main:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)

    if os.name != "nt" or not is_main:
        try:
            torch.set_num_threads(1)
        except RuntimeError as exc:
            logger.debug("Failed setting torch num threads: %s", exc)

    if os.name == "nt" or not torch.cuda.is_available():
        return "cpu"
    gpu_count = torch.cuda.device_count()
    # Spread worker processes across GPUs round-robin by process id.
    return f"cuda:{os.getpid() % gpu_count}" if gpu_count > 1 else "cuda:0"


def _load_audit_primitives(a: argparse.Namespace) -> None:
    """Build the Phase 2 auditor and hashers and the Phase 3 transcoder."""
    WORKER.audit.auditor = VisionAuditor(CONFIG)
    WORKER.audit.exact_hasher = ExactHasher(no_hash=a.no_hash)
    WORKER.audit.perceptual_hasher = PerceptualHasher(no_hash=a.no_hash)
    policy = ImageFormatPolicy(
        format=CONFIG.get("image_format", "webp"),
        quality=int(CONFIG.get("image_quality", 92)),
        target_quality=int(CONFIG.get("target_quality", 95)),
        mask_format=CONFIG.get("mask_format", "webp-lossless"),
    )
    WORKER.audit.transcoder = ImageTranscoder(policy)


def _load_generation_labeler(label_strategy: str | None, mission: str, device: str) -> Any:
    """Load the detection/segmentation labeler a label strategy needs, or None."""
    if label_strategy not in ("yolo_detection", "parsenet_segmentation"):
        return None
    seg_mode = label_strategy == "parsenet_segmentation"
    try:
        return get_labeler("segmentation" if seg_mode else mission, device)
    except MODEL_LOAD_ERRORS as exc:
        logger.warning("Worker failed loading %s labeler: %s", label_strategy, exc)
        return None


def _load_quality_sentry(a: argparse.Namespace, mission: str, device: str) -> None:
    """Load the NIMA QualitySentry used for quality vetting, when its weights exist."""
    if mission not in ["quality", "classification", "diffusion"] or a.no_vetting:
        return
    model_type = (
        "aesthetic"
        if mission == "diffusion" or (a.model and "aesthetic" in a.model)
        else "technical"
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", f"nima_{model_type}_best.pth")
    if not os.path.exists(model_path):
        return
    try:
        WORKER.models.sentry = load_model_module("quality_scorer").QualitySentry(
            model_path, model_name=model_type, device=device
        )
    except MODEL_LOAD_ERRORS as exc:
        logger.warning("Worker failed loading QualitySentry (%s): %s", model_path, exc)


def _load_vision_models(config: dict[str, Any], mission: str, device: str) -> None:
    """Load the diffusion captioner and the CLIP style manifold when the run needs them."""
    if mission == "diffusion":
        try:
            WORKER.models.captioner = load_model_module("diffusion").CaptionSentry(device=device)
        except MODEL_LOAD_ERRORS as exc:
            logger.warning("Worker failed loading CaptionSentry: %s", exc)

    # Style manifold (CLIP) — loaded on CPU to conserve VRAM.
    if "clip" in str(config):
        try:
            WORKER.models.clip_manifold = load_model_module("encoder").CLIPManifold(device="cpu")
        except MODEL_LOAD_ERRORS as exc:
            logger.warning("Worker failed loading CLIPManifold: %s", exc)


def _build_generators(label_strategy: str | None, mask_strategy: str | None,
                      labeler: Any) -> None:
    """Construct the Phase 5 label and mask generators after the model wrappers load."""
    if label_strategy:
        try:
            WORKER.generators.label_gen = LabelGenerator(
                label_strategy,
                captioner=WORKER.models.captioner,
                clip=WORKER.models.clip_manifold,
                labeler=labeler,
                sentry=WORKER.models.sentry,
            )
        except MODEL_LOAD_ERRORS as e:
            logger.warning("LabelGenerator init failed: %s", e)
            WORKER.generators.label_gen = None

    if mask_strategy:
        try:
            WORKER.generators.mask_gen = MaskGenerator(mask_strategy, labeler=labeler)
        except MODEL_LOAD_ERRORS as e:
            logger.warning("MaskGenerator init failed: %s", e)
            WORKER.generators.mask_gen = None


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

    if dped_cache:
        WORKER.dped_cache = dped_cache
    if physical_index:
        WORKER.physical_index = physical_index

    for module_name in ("quality_scorer", "diffusion", "encoder"):
        load_model_module(module_name)

    device = _configure_process()
    _load_audit_primitives(a)

    mission = detect_task(a.model)
    label_strategy = CONFIG.get("label_strategy") or _infer_label_strategy(mission)
    mask_strategy = CONFIG.get("mask_strategy") or _infer_mask_strategy(mission)
    gen_labeler = _load_generation_labeler(label_strategy, mission, device)

    _load_quality_sentry(a, mission, device)
    if mission in ["quality", "classification", "diffusion", "restoration"]:
        GT_CACHE.load(INPUT_ROOT, a.model or "")
    _load_vision_models(config, mission, device)
    _build_generators(label_strategy, mask_strategy, gen_labeler)


def get_labeler(task: str, device: str = "cuda"):
    """Singleton labeler per task type (detection / segmentation / face)."""
    if task not in WORKER.models.labelers:
        if task == "face_detection":
            mode = "face_landmarks"
        else:
            mode = "segmentation" if "seg" in task else "detection"
        WORKER.models.labelers[task] = load_model_module("detection").AutoLabeler(
            mode=mode, device=device
        )
    return WORKER.models.labelers[task]


# ─── Phase 3 write helpers ──────────────────────────────────────────────────
def _transcode_to(transcoder: ImageTranscoder, img: Image.Image, out_path: Path,
                  kind: ImageKind) -> Path:
    """Encode an image with the Phase 3 transcoder and write it next to ``out_path``."""
    encoded, fmt = transcoder.encode(ensure_srgb(img), kind=kind)
    final_path = out_path.with_suffix(transcoder.extension_for(fmt))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_path, "wb") as f:
        f.write(encoded)
    return final_path


def link_or_copy(source: Path, dest: Path, what: str) -> None:
    """Hard-link ``source`` to ``dest``, falling back to a copy; failures are logged."""
    try:
        os.link(str(source), str(dest))
        return
    except (OSError, AttributeError) as exc:
        logger.debug("Hardlink creation skipped on %s, falling back to copy: %s", dest.name, exc)
    try:
        shutil.copy2(str(source), str(dest))
    except OSError as exc:
        logger.debug("%s copy fallback error for %s: %s", what, dest.name, exc)


def write_image(
    *,
    loaded_img: Image.Image | None,
    source_bytes: bytes,
    source_path: Path,
    out_path: Path,
    kind: ImageKind,
) -> Path:
    """Write a single image to disk, honoring the Phase 3 transcode policy."""
    transcoder = WORKER.audit.transcoder
    if transcoder is not None and transcoder.enabled:
        try:
            img = loaded_img
            if img is None:
                if source_bytes:
                    img = Image.open(io.BytesIO(source_bytes))
                else:
                    img = Image.open(source_path)
            return _transcode_to(transcoder, img, out_path, kind)
        except KeepFormatError as kfe:
            logger.debug("Retaining format for %s per policy: %s", source_path.name, kfe)
        except SAMPLE_PROCESSING_ERRORS as e:
            logger.warning("Transcode failed for %s: %s; using copy fallback.", source_path.name, e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if loaded_img is not None:
        save_fmt = "PNG" if out_path.suffix == ".png" else "JPEG"
        loaded_img.save(out_path, save_fmt, quality=95 if save_fmt == "JPEG" else None)
        return out_path
    if source_bytes:
        with open(out_path, "wb") as f:
            f.write(source_bytes)
        return out_path
    link_or_copy(source_path, out_path, "Image")
    return out_path
