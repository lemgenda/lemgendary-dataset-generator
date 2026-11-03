"""Per-sample processors run inside the compiler's worker pool.

``process_image`` turns one source image (a path, raw bytes or a Parquet
``{"bytes": ...}`` cell) into the manifold's image, target and label files
and returns its registry record. ``process_diffusion`` does the same for
text-image samples. ``batch_worker`` runs a list of such calls in one task.
"""

import io
import logging
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from PIL import Image

from core.compiler_config import CONFIG, get_args
from core.compiler_worker import WORKER, get_labeler, link_or_copy, torch, write_image
from core.error_types import PARQUET_READ_ERRORS, SAMPLE_PROCESSING_ERRORS
from core.sample_annotations import (
    Annotation,
    AnnotationSource,
    LabelContent,
    label_lines,
    parse_annotations,
    write_label_file,
)
from core.sample_scoring import (
    NimaScore,
    ScoreSource,
    authenticity_score,
    ground_truth_score,
    model_score,
)
from core.sample_targets import resolve_target
from utils.image import ensure_srgb, is_black_image

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".npy")


@dataclass
class _ImageSample:
    """One input image and the output paths derived from it."""

    slug: str
    task: str
    split: str
    output_root: Path
    name: str
    img_path: Path
    is_virtual: bool
    img_data: bytes = b""
    img: Any = None
    dims: tuple[int, int] = (0, 0)

    @property
    def ext(self) -> str:
        """Output extension: the source's when it is a known image type, else ``.jpg``."""
        ext = ".jpg" if self.is_virtual else self.img_path.suffix.lower()
        return ext if ext in _IMAGE_EXTS else ".jpg"

    def image_out_path(self) -> Path:
        """Where the input image is written (``targets/``/``masks/`` sources keep their role)."""
        parts_ok = not self.is_virtual and len(self.img_path.parts) >= 3
        source_parent = self.img_path.parent.parent.name.lower() if parts_ok else "images"
        out_dir = source_parent if source_parent in ["targets", "masks"] else "images"
        return self.output_root / out_dir / self.split / f"{self.name}{self.ext}"

    def target_out_path(self) -> Path:
        """Where the paired target (or segmentation mask) is written."""
        tgt_dir = "masks" if self.task == "segmentation" else "targets"
        return self.output_root / tgt_dir / self.split / f"{self.name}{self.ext}"

    def skip_record(self) -> dict[str, Any]:
        """Registry record for a sample that is already on disk."""
        return {"name": self.name, "source": self.slug, "task": self.task, "split": self.split,
                "hash": "skipped", "nima_score": 1.0, "size": 0}


def _open_sample(img_input: Any, slug: str, idx: int) -> tuple[Path, bytes, Any] | None:
    """Return ``(path, bytes, image)`` for the input; None when a bytes cell is empty."""
    if not isinstance(img_input, (bytes, dict)):
        return Path(str(img_input)), b"", None
    if isinstance(img_input, dict) and "bytes" in img_input:
        img_data = img_input["bytes"]
    else:
        img_data = img_input if isinstance(img_input, bytes) else b""
    if img_data is None:
        return None
    return Path(f"virtual_{slug}_{idx:09d}.jpg"), img_data, Image.open(io.BytesIO(img_data))


def _load_pixels(sample: _ImageSample) -> Any:
    """Decode the sample into an sRGB PIL image (``.npy`` arrays are converted)."""
    if sample.is_virtual:
        return ensure_srgb(Image.open(io.BytesIO(sample.img_data)))
    if sample.img_path.suffix.lower() != ".npy":
        return ensure_srgb(Image.open(sample.img_path))
    data = np.load(sample.img_path)
    if data.ndim == 3 and data.shape[0] in [1, 3]:
        data = data.transpose(1, 2, 0)
    if data.dtype in [np.float32, np.float64]:
        data = (data * 255).clip(0, 255).astype(np.uint8)
    return ensure_srgb(Image.fromarray(data))


def _passes_nima_gate(sample: _ImageSample, prefix: str, idx: int, nima: NimaScore) -> bool:
    """Apply the auditor's NIMA threshold (5.5 for diffusion, CONFIG otherwise)."""
    if WORKER.auditor is None:
        return True
    is_authenticity = "authentic" in prefix.lower()
    threshold = 5.5 if sample.task == "diffusion" else CONFIG["nima_threshold"]
    return WORKER.auditor.passes_nima(
        sample.task, sample.slug, is_authenticity, nima.score, nima.probs, threshold, idx,
    )


def _score_sample(sample: _ImageSample, prefix: str, fmt: str, ann_data: Any) -> NimaScore:
    """Resolve the sample's NIMA score from ground truth, path keywords or the model."""
    nima = ground_truth_score(ScoreSource(sample.slug, fmt, ann_data, sample.img_path))
    is_authenticity = "authentic" in prefix.lower()
    if is_authenticity:
        nima = authenticity_score(sample.img_path, nima)
    if nima.probs[0] == 1.0 and sample.task in ["quality", "diffusion"] and not is_authenticity:
        nima = model_score(sample.img, nima)
    return nima


def _hashes(sample: _ImageSample) -> tuple[Any, str | None]:
    """Exact hash (when dedup is on) and perceptual hash (when perceptual dedup is on)."""
    hash_target = sample.img_data if sample.is_virtual else sample.img_path
    h = (
        WORKER.exact_hasher.hash(hash_target)
        if (CONFIG["enable_dedup"] and WORKER.exact_hasher)
        else None
    )
    ph: str | None = None
    if (CONFIG.get("enable_perceptual_dedup") and WORKER.perceptual_hasher
            and sample.img is not None):
        p_hashes = WORKER.perceptual_hasher.hash(sample.img)
        if p_hashes is not None:
            ph = f"{p_hashes[0]}:{p_hashes[1]}"
    return h, ph


@dataclass
class _Target:
    """The resolved target for a sample: an in-memory image and/or a path on disk."""

    image: Any
    path: str | None
    clean_only: bool


def _write_restoration_target(sample: _ImageSample, target: _Target, out_img: Path) -> None:
    """Write the target for restoration, super-resolution and segmentation samples."""
    out_tgt = sample.target_out_path()
    if target.path:
        write_image(loaded_img=None, source_bytes=b"", source_path=Path(target.path),
                    out_path=out_tgt, kind="target")
    elif target.image:
        write_image(loaded_img=target.image, source_bytes=b"", source_path=sample.img_path,
                    out_path=out_tgt, kind="target")
    elif target.clean_only:
        write_image(loaded_img=sample.img, source_bytes=sample.img_data,
                    source_path=sample.img_path, out_path=out_tgt, kind="image")
    elif sample.task == "segmentation" and WORKER.mask_gen is not None and sample.img is not None:
        mask_result = WORKER.mask_gen.generate(
            sample.img, context={"task": sample.task, "slug": sample.slug}
        )
        if mask_result.value is not None:
            write_image(loaded_img=mask_result.value, source_bytes=b"",
                        source_path=sample.img_path, out_path=out_tgt, kind="mask")
    else:
        link_or_copy(out_img, out_tgt, "Target")


def _write_parameter_target(sample: _ImageSample, target: _Target, out_img: Path) -> None:
    """Write the target for parameter-prediction samples (the input itself as fallback)."""
    out_tgt = sample.target_out_path()
    if target.path:
        write_image(loaded_img=None, source_bytes=b"", source_path=Path(target.path),
                    out_path=out_tgt, kind="target")
    elif target.image:
        write_image(loaded_img=target.image, source_bytes=b"", source_path=sample.img_path,
                    out_path=out_tgt, kind="target")
    else:
        link_or_copy(out_img, out_tgt, "Param prediction")


def _write_outputs(sample: _ImageSample, target: _Target) -> Path:
    """Write the image and its target; return the image's final path."""
    out_img = sample.image_out_path()
    on_disk = WORKER.physical_index and sample.name.lower() in WORKER.physical_index
    if not on_disk and not target.clean_only:
        out_img = write_image(loaded_img=sample.img, source_bytes=sample.img_data,
                              source_path=sample.img_path, out_path=out_img, kind="image")
    if sample.task in ["restoration", "super-resolution", "segmentation"]:
        _write_restoration_target(sample, target, out_img)
    elif sample.task == "parameter_prediction":
        _write_parameter_target(sample, target, out_img)
    return out_img


def _auto_label(sample: _ImageSample) -> list[Annotation]:
    """Label the image with the Phase 5 generator, or the task's AutoLabeler without one."""
    if WORKER.label_gen is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cast("list[Annotation]", get_labeler(sample.task, device).predict(sample.img))
    gen_result = WORKER.label_gen.generate(
        sample.img,
        context={"task": sample.task, "slug": sample.slug, "path": str(sample.img_path)},
    )
    if gen_result.kind == "label" and isinstance(gen_result.value, dict):
        return cast("list[Annotation]", gen_result.value.get("annotations") or [])
    return []


def _annotate(sample: _ImageSample, src: AnnotationSource,
              skip_labeling: bool) -> tuple[list[Annotation], bool]:
    """Parse the source annotations, falling back to auto-labeling; flag auto-labels."""
    annotations = parse_annotations(src)
    if (annotations or sample.task in ["quality", "classification"]
            or get_args().no_labeling or skip_labeling or sample.img is None):
        return annotations, False
    labels = _auto_label(sample)
    return labels, bool(labels)


def _file_size(path: Path) -> int:
    """Size of ``path`` in bytes, 0 when it is missing or unreadable."""
    try:
        if path.exists():
            return path.stat().st_size
    except OSError as exc:
        logger.debug("Failed reading file size for %s: %s", path, exc)
    return 0


def _process_image(
    img_input, prefix, slug, idx, task, fmt, ann_data, split,
    output_root_str, skip_labeling,
):
    """Body of ``process_image``; exceptions propagate to the caller's handler."""
    opened = _open_sample(img_input, slug, idx)
    if opened is None:
        return None
    img_path, img_data, img = opened
    sample = _ImageSample(slug=slug, task=task, split=split,
                          output_root=Path(output_root_str), name=f"{prefix}_{slug}_{idx:09d}",
                          img_path=img_path, is_virtual=isinstance(img_input, (bytes, dict)),
                          img_data=img_data, img=img)

    if WORKER.physical_index and sample.name.lower() in WORKER.physical_index:
        return sample.skip_record()

    target_img, target_path = resolve_target(task, ann_data, img_path, sample.is_virtual)

    if (task in ["quality", "diffusion"] and not get_args().no_vetting) or not skip_labeling:
        sample.img = _load_pixels(sample)
        sample.dims = sample.img.size
        if WORKER.auditor is not None and not WORKER.auditor.audit_image(
                sample.img, task, slug).valid:
            return None

    nima = _score_sample(sample, prefix, fmt, ann_data)
    if not _passes_nima_gate(sample, prefix, idx, nima):
        return None

    h, ph = _hashes(sample)
    clean_only = ("parsenet" in slug.lower() or "codeformer" in slug.lower()) \
        and task == "restoration"
    if clean_only and not target_path and not sample.is_virtual:
        target_path = str(img_path)
    out_img_path = _write_outputs(sample, _Target(target_img, target_path, clean_only))

    annotations, is_autolabeled = _annotate(
        sample,
        AnnotationSource(fmt, ann_data, prefix, task, *sample.dims),
        skip_labeling,
    )

    label_file_path = sample.output_root / "labels" / split / f"{sample.name}.txt"
    if not skip_labeling or annotations or task in ["quality", "classification"]:
        content = LabelContent(task, nima.probs, annotations, ann_data, img_path)
        write_label_file(label_file_path, label_lines(content, *sample.dims))

    return {
        "name": sample.name, "source": slug, "task": task, "split": split,
        "hash": h, "perceptual_hash": ph,
        "nima_score": round(nima.score, 3), "is_autolabeled": is_autolabeled,
        "has_segmentation": any(a_["type"] == "segmentation" for a_ in annotations),
        "has_pose": any(a_["type"] == "pose" for a_ in annotations),
        "label_path": str(label_file_path.resolve()),
        "path": str(out_img_path.resolve()),
        "size": _file_size(out_img_path),
        "clip_latent": None,
    }


def process_image(
    img_input, prefix, slug, idx, task, fmt, ann_data, split,
    output_root_str, skip_labeling=False,
):
    """Worker function for parallel processing."""
    try:
        return _process_image(img_input, prefix, slug, idx, task, fmt, ann_data, split,
                              output_root_str, skip_labeling)
    except SAMPLE_PROCESSING_ERRORS as e:
        failed = img_input if isinstance(img_input, (str, Path)) else f"virtual_{slug}_{idx:09d}"
        logger.error("Failed processing %s: %s", failed, e)
        return None


# ─── Diffusion samples ──────────────────────────────────────────────────────
def _open_diffusion_image(img_path: Any) -> tuple[Any, bool] | None:
    """Return ``(image, is_virtual)`` for a path or bytes input; None when unusable."""
    if isinstance(img_path, (bytes, dict)):
        raw = img_path["bytes"] if isinstance(img_path, dict) and "bytes" in img_path else img_path
        if not isinstance(raw, bytes):
            return None
        return Image.open(io.BytesIO(raw)), True
    path = Path(img_path) if isinstance(img_path, str) else img_path
    if not path.exists():
        return None
    return Image.open(path), False


def _caption(img: Any, img_path: Any, is_virtual: bool) -> str:
    """Use the sidecar ``.txt`` caption when present, else the captioner's."""
    if not WORKER.captioner:
        return "a high quality image"
    caption_file = (
        Path(img_path).parent / (Path(img_path).stem + ".txt")
        if not is_virtual and isinstance(img_path, (str, Path))
        else None
    )
    if caption_file and caption_file.exists():
        return caption_file.read_text().strip()
    return WORKER.captioner.generate(img)


def _style(img: Any) -> tuple[str, Any]:
    """CLIP style tag and the float32 latent blob, when the CLIP manifold is loaded."""
    if not WORKER.clip_manifold:
        return "standard", None
    style_tag = WORKER.clip_manifold.tag_style(img)
    clip_latent = WORKER.clip_manifold.extract_features(img).cpu().numpy().flatten().tolist()
    if not clip_latent:
        return style_tag, None
    return style_tag, sqlite3.Binary(np.array(clip_latent).astype(np.float32).tobytes())


def _encode_diffusion_image(img: Any) -> bytes:
    """Encode with the Phase 3 transcoder, or as JPEG quality 95 without one."""
    if WORKER.transcoder is not None and WORKER.transcoder.enabled:
        img_bytes, _fmt = WORKER.transcoder.encode(img, kind="image")
        return img_bytes
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _process_diffusion(img_path, prefix, slug, idx, split):
    """Body of ``process_diffusion``; exceptions propagate to the caller's handler."""
    opened = _open_diffusion_image(img_path)
    if opened is None:
        return None
    img, is_virtual = opened
    img = ensure_srgb(img)
    if is_black_image(img, CONFIG["black_threshold"]):
        return None

    nima_score: Any = 10.0
    if WORKER.sentry:
        nima_score = WORKER.sentry.score(img)
        if nima_score < CONFIG["nima_threshold"]:
            return None

    size = int(CONFIG.get("diffusion_size", 512))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    caption = _caption(img, img_path, is_virtual)
    style_tag, latent_blob = _style(img)

    h = (
        WORKER.exact_hasher.hash(img)
        if (CONFIG["enable_dedup"] and WORKER.exact_hasher)
        else None
    )
    ph: str | None = None
    if CONFIG.get("enable_perceptual_dedup") and WORKER.perceptual_hasher:
        p_hashes = WORKER.perceptual_hasher.hash(img)
        if p_hashes is not None:
            ph = f"{p_hashes[0]}:{p_hashes[1]}"

    img_bytes = _encode_diffusion_image(img)
    nima_val = (
        float(nima_score[0]) if isinstance(nima_score, (tuple, list)) else float(nima_score)
    )
    return {
        "name": f"{prefix}_{idx:09d}", "source": slug, "task": "diffusion", "split": split,
        "hash": h, "perceptual_hash": ph,
        "nima_score": round(nima_val, 3),
        "caption": caption, "style_tag": style_tag, "clip_latent": latent_blob,
        "img_bytes": img_bytes, "size": len(img_bytes),
    }


def process_diffusion(img_path, prefix, slug, idx, split):
    """Text-image processor for Diffusion manifolds."""
    try:
        return _process_diffusion(img_path, prefix, slug, idx, split)
    except SAMPLE_PROCESSING_ERRORS as e:
        safe_path = "virtual_bytes" if isinstance(img_path, (bytes, dict)) else img_path
        logger.error("Failed processing diffusion sample %s: %s", safe_path, e)
        return None


# ─── Batch worker ───────────────────────────────────────────────────────────
def _status_record(name: str, source: str, task: str, status: str) -> dict[str, Any]:
    """Registry record for a row that produced no sample (dropped, skipped or failed)."""
    return {"name": name, "source": source, "task": task, "split": "skipped",
            "hash": status, "nima_score": 1.0, "size": 0}


def process_parquet_shard(
    pq_path, prefix, c_slug, start_idx, task, fmt,
    output_root_str, skip_lbl, train_prob, existing_names,
    existing_on_disk, keep_prob, num_rows,
):
    """Process an entire Parquet shard as one worker task."""
    try:
        df = pd.read_parquet(pq_path)
    except PARQUET_READ_ERRORS as e:
        logger.warning("Skipping corrupted virtual parquet shard %s: %s", pq_path, e)
        return [{"hash": "skipped"}] * num_rows

    results = []
    for offset, row in enumerate(df.itertuples()):
        current_idx = start_idx + offset
        name = f"{prefix}_{c_slug}_{current_idx:09d}"

        if keep_prob < 1.0 and random.random() > keep_prob:
            results.append(_status_record(name, c_slug, task, "dropped"))
            continue
        if name in existing_names or name.lower() in existing_on_disk:
            results.append(_status_record(name, c_slug, task, "skipped"))
            continue
        img_bytes = getattr(row, "image", getattr(row, "pixel_values", None))
        if img_bytes is None:
            results.append(_status_record(name, c_slug, task, "skipped"))
            continue

        split = "train" if random.random() < train_prob else "val"
        if task == "diffusion":
            res = process_diffusion(img_bytes, prefix, c_slug, current_idx, split)
        else:
            row_dict = {k: getattr(row, k) for k in df.columns}
            res = process_image(img_bytes, prefix, c_slug, current_idx, task, fmt,
                                row_dict, split, output_root_str, skip_lbl)
        results.append(res or _status_record(name, c_slug, task, "failed"))
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
        except SAMPLE_PROCESSING_ERRORS as exc:
            logger.warning(
                "Worker task failed: %s with args %s: %s",
                getattr(task_func, "__name__", str(task_func)),
                task_args,
                exc,
            )
            results.append(None)
    return results
