"""The per-image processor run inside the compiler's worker pool.

``process_image`` turns one ``ImageJob`` (a path, raw bytes or a Parquet
``{"bytes": ...}`` cell) into the manifold's image, target and label files
and returns its registry record. Diffusion samples are handled by
``core.sample_diffusion``; Parquet shards and task batches by
``core.sample_batch``.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from audit.vision_audit import NimaCheck
from core.compiler_config import CONFIG, get_args
from core.compiler_worker import WORKER, get_labeler, link_or_copy, torch, write_image
from core.error_types import SAMPLE_PROCESSING_ERRORS
from core.sample_annotations import (
    Annotation,
    AnnotationSource,
    LabelContent,
    label_lines,
    parse_annotations,
    write_label_file,
)
from core.sample_jobs import ImageJob
from core.sample_scoring import (
    NimaScore,
    ScoreSource,
    authenticity_score,
    dedup_hashes,
    ground_truth_score,
    model_score,
)
from core.sample_targets import resolve_target
from utils.image import ensure_srgb

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".npy")


@dataclass
class _ImageSample:
    """One input image and the output paths derived from it."""

    job: ImageJob
    img_path: Path
    is_virtual: bool
    img_data: bytes = b""
    img: Any = None
    dims: tuple[int, int] = (0, 0)

    @property
    def slug(self) -> str:
        """The cleaned source slug."""
        return self.job.key.slug

    @property
    def task(self) -> str:
        """The manifold's task."""
        return self.job.context.task

    @property
    def split(self) -> str:
        """``train`` or ``val``."""
        return self.job.split

    @property
    def output_root(self) -> Path:
        """The manifold folder."""
        return Path(self.job.context.output_root)

    @property
    def name(self) -> str:
        """The sample's registry and file name."""
        return self.job.key.name

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


def _passes_nima_gate(sample: _ImageSample, nima: NimaScore) -> bool:
    """Apply the auditor's NIMA threshold (5.5 for diffusion, CONFIG otherwise)."""
    if WORKER.audit.auditor is None:
        return True
    is_authenticity = "authentic" in sample.job.key.prefix.lower()
    threshold = 5.5 if sample.task == "diffusion" else CONFIG["nima_threshold"]
    return WORKER.audit.auditor.passes_nima(NimaCheck(
        sample.task, sample.slug, is_authenticity, nima.score, nima.probs, threshold,
        sample.job.key.idx,
    ))


def _score_sample(sample: _ImageSample) -> NimaScore:
    """Resolve the sample's NIMA score from ground truth, path keywords or the model."""
    job = sample.job
    nima = ground_truth_score(ScoreSource(sample.slug, job.context.fmt, job.ann_data,
                                          sample.img_path))
    is_authenticity = "authentic" in job.key.prefix.lower()
    if is_authenticity:
        nima = authenticity_score(sample.img_path, nima)
    if nima.probs[0] == 1.0 and sample.task in ["quality", "diffusion"] and not is_authenticity:
        nima = model_score(sample.img, nima)
    return nima


@dataclass
class _Target:
    """The resolved target for a sample: an in-memory image and/or a path on disk."""

    image: Any
    path: str | None
    clean_only: bool
    mask: Any = None

    def has_source(self) -> bool:
        """True when a real target exists (never the degraded input itself)."""
        return bool(self.path or self.image or self.clean_only or self.mask is not None)


# Tasks whose samples are (input, target) pairs and are rejected without a target.
_PAIRED_TASKS = ("restoration", "super-resolution", "segmentation")


def _generated_mask(sample: _ImageSample) -> Any:
    """A segmentation mask from the Phase 5 generator, or None."""
    if (sample.task != "segmentation" or WORKER.generators.mask_gen is None
            or sample.img is None):
        return None
    return WORKER.generators.mask_gen.generate(
        sample.img, context={"task": sample.task, "slug": sample.slug}
    ).value


def _write_restoration_target(sample: _ImageSample, target: _Target) -> None:
    """Write the target for restoration, super-resolution and segmentation samples.

    The caller has checked :meth:`_Target.has_source`; the degraded input is
    never copied as its own target (that produced identity pairs).
    """
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
    else:
        write_image(loaded_img=target.mask, source_bytes=b"",
                    source_path=sample.img_path, out_path=out_tgt, kind="mask")


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


def _write_outputs(sample: _ImageSample, target: _Target) -> Path | None:
    """Write the image and its target; return the image's final path.

    A paired-task sample without a target is rejected (None) before anything
    is written.
    """
    if sample.task in _PAIRED_TASKS and not target.has_source():
        target.mask = _generated_mask(sample)
        if not target.has_source():
            logger.warning("Rejected %s (%s): no target found", sample.name, sample.task)
            return None
    out_img = sample.image_out_path()
    on_disk = WORKER.physical_index and sample.name.lower() in WORKER.physical_index
    if not on_disk and not target.clean_only:
        out_img = write_image(loaded_img=sample.img, source_bytes=sample.img_data,
                              source_path=sample.img_path, out_path=out_img, kind="image")
    if sample.task in _PAIRED_TASKS:
        _write_restoration_target(sample, target)
    elif sample.task == "parameter_prediction":
        _write_parameter_target(sample, target, out_img)
    return out_img


def _auto_label(sample: _ImageSample) -> list[Annotation]:
    """Label the image with the Phase 5 generator, or the task's AutoLabeler without one."""
    if WORKER.generators.label_gen is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cast("list[Annotation]", get_labeler(sample.task, device).predict(sample.img))
    gen_result = WORKER.generators.label_gen.generate(
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


def _prepare_sample(job: ImageJob) -> _ImageSample | None:
    """Open the job's input; None when a bytes cell is empty."""
    opened = _open_sample(job.img_input, job.key.slug, job.key.idx)
    if opened is None:
        return None
    img_path, img_data, img = opened
    return _ImageSample(job=job, img_path=img_path,
                        is_virtual=isinstance(job.img_input, (bytes, dict)),
                        img_data=img_data, img=img)


def _vet_pixels(sample: _ImageSample) -> bool:
    """Decode the pixels when vetting or labeling needs them; False when the audit rejects."""
    skip_labeling = sample.job.context.skip_labeling
    if (sample.task in ["quality", "diffusion"] and not get_args().no_vetting) or not skip_labeling:
        sample.img = _load_pixels(sample)
        sample.dims = sample.img.size
        if WORKER.audit.auditor is not None and not WORKER.audit.auditor.audit_image(
                sample.img, sample.task, sample.slug).valid:
            return False
    return True


def _write_sample_files(sample: _ImageSample, target_img: Any,
                        target_path: str | None) -> Path | None:
    """Write the image and its target (clean-only restoration pairs reuse the input)."""
    slug = sample.slug.lower()
    clean_only = ("parsenet" in slug or "codeformer" in slug) and sample.task == "restoration"
    if clean_only and not target_path and not sample.is_virtual:
        target_path = str(sample.img_path)
    return _write_outputs(sample, _Target(target_img, target_path, clean_only))


def _label_and_record(sample: _ImageSample, nima: NimaScore, hashes: tuple[Any, str | None],
                      out_img_path: Path) -> dict[str, Any]:
    """Write the label file and return the sample's registry record."""
    job = sample.job
    task = sample.task
    skip_labeling = job.context.skip_labeling
    annotations, is_autolabeled = _annotate(
        sample,
        AnnotationSource(job.context.fmt, job.ann_data, job.key.prefix, task, *sample.dims),
        skip_labeling,
    )

    label_file_path = sample.output_root / "labels" / sample.split / f"{sample.name}.txt"
    if not skip_labeling or annotations or task in ["quality", "classification"]:
        content = LabelContent(task, nima.probs, annotations, job.ann_data, sample.img_path)
        write_label_file(label_file_path, label_lines(content, *sample.dims))

    return {
        "name": sample.name, "source": sample.slug, "task": task, "split": sample.split,
        "hash": hashes[0], "perceptual_hash": hashes[1],
        "nima_score": round(nima.score, 3), "is_autolabeled": is_autolabeled,
        "has_segmentation": any(a_["type"] == "segmentation" for a_ in annotations),
        "has_pose": any(a_["type"] == "pose" for a_ in annotations),
        "label_path": str(label_file_path.resolve()),
        "path": str(out_img_path.resolve()),
        "size": _file_size(out_img_path),
        "clip_latent": None,
    }


def _process_image(job: ImageJob) -> dict[str, Any] | None:
    """Body of ``process_image``; exceptions propagate to the caller's handler."""
    sample = _prepare_sample(job)
    if sample is None:
        return None

    if WORKER.physical_index and sample.name.lower() in WORKER.physical_index:
        return sample.skip_record()

    target_img, target_path = resolve_target(sample.task, job.ann_data, sample.img_path,
                                             sample.is_virtual)
    if not _vet_pixels(sample):
        return None

    nima = _score_sample(sample)
    if not _passes_nima_gate(sample, nima):
        return None

    hashes = dedup_hashes(sample.img_data if sample.is_virtual else sample.img_path, sample.img)
    out_img_path = _write_sample_files(sample, target_img, target_path)
    if out_img_path is None:
        return None
    return _label_and_record(sample, nima, hashes, out_img_path)


def process_image(job: ImageJob) -> dict[str, Any] | None:
    """Worker function for parallel processing."""
    try:
        return _process_image(job)
    except SAMPLE_PROCESSING_ERRORS as e:
        failed = (job.img_input if isinstance(job.img_input, (str, Path))
                  else f"virtual_{job.key.slug}_{job.key.idx:09d}")
        logger.error("Failed processing %s: %s", failed, e)
        return None
