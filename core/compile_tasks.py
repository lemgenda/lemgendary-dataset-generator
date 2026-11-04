"""Turning a manifold's sources into worker task tuples, and NSFW ratio balancing.

Each task is ``(callable, *args)`` for ``batch_worker``:

- ``(process_image, ImageJob)`` for one source image,
- ``(process_parquet_shard, ShardJob)`` for one virtual Parquet shard,
- ``(process_diffusion, img_path, prefix, slug, idx, split)`` for diffusion.
"""

import dataclasses
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

from core.compile_sources import ResolvedRef, SourceAnnotations
from core.compiler_core import (
    CONFIG,
    ExistingSamples,
    ImageJob,
    SampleKey,
    ShardJob,
    SourceContext,
    process_diffusion,
    process_image,
    process_parquet_shard,
)
from core.error_types import PARQUET_READ_ERRORS
from core.lazy_imports import load_parquet

logger = logging.getLogger(__name__)

# 2026: Task tuples passed to batch_worker. Each item is
#     (callable, *heterogeneous_args)
# Annotated as tuple[Any, ...] so downstream consumers accept the mix.
TaskItem = tuple[Any, ...]

# 2026: Type aliases for the three parser return shapes, used at cast sites.
CocoAnnData = tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]
ParquetAnnData = tuple[Path, dict[str, Any], list[str]]
MatlabAnnData = tuple[dict[str, Any], str]

# Per-class cap on the validation split of the CodeFormer real-vs-fake faces source.
_FACE_VAL_QUOTA = 1000

# Restoration manifolds drop source paths that belong to another degradation.
_RAIN_KEYS = ["rain", "droplet"]
_NOT_NOISE_KEYS = ["blur", "haze", "lowlight", "exposure"]
_NOT_BLUR_KEYS = ["noise", "haze", "lowlight", "exposure"]


@dataclass
class ManifoldContext:
    """What every source of one manifold shares while its tasks are planned."""

    model_key: str
    model_config: dict[str, Any]
    task: str
    pascal_name: str
    output_root: Path
    existing: ExistingSamples

    @property
    def skip_labeling(self) -> bool:
        """True when the registry disables labeling for this manifold."""
        return not self.model_config.get("labeling", True)

    def train_probability(self) -> float:
        """Probability that a sample goes to the train split."""
        model_val_split = self.model_config.get("val_split", None)
        if model_val_split is not None:
            return 1.0 - float(model_val_split)
        if self.task == "diffusion" or "image_to_text" in self.model_key:
            return 1.0
        return CONFIG["train_split"]

    def source_context(self, fmt: str | None) -> SourceContext:
        """The per-source context the sample processors receive."""
        return SourceContext(self.task, fmt, str(self.output_root), self.skip_labeling)


# ─── Annotation lookups ─────────────────────────────────────────────────────
# PRE-COMPUTE ANNOTATION LOOKUPS TO AVOID O(N^2) BOTTLENECKS
# Explicit types here are essential: without them, the dicts infer
# as dict[str, Unknown] and downstream .get() lookups lose their
# key/value types, causing false positives on every consumer.
@dataclass
class AnnotationLookup:
    """Per-file indexes into a source's COCO, Parquet or MATLAB annotations."""

    coco_file_to_id: dict[str, int] = field(default_factory=dict)
    parquet_map: dict[str, Any] = field(default_factory=dict)
    matlab_map: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _index_parquet(ann_data: Any, lookup: AnnotationLookup) -> None:
    """Group the Parquet annotation rows by file name."""
    pq_path, mapping, _cols = cast(ParquetAnnData, ann_data)
    try:
        df = pd.read_parquet(str(pq_path))
    except PARQUET_READ_ERRORS as e:
        logger.warning("Skipping corrupted parquet %s: %s", pq_path, e)
        df = pd.DataFrame()
    file_col = mapping.get("file_name", "file_name")
    if (
        file_col in df.columns
        and len(df) > 0
        and (df[file_col].dtype != 'object' or isinstance(df[file_col].iloc[0], str))
    ):
        for fname, group in df.groupby(file_col):
            lookup.parquet_map[str(fname)] = group


def _index_matlab(ann_data: Any, lookup: AnnotationLookup) -> None:
    """Group the MATLAB annotation entries by image name."""
    data, key = cast(MatlabAnnData, ann_data)
    for entry in data.get(key, []):
        if not isinstance(entry, dict):
            logger.debug("Skipping non-mapping matlab entry: %r", entry)
            continue
        fname = entry.get("image_name", entry.get("name"))
        if fname:
            lookup.matlab_map.setdefault(fname, []).append(entry)


def build_lookup(ann: SourceAnnotations, is_virtual: bool) -> AnnotationLookup:
    """Index the source's annotations by file name."""
    lookup = AnnotationLookup()
    if ann.fmt == "coco" and ann.ann_data:
        images_meta, _anns_meta = cast(CocoAnnData, ann.ann_data)
        for k, v in images_meta.items():
            lookup.coco_file_to_id[v["file_name"]] = k
    elif ann.fmt == "parquet" and ann.ann_data and not is_virtual:
        _index_parquet(ann.ann_data, lookup)
    elif ann.fmt == "matlab" and ann.ann_data:
        _index_matlab(ann.ann_data, lookup)
    return lookup


def _sidecar_annotation(ann: SourceAnnotations, img_path: Path) -> str | None:
    """Path of the image's ``.xml`` / ``.txt`` / ``.npz`` sidecar, when it exists."""
    if ann.ann_path is None:
        return None
    ext = ".xml" if ann.fmt == "xml" else (".txt" if ann.fmt == "yolo" else ".npz")
    ann_file = ann.ann_path / f"{img_path.stem}{ext}"
    return str(ann_file) if ann_file.exists() else None


def _coco_annotation(ann: SourceAnnotations, lookup: AnnotationLookup, img_path: Path) -> Any:
    """The COCO annotations of one image."""
    img_id = lookup.coco_file_to_id.get(img_path.name)
    if img_id is None:
        return None
    return cast(CocoAnnData, ann.ann_data)[1].get(img_id, [])


def _parquet_annotation(ann: SourceAnnotations, lookup: AnnotationLookup,
                        img_path: Path) -> Any:
    """The Parquet rows (and column mapping) of one image."""
    _pq_path, mapping, _cols = cast(ParquetAnnData, ann.ann_data)
    df_subset = lookup.parquet_map.get(img_path.name)
    if df_subset is not None and not df_subset.empty:
        return df_subset, mapping
    return None


_ANNOTATION_PICKERS = {
    "coco": _coco_annotation,
    "parquet": _parquet_annotation,
    "matlab": lambda _ann, lookup, img_path: lookup.matlab_map.get(img_path.name, []),
    "safetensors": lambda ann, _lookup, _img_path: ann.ann_data,
    "xml": lambda ann, _lookup, img_path: _sidecar_annotation(ann, img_path),
    "yolo": lambda ann, _lookup, img_path: _sidecar_annotation(ann, img_path),
    "npz": lambda ann, _lookup, img_path: _sidecar_annotation(ann, img_path),
}


def specific_annotation(ann: SourceAnnotations, lookup: AnnotationLookup,
                        img_path: Path) -> Any:
    """The annotation data that belongs to one image of the source."""
    picker = _ANNOTATION_PICKERS.get(ann.fmt or "")
    if not ann.ann_data or picker is None:
        return None
    return picker(ann, lookup, img_path)


# ─── Image tasks ────────────────────────────────────────────────────────────
def _excluded_restoration_path(model_key: str, img_path_str: str) -> bool:
    """True when a restoration source path belongs to another degradation."""
    p_low = img_path_str.lower()
    m_low = model_key.lower()
    if ("deraining" not in m_low and "multitask" not in m_low
            and any(k in p_low for k in _RAIN_KEYS)):
        return True
    if "denoising" in m_low and any(k in p_low for k in _NOT_NOISE_KEYS):
        return True
    return "debluring" in m_low and any(k in p_low for k in _NOT_BLUR_KEYS)


def _face_val_split(counts: dict[str, int], img_path: Path, split: str) -> str:
    """Put the first 1000 real and 1000 fake CodeFormer faces into the val split.

    ``counts`` holds the val images taken so far per class; images that are
    neither real nor fake keep ``split``.
    """
    parent = img_path.parent.name.lower()
    kind = "real" if "real" in parent else ("fake" if "fake" in parent else None)
    if kind is None:
        return split
    if counts[kind] < _FACE_VAL_QUOTA:
        counts[kind] += 1
        return "val"
    return "train"


def image_tasks(ctx: ManifoldContext, src: ResolvedRef, ann: SourceAnnotations,
                images: list[str]) -> list[TaskItem]:
    """One task per source image that is not registered or on disk yet."""
    lookup = build_lookup(ann, is_virtual=False)
    train_prob = ctx.train_probability()
    source_ctx = ctx.source_context(ann.fmt)
    face_counts = ({"real": 0, "fake": 0}
                   if ctx.model_key == "codeformer"
                   and "realvsfakefaces" in ctx.pascal_name.lower()
                   else None)

    tasks: list[TaskItem] = []
    for i, img_path_str in enumerate(images):
        key = SampleKey(ctx.pascal_name, src.c_slug, i)
        if key.name in ctx.existing:
            continue
        if ctx.task == "restoration" and _excluded_restoration_path(ctx.model_key, img_path_str):
            continue

        img_path = Path(img_path_str)
        split = "train" if random.random() < train_prob else "val"
        if face_counts is not None:
            split = _face_val_split(face_counts, img_path, split)

        specific_ann_data = specific_annotation(ann, lookup, img_path)
        if ctx.task == "diffusion":
            tasks.append((process_diffusion, img_path, ctx.pascal_name, src.c_slug, i, split))
        else:
            tasks.append((process_image,
                          ImageJob(img_path, key, source_ctx, specific_ann_data, split)))
    return tasks


# ─── Virtual Parquet shard tasks ────────────────────────────────────────────
def _shard_rows(pq_path: Any) -> int | None:
    """Row count of a Parquet shard (pyarrow metadata, pandas fallback); None when corrupt."""
    try:
        return load_parquet().read_metadata(str(pq_path)).num_rows
    except (ImportError, *PARQUET_READ_ERRORS) as meta_exc:
        logger.debug(
            "pyarrow metadata read failed for %s (%s); counting rows via pandas",
            pq_path,
            meta_exc,
        )
    try:
        return pd.read_parquet(pq_path, columns=[]).shape[0]
    except PARQUET_READ_ERRORS as e:
        logger.warning(
            "Skipping corrupted virtual parquet shard %s: %s", pq_path, e
        )
        return None


def shard_tasks(ctx: ManifoldContext, src: ResolvedRef,
                ann: SourceAnnotations) -> list[TaskItem]:
    """One task per non-empty Parquet shard of a virtual source."""
    train_prob = ctx.train_probability()
    source_ctx = ctx.source_context(ann.fmt)
    tasks: list[TaskItem] = []
    global_idx = 0
    for pq_path, _mapping, _cols in ann.shards:
        num_rows = _shard_rows(pq_path)
        if not num_rows:
            continue
        job = ShardJob(pq_path, SampleKey(ctx.pascal_name, src.c_slug, global_idx), source_ctx,
                       train_prob, ctx.existing, 1.0, num_rows)
        tasks.append((process_parquet_shard, job))
        global_idx += num_rows
    return tasks


# ─── Counting and NSFW balancing ────────────────────────────────────────────
def _shard_job(t: TaskItem) -> ShardJob | None:
    """The ShardJob of a process_parquet_shard task, None for other tasks."""
    if t[0].__name__ == "process_parquet_shard":
        return cast(ShardJob, t[1])
    return None


def count_one(t: TaskItem) -> int:
    """Return the expected sample count for one task item.

    Parquet shard tasks carry their row count in their ShardJob; every
    other task produces exactly one output.
    """
    job = _shard_job(t)
    return job.num_rows if job is not None else 1


def get_count(task_list: list[TaskItem]) -> int:
    """Expected sample count of a task list."""
    return sum(count_one(t) for t in task_list)


def balance_nsfw(sfw_tasks: list[TaskItem], nsfw_tasks: list[TaskItem],
                 target_nsfw_ratio: float) -> list[TaskItem]:
    """2026 Strategy: Dynamic Ratio Balancing (v5.8) - cap the NSFW share of the manifold."""
    sfw_count = get_count(sfw_tasks)
    nsfw_count = get_count(nsfw_tasks)
    if not (target_nsfw_ratio > 0 and nsfw_count > 0):
        return nsfw_tasks
    max_nsfw = int(sfw_count * target_nsfw_ratio / (1.0 - target_nsfw_ratio))
    if nsfw_count <= max_nsfw:
        return nsfw_tasks

    print(
        f"[BALANCING] NSFW pool ({nsfw_count}) exceeds {target_nsfw_ratio * 100}% cap."
        f" Capping at {max_nsfw} samples."
    )
    nsfw_keep_prob = max_nsfw / nsfw_count
    new_nsfw_tasks: list[TaskItem] = []
    for item in nsfw_tasks:
        job = _shard_job(item)
        if job is not None:
            new_nsfw_tasks.append((item[0], dataclasses.replace(
                job, keep_prob=nsfw_keep_prob, num_rows=int(job.num_rows * nsfw_keep_prob),
            )))
        elif random.random() <= nsfw_keep_prob:
            new_nsfw_tasks.append(item)
    return new_nsfw_tasks
