"""Source refs of a manifold: locating the dataset folder, its annotations and its images.

A ref is a ``hf://``, ``gh://`` or ``kaggle://`` slug under ``INPUT_ROOT``
or a ``manifold://<Name>`` sibling manifold that is recirculated. URL-based
Parquet manifests are downloaded here before the images are scanned.
"""

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from core.compiler_core import (
    INPUT_ROOT,
    OUT_PARENT,
    clean_slug,
    detect_annotations,
    download_image,
    parse_coco,
    parse_matlab,
    parse_parquet,
)
from core.error_types import PARQUET_READ_ERRORS

logger = logging.getLogger(__name__)

VALID_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".safetensors",
    ".tiff",
    ".tif",
    ".bmp",
    ".npy",
}

# Task tags of recirculated sibling manifolds (``manifold://<Name>``).
RECIRCULATION_TAGS = {
    "NafNetDebluring": "deblur",
    "NafNetDenoising": "denoise",
    "MprNetDeraining": "derain",
    "FfaNetIndoor": "dehaze_indoor",
    "FfaNetOutdoor": "dehaze_outdoor",
    "MirNetLowLight": "lowlight",
    "MirNetExposure": "exposure",
    "UltraZoom": "superres",
    "FilmRestorer": "vintage",
    "CodeFormer": "face_restorer",
    "ParseNet": "face_parser"
}

_DOWNLOAD_WORKERS = 16


def fast_scan(path, valid_exts):
    """Yield every file under ``path`` whose extension is in ``valid_exts``."""
    for entry in os.scandir(path):
        if entry.is_dir():
            yield from fast_scan(entry.path, valid_exts)
        elif entry.is_file():
            ext = entry.name[entry.name.rfind('.'):].lower()
            if ext in valid_exts:
                yield entry.path


def _require_ann_path(ann_path: Path | None) -> Path:
    """Narrow ann_path to non-None.

    detect_annotations()'s contract is that a non-None format is always
    paired with a non-None path. This helper converts that invariant into
    a runtime check at the type-checker boundary, so parse_* callers can
    accept a plain Path without any suppression.
    """
    if ann_path is None:
        raise RuntimeError(
            "Annotation format was detected but no path was returned. "
            "This indicates a bug in converters.detect_annotations()."
        )
    return ann_path


@dataclass
class ResolvedRef:
    """A source ref mapped to its folder and to the slug its sample names use."""

    ref: str
    tag: str
    slug: str
    c_slug: str
    dataset: Path


def _resolve_recirculated(ref: str, affixes: tuple[str, str]) -> tuple[Path, str, str] | None:
    """``(dataset, slug, c_slug)`` of a ``manifold://`` ref; None when it is not compiled yet."""
    prefix_str, suffix_str = affixes
    m_name = ref.replace("manifold://", "")

    current_suffix = "" if m_name.endswith("MultiTask") else suffix_str
    m_path = OUT_PARENT / f"{prefix_str}{m_name}{current_suffix}"

    if not m_path.exists():
        print(f"[SKIP] Manifold {m_name} not found at {m_path}")
        return None
    dataset = m_path / "images"
    if not dataset.exists():
        dataset = m_path / "targets"
    if not dataset.exists():
        dataset = m_path / "masks"

    slug = f"compiled_{m_name}"
    task_tag = RECIRCULATION_TAGS.get(m_name)
    print(
        f"[RECIRCULATION] Using compiled manifold: {m_name} | Task Tag: {task_tag}"
    )
    c_slug = f"{task_tag}_compiled_{m_name}" if task_tag else clean_slug(slug)
    return dataset, slug, c_slug


def _resolve_remote(ref: str) -> tuple[Path, str]:
    """``(dataset, slug)`` of a downloaded ``hf://`` / ``gh://`` / ``kaggle://`` ref."""
    slug = (
        ref.replace('hf://', '')
        .replace('gh://', '')
        .replace('kaggle://', '')
        .split('/')[-1]
    )
    if ":" not in slug:
        return INPUT_ROOT / slug, slug
    repo_slug = slug.split(":")[0]
    target_slug = (
        slug.split(":")[-1]
        .replace(".tgz", "")
        .replace(".tar.gz", "")
        .replace(".zip", "")
    )
    return INPUT_ROOT / repo_slug / target_slug, target_slug


def _discover_dataset(ref: str, slug: str, dataset: Path) -> Path:
    """Fall back to the lower-cased slug, then to the first folder whose name contains it."""
    if dataset.is_dir():
        return dataset
    dataset = INPUT_ROOT / slug.lower()
    if dataset.is_dir():
        return dataset
    try:
        matches = [
            d
            for d in INPUT_ROOT.iterdir()
            if d.is_dir() and slug.lower() in d.name.lower()
        ]
        if matches:
            dataset = matches[0]
            print(f"[DISCOVERY] Mapping {ref} -> {dataset.name}")
    except OSError as exc:
        logger.warning(
            "Discovery scan for %s in %s failed: %s", slug, INPUT_ROOT, exc
        )
    return dataset


def resolve_ref(ref_entry: dict[str, Any], affixes: tuple[str, str]) -> ResolvedRef | None:
    """Map a ``refs`` entry to its source folder; None (after a [SKIP] line) when missing."""
    ref = ref_entry["ref"]
    if ref.startswith("manifold://"):
        recirculated = _resolve_recirculated(ref, affixes)
        if recirculated is None:
            return None
        dataset, slug, c_slug = recirculated
    else:
        dataset, slug = _resolve_remote(ref)
        c_slug = clean_slug(slug)

    dataset = _discover_dataset(ref, slug, dataset)
    if not dataset.is_dir():
        print(f"[SKIP] Source {ref} not found in {INPUT_ROOT}")
        return None
    return ResolvedRef(ref, ref_entry.get("tag", "sfw"), slug, c_slug, dataset)


@dataclass
class SourceAnnotations:
    """The annotation format of a source, its path and the parsed data."""

    fmt: str | None
    ann_path: Path | None
    ann_data: Any = None
    shards: list[Any] = field(default_factory=list)


def _parse_parquet_shards(dataset: Path) -> list[Any]:
    """Parse the schema of every Parquet file under ``dataset``; broken files are logged."""
    shards = []
    for ap in list(dataset.rglob("*.parquet")):
        try:
            shards.append(parse_parquet(ap))
        except PARQUET_READ_ERRORS as e:
            logger.warning("Failed to parse %s: %s", ap, e)
    return shards


def load_annotations(dataset: Path) -> SourceAnnotations:
    """Detect and parse the annotations of a source folder."""
    fmt, ann_path = detect_annotations(dataset)
    ann = SourceAnnotations(fmt, ann_path)
    if fmt == "coco":
        ann.ann_data = parse_coco(_require_ann_path(ann_path))
    elif fmt == "parquet":
        ann.shards = _parse_parquet_shards(dataset)
        ann.ann_data = ann.shards[0] if ann.shards else None
    elif fmt == "matlab":
        ann.ann_data = parse_matlab(_require_ann_path(ann_path))
    elif fmt in ["xml", "yolo"]:
        ann.ann_data = _require_ann_path(ann_path)
    return ann


def _has_column(ann: SourceAnnotations, names: tuple[str, ...]) -> bool:
    """True when any parsed Parquet shard has one of the columns in ``names``."""
    return any(any(n in cols for n in names) for _, _, cols in ann.shards)


def _missing_downloads(ann: SourceAnnotations, dl_dir: Path) -> list[tuple[str, str]]:
    """``(url, dest)`` for every manifest row whose image is not downloaded yet."""
    to_download = []
    for pq_path, mapping, cols in ann.shards:
        url_col = mapping.get("url", "url")
        key_col = mapping.get("key", "key")
        if url_col not in cols:
            continue
        try:
            df = pd.read_parquet(pq_path)
        except PARQUET_READ_ERRORS as e:
            logger.warning(
                "Skipping corrupted lazy parquet shard %s: %s", pq_path, e
            )
            continue
        for row in df.itertuples():
            url = getattr(row, url_col)
            key = str(getattr(row, key_col, hashlib.md5(url.encode()).hexdigest()))
            dest = dl_dir / f"{key}.jpg"
            if not dest.exists():
                to_download.append((url, str(dest)))
    return to_download


def _download_all(to_download: list[tuple[str, str]], slug: str) -> None:
    """Fetch the missing manifest images in parallel."""
    print(
        f"[RETRIEVAL] Downloading {len(to_download)} missing images for {slug}..."
    )
    with requests.Session() as session, ThreadPoolExecutor(
            max_workers=_DOWNLOAD_WORKERS) as dl_executor:
        dl_tasks = [
            dl_executor.submit(download_image, url, dest, session)
            for url, dest in to_download
        ]
        for _ in tqdm(
            as_completed(dl_tasks),
            total=len(dl_tasks),
            desc="   -> Downloading",
            leave=False,
        ):
            pass


def scan_images(src: ResolvedRef, ann: SourceAnnotations) -> tuple[list[str], bool]:
    """Return the source's image paths and whether it is a virtual (sharded Parquet) source.

    URL-based Parquet manifests are downloaded to ``<dataset>/downloads``
    first and their downloaded images returned.
    """
    images = list(fast_scan(str(src.dataset), VALID_EXTS))
    if images or ann.fmt != "parquet" or not ann.shards:
        return images, False

    if _has_column(ann, ("image", "pixel_values")):
        print(
            f"[VIRTUAL] {src.slug} identified as Sharded Parquet dataset"
            f" ({len(ann.shards)} shards)."
        )
        return images, True

    if not _has_column(ann, ("url",)):
        return images, False
    print(
        f"[LAZY] {src.slug} identified as URL-based manifest. Commencing background"
        " retrieval..."
    )
    dl_dir = src.dataset / "downloads"
    dl_dir.mkdir(exist_ok=True)
    to_download = _missing_downloads(ann, dl_dir)
    if to_download:
        _download_all(to_download, src.slug)
    return list(fast_scan(str(dl_dir), VALID_EXTS)), False
