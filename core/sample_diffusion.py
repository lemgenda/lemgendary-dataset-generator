"""Text-image processor for diffusion manifolds, run inside the compiler's worker pool.

``process_diffusion`` vets, captions, style-tags and encodes one image (a
path, raw bytes or a Parquet ``{"bytes": ...}`` cell) and returns its
registry record; the encoded bytes go into the WebDataset shards.
"""

import io
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.compiler_config import CONFIG
from core.compiler_worker import WORKER
from core.error_types import SAMPLE_PROCESSING_ERRORS
from core.sample_scoring import dedup_hashes
from utils.image import ensure_srgb, is_black_image

logger = logging.getLogger(__name__)


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
    if not WORKER.models.captioner:
        return "a high quality image"
    caption_file = (
        Path(img_path).parent / (Path(img_path).stem + ".txt")
        if not is_virtual and isinstance(img_path, (str, Path))
        else None
    )
    if caption_file and caption_file.exists():
        return caption_file.read_text().strip()
    return WORKER.models.captioner.generate(img)


def _style(img: Any) -> tuple[str, Any]:
    """CLIP style tag and the float32 latent blob, when the CLIP manifold is loaded."""
    if not WORKER.models.clip_manifold:
        return "standard", None
    style_tag = WORKER.models.clip_manifold.tag_style(img)
    clip_latent = WORKER.models.clip_manifold.extract_features(img).cpu().numpy().flatten().tolist()
    if not clip_latent:
        return style_tag, None
    return style_tag, sqlite3.Binary(np.array(clip_latent).astype(np.float32).tobytes())


def _encode_diffusion_image(img: Any) -> bytes:
    """Encode with the Phase 3 transcoder, or as JPEG quality 95 without one."""
    if WORKER.audit.transcoder is not None and WORKER.audit.transcoder.enabled:
        img_bytes, _fmt = WORKER.audit.transcoder.encode(img, kind="image")
        return img_bytes
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _sentry_score(img: Any) -> Any:
    """QualitySentry score (10.0 without a sentry); None when it is below the threshold."""
    if not WORKER.models.sentry:
        return 10.0
    nima_score = WORKER.models.sentry.score(img)
    if nima_score < CONFIG["nima_threshold"]:
        return None
    return nima_score


def _vetted_image(img_path: Any) -> tuple[Any, bool, Any] | None:
    """Open, vet (black frame, QualitySentry) and resize the input; None when rejected."""
    opened = _open_diffusion_image(img_path)
    if opened is None:
        return None
    img, is_virtual = opened
    img = ensure_srgb(img)
    if is_black_image(img, CONFIG["black_threshold"]):
        return None

    nima_score = _sentry_score(img)
    if nima_score is None:
        return None

    size = int(CONFIG.get("diffusion_size", 512))
    return img.resize((size, size), Image.Resampling.LANCZOS), is_virtual, nima_score


def _process_diffusion(img_path, prefix, slug, idx, split):
    """Body of ``process_diffusion``; exceptions propagate to the caller's handler."""
    vetted = _vetted_image(img_path)
    if vetted is None:
        return None
    img, is_virtual, nima_score = vetted
    caption = _caption(img, img_path, is_virtual)
    style_tag, latent_blob = _style(img)

    h, ph = dedup_hashes(img, img)

    img_bytes = _encode_diffusion_image(img)
    return {
        "name": f"{prefix}_{idx:09d}", "source": slug, "task": "diffusion", "split": split,
        "hash": h, "perceptual_hash": ph,
        "nima_score": round(
            float(nima_score[0]) if isinstance(nima_score, (tuple, list)) else float(nima_score),
            3,
        ),
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
