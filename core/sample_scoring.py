"""NIMA quality scores for a sample: ground-truth votes first, then the QualitySentry model.

``dedup_hashes`` computes the exact and perceptual dedup hashes of a sample.

The score is a 1-10 mean opinion score plus a 10-bin probability vector.
Known datasets (AVA, AADB, LAION, TID2013) supply their own ground truth;
authenticity manifolds encode real/fake as the two extreme bins.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from core.compiler_config import CONFIG
from core.compiler_worker import GT_CACHE, WORKER
from utils.math import get_gaussian_probs

logger = logging.getLogger(__name__)

_AUTH_NEUTRAL_PARENTS = ("sut-project", "ai-generated-images-vs-real-images",
                         "real vs fake faces", "raw-sets")
_AUTH_FAKE_KEYS = ("sut-project", "midjourney", "diffusion", "ai", "fake", "gan", "generated")
_AUTH_REAL_KEYS = ("ffhq", "div2k", "celebahq", "human", "real", "afhq", "nature")


def _one_hot(index: int) -> list[float]:
    """Return a 10-bin vector with all mass on ``index``."""
    probs = [0.0] * 10
    probs[index] = 1.0
    return probs


@dataclass
class NimaScore:
    """Mean opinion score and its 10-bin distribution; defaults to the 'unscored' state."""

    score: float = 1.0
    probs: list[float] = field(default_factory=lambda: _one_hot(0))

    @classmethod
    def from_mean(cls, score: float) -> "NimaScore":
        """Build a score with a Gaussian distribution around ``score``."""
        return cls(score, get_gaussian_probs(score))


@dataclass(frozen=True)
class ScoreSource:
    """What the ground-truth lookups need to know about a sample."""

    slug: str
    fmt: str
    ann_data: Any
    img_path: Path


def _ava_score(src: ScoreSource) -> NimaScore | None:
    """AVA: the vote histogram keyed by numeric image id."""
    try:
        img_id = int(src.img_path.stem)
        if img_id in GT_CACHE.ava:
            probs = [GT_CACHE.ava[img_id][f"vote_{i}"] for i in range(1, 11)]
            return NimaScore(sum(p * (i + 1) for i, p in enumerate(probs)), probs)
    except (ValueError, KeyError) as exc:
        logger.debug("AVA score extraction fallback for %s: %s", src.img_path.name, exc)
    return None


def _aadb_score(src: ScoreSource) -> NimaScore | None:
    """AADB: a 0-1 score rescaled to 1-10."""
    try:
        raw_score = GT_CACHE.aadb.get(src.img_path.name)
        if raw_score is not None:
            return NimaScore.from_mean((raw_score * 9.0) + 1.0)
    except (TypeError, ValueError) as exc:
        logger.debug("AADB score extraction fallback for %s: %s", src.img_path.name, exc)
    return None


def _laion_score(src: ScoreSource) -> NimaScore | None:
    """LAION: the ``aesthetic_score`` column of the annotation row."""
    try:
        if isinstance(src.ann_data, dict):
            val = src.ann_data.get("aesthetic_score", src.ann_data.get("score", 6.5))
            return NimaScore.from_mean(float(val) if val is not None else 6.5)
        if src.fmt == "parquet" and src.ann_data:
            df_subset, mapping = cast("tuple[Any, dict[str, Any]]", src.ann_data)
            col = mapping.get("aesthetic_score", "aesthetic_score")
            if col in df_subset.columns:
                return NimaScore.from_mean(float(df_subset[col].iloc[0]))
    except (TypeError, ValueError, KeyError) as exc:
        logger.debug("LAION score extraction fallback for %s: %s", src.img_path.name, exc)
    return None


def _tid_score(src: ScoreSource) -> NimaScore | None:
    """TID2013: the MOS keyed by lower-case file name."""
    try:
        raw_score = GT_CACHE.tid.get(src.img_path.name.lower())
        if raw_score is not None:
            return NimaScore.from_mean(raw_score)
    except (TypeError, ValueError) as exc:
        logger.debug("TID score extraction fallback for %s: %s", src.img_path.name, exc)
    return None


def ground_truth_score(src: ScoreSource) -> NimaScore:
    """Score from the first ground-truth source that applies, else the unscored default."""
    found: NimaScore | None = None
    if "ava" in src.slug and GT_CACHE.ava:
        found = _ava_score(src)
    elif "aadb" in src.slug and GT_CACHE.aadb:
        found = _aadb_score(src)
    elif "laion" in src.slug:
        found = _laion_score(src)
    elif GT_CACHE.tid and src.img_path.name.lower() in GT_CACHE.tid:
        found = _tid_score(src)
    return found or NimaScore()


def authenticity_score(img_path: Path, current: NimaScore) -> NimaScore:
    """Encode fake sources as the lowest bin and real sources as the highest."""
    parent_name = img_path.parent.name.lower()
    if parent_name in _AUTH_NEUTRAL_PARENTS:
        parent_name = ""
    path_str = f"{parent_name}/{img_path.name.lower()}".lower()
    if any(k in path_str for k in _AUTH_FAKE_KEYS):
        return NimaScore(1.0, _one_hot(0))
    if any(k in path_str for k in _AUTH_REAL_KEYS):
        return NimaScore(10.0, _one_hot(9))
    return current


def model_score(img: Any, current: NimaScore) -> NimaScore:
    """Score with the loaded QualitySentry, or keep ``current`` when none is loaded."""
    if not WORKER.models.sentry:
        return current
    score, probs = WORKER.models.sentry.score(img, return_probs=True)
    return NimaScore(score, probs)


def dedup_hashes(hash_target: Any, img: Any) -> tuple[Any, str | None]:
    """Exact hash (when dedup is on) and perceptual hash (when perceptual dedup is on)."""
    h = (
        WORKER.audit.exact_hasher.hash(hash_target)
        if (CONFIG["enable_dedup"] and WORKER.audit.exact_hasher)
        else None
    )
    ph: str | None = None
    if (CONFIG.get("enable_perceptual_dedup") and WORKER.audit.perceptual_hasher
            and img is not None):
        p_hashes = WORKER.audit.perceptual_hasher.hash(img)
        if p_hashes is not None:
            ph = f"{p_hashes[0]}:{p_hashes[1]}"
    return h, ph
