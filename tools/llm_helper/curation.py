"""SOTA LLM Data Curation Pipeline: Quality Filtering, MinHash/Semantic Dedup,
Decontamination, Metadata Enrichment, and Deterministic Lineage.
"""

from __future__ import annotations

import functools
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .token_budget import (
    ASSISTANT_HARD_MAX_TOKENS,
    MAX_TOTAL_TOKENS,
    count_record_tokens,
)

_WORD_RE = re.compile(r"\w+")
_PRIME = (1 << 61) - 1


@dataclass(frozen=True)
class CurationStats:
    """Summary counts from running the full LLM curation pipeline."""

    input_count: int
    quality_rejected: int
    dedup_rejected: int
    decontam_rejected: int
    output_count: int
    token_budget_rejected: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return stats as a JSON-serializable dictionary."""
        return {
            "input_count": self.input_count,
            "token_budget_rejected": self.token_budget_rejected,
            "quality_rejected": self.quality_rejected,
            "dedup_rejected": self.dedup_rejected,
            "decontam_rejected": self.decontam_rejected,
            "output_count": self.output_count,
        }


def _record_text(record: dict[str, Any]) -> str:
    """Extract concatenated user + assistant text from a ChatML record."""
    parts = [
        str(m.get("content", ""))
        for m in record.get("messages", [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]
    return "\n".join(parts).strip()


def _shannon_entropy(tokens: list[str]) -> float:
    """Compute normalized Shannon token entropy in [0, 1]."""
    if len(tokens) <= 1:
        return 0.0
    counts = Counter(tokens)
    total = float(len(tokens))
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_ent = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return min(1.0, entropy / max_ent)


def _repetition_ratio(tokens: list[str], n: int = 3) -> float:
    """Return fraction of repeated n-grams in ``tokens``."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    unique = len(set(ngrams))
    return max(0.0, 1.0 - (unique / float(len(ngrams))))


def score_record_quality(record: dict[str, Any]) -> tuple[float, str]:
    """Compute heuristic quality score in [0, 1] and DEITA-inspired difficulty tier."""
    text = _record_text(record)
    tokens = _WORD_RE.findall(text.lower())
    if len(tokens) < 8:
        return 0.0, "basic"

    entropy = _shannon_entropy(tokens)
    rep_ratio = _repetition_ratio(tokens, n=3)
    len_score = min(1.0, len(tokens) / 120.0)
    quality = max(0.0, min(1.0, (0.45 * entropy) + (0.35 * len_score) + (0.20 * (1.0 - rep_ratio))))

    has_code = "```" in text or "`" in text
    has_math = "$$" in text or "\\(" in text
    if len(tokens) >= 140 or (has_code and has_math):
        difficulty = "advanced"
    elif len(tokens) >= 50 or has_code or has_math:
        difficulty = "intermediate"
    else:
        difficulty = "basic"
    return round(quality, 4), difficulty


def enrich_record_metadata(
    record: dict[str, Any],
    default_license: str = "CC-BY-NC-4.0",
    default_language: str = "en",
) -> dict[str, Any]:
    """Enrich a record with license, language, difficulty, token counts, and sha256."""
    enriched = dict(record)
    quality, difficulty = score_record_quality(enriched)
    content_sha = hashlib.sha256(_record_text(enriched).encode("utf-8")).hexdigest()
    total_tokens, assistant_tokens = count_record_tokens(enriched)
    enriched.setdefault("license", default_license)
    enriched.setdefault("language", default_language)
    enriched["difficulty"] = difficulty
    enriched["quality_score"] = quality
    enriched["total_tokens"] = total_tokens
    enriched["assistant_tokens"] = assistant_tokens
    enriched["content_sha256"] = content_sha
    return enriched


def filter_by_token_budget(
    records: list[dict[str, Any]],
    max_total_tokens: int = MAX_TOTAL_TOKENS,
    max_assistant_tokens: int = ASSISTANT_HARD_MAX_TOKENS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enforce hard total and assistant token ceilings, returning (accepted, rejected)."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rec in records:
        total_tokens, assistant_tokens = count_record_tokens(rec)
        if total_tokens > max_total_tokens:
            rejected.append({**rec, "rejection_reason": "exceeds_max_total_tokens"})
        elif assistant_tokens > max_assistant_tokens:
            rejected.append({**rec, "rejection_reason": "exceeds_max_assistant_tokens"})
        else:
            accepted.append(rec)
    return accepted, rejected


def heuristic_quality_filter(
    records: list[dict[str, Any]],
    min_quality: float = 0.30,
    max_repetition: float = 0.65,
    *,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
    max_assistant_tokens: int = ASSISTANT_HARD_MAX_TOKENS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter records by token ceilings, quality score, and n-gram repetition."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rec in records:
        enriched = enrich_record_metadata(rec)
        if enriched["total_tokens"] > max_total_tokens:
            rejected.append({**enriched, "rejection_reason": "exceeds_max_total_tokens"})
            continue
        if enriched["assistant_tokens"] > max_assistant_tokens:
            rejected.append({**enriched, "rejection_reason": "exceeds_max_assistant_tokens"})
            continue
        tokens = _WORD_RE.findall(_record_text(enriched).lower())
        rep = _repetition_ratio(tokens, n=3)
        if enriched["quality_score"] < min_quality or rep > max_repetition:
            rejected.append({**enriched, "rejection_reason": "low_quality_or_high_repetition"})
        else:
            accepted.append(enriched)
    return accepted, rejected


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


@functools.lru_cache(maxsize=8)
def _minhash_coeffs(num_perm: int) -> tuple[tuple[int, int], ...]:
    """Return deterministic universal-hashing ``(a, b)`` pairs modulo ``_PRIME``."""
    coeffs: list[tuple[int, int]] = []
    for seed in range(num_perm):
        digest = hashlib.blake2b(f"mh_coeff:{seed}".encode(), digest_size=16).digest()
        a = (int.from_bytes(digest[:8], "little") % (_PRIME - 1)) | 1
        b = int.from_bytes(digest[8:], "little") % _PRIME
        coeffs.append((a, b))
    return tuple(coeffs)


def _minhash_signature(shingles: set[str], num_perm: int = 32) -> tuple[int, ...]:
    if not shingles:
        return (0,) * num_perm
    base_hashes = [
        int.from_bytes(hashlib.blake2b(sh.encode(), digest_size=8).digest(), "little")
        for sh in shingles
    ]
    return tuple(
        min(((a * h + b) % _PRIME) for h in base_hashes)
        for a, b in _minhash_coeffs(num_perm)
    )


def _cosine_ngram_similarity(a_grams: set[str], b_grams: set[str]) -> float:
    if not a_grams or not b_grams:
        return 0.0
    inter = len(a_grams & b_grams)
    return inter / math.sqrt(float(len(a_grams) * len(b_grams)))


def _passes_cardinality_bound(len_a: int, len_b: int, cos_threshold: float) -> bool:
    """Cauchy-Schwarz necessary bound for cosine similarity >= ``cos_threshold``."""
    if len_a == 0 or len_b == 0:
        return False
    lo, hi = (len_a, len_b) if len_a <= len_b else (len_b, len_a)
    return float(lo) >= (cos_threshold * cos_threshold) * float(hi)


def _is_near_duplicate(
    grams: set[str],
    sig: tuple[int, ...],
    candidates: list[int],
    kept_sigs: list[tuple[tuple[int, ...], set[str]]],
    thresholds: tuple[float, float, int],
) -> bool:
    """Check exact Jaccard and cosine similarity against LSH candidate indices."""
    jacc_th, cos_th, num_perm = thresholds
    len_g = len(grams)
    for idx in candidates:
        k_sig, k_grams = kept_sigs[idx]
        jacc = sum(1 for a, b in zip(sig, k_sig, strict=True) if a == b) / float(num_perm)
        if jacc >= jacc_th:
            return True
        if _passes_cardinality_bound(len_g, len(k_grams), cos_th) and (
            _cosine_ngram_similarity(grams, k_grams) >= cos_th
        ):
            return True
    return False


class _LSHBandIndex:
    """16-band LSH index over 32-perm MinHash signatures."""

    def __init__(self, num_perm: int = 32, band_size: int = 2) -> None:
        self.band_size = band_size
        self.num_bands = num_perm // band_size
        self._buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    def add(self, sig: tuple[int, ...], item_idx: int) -> None:
        """Index ``item_idx`` across every signature band."""
        for b_idx in range(self.num_bands):
            start = b_idx * self.band_size
            key = (b_idx, sig[start : start + self.band_size])
            self._buckets.setdefault(key, []).append(item_idx)

    def query(self, sig: tuple[int, ...]) -> list[int]:
        """Return sorted unique candidate indices sharing at least one band."""
        cand_set: set[int] = set()
        for b_idx in range(self.num_bands):
            start = b_idx * self.band_size
            key = (b_idx, sig[start : start + self.band_size])
            cand_set.update(self._buckets.get(key, ()))
        return sorted(cand_set)


def minhash_semantic_dedup(
    records: list[dict[str, Any]],
    jaccard_threshold: float = 0.85,
    cosine_threshold: float = 0.92,
    num_perm: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate records using LSH-indexed MinHash Jaccard + n-gram cosine similarity."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    kept_sigs: list[tuple[tuple[int, ...], set[str]]] = []
    use_lsh = num_perm >= 8 and jaccard_threshold >= 0.5 and cosine_threshold >= 0.5
    lsh = _LSHBandIndex(num_perm=num_perm) if use_lsh else None

    for rec in records:
        grams = _char_ngrams(_record_text(rec), n=5)
        sig = _minhash_signature(grams, num_perm=num_perm)
        candidates = lsh.query(sig) if lsh is not None else list(range(len(kept_sigs)))

        if _is_near_duplicate(
            grams, sig, candidates, kept_sigs, (jaccard_threshold, cosine_threshold, num_perm)
        ):
            rejected.append({**rec, "rejection_reason": "near_or_semantic_duplicate"})
            continue

        new_idx = len(kept_sigs)
        accepted.append(rec)
        kept_sigs.append((sig, grams))
        if lsh is not None:
            lsh.add(sig, new_idx)
    return accepted, rejected


def decontaminate_splits(
    train_recs: list[dict[str, Any]],
    val_recs: list[dict[str, Any]],
    test_recs: list[dict[str, Any]],
    overlap_threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove train records with high n-gram/semantic overlap against val/test splits."""
    eval_grams = [_char_ngrams(_record_text(r), n=8) for r in (val_recs + test_recs)]
    use_lsh = overlap_threshold >= 0.5 and len(eval_grams) > 32
    lsh = _LSHBandIndex(num_perm=32) if use_lsh else None
    if lsh is not None:
        for e_idx, e_grams in enumerate(eval_grams):
            lsh.add(_minhash_signature(e_grams, num_perm=32), e_idx)

    clean_train: list[dict[str, Any]] = []
    contaminated: list[dict[str, Any]] = []
    for rec in train_recs:
        t_grams = _char_ngrams(_record_text(rec), n=8)
        cand_indices = (
            lsh.query(_minhash_signature(t_grams, num_perm=32))
            if lsh is not None
            else list(range(len(eval_grams)))
        )
        leaked = any(
            _passes_cardinality_bound(len(t_grams), len(eval_grams[idx]), overlap_threshold)
            and _cosine_ngram_similarity(t_grams, eval_grams[idx]) >= overlap_threshold
            for idx in cand_indices
        )
        if leaked:
            contaminated.append({**rec, "rejection_reason": "eval_split_contamination"})
        else:
            clean_train.append(rec)
    return clean_train, contaminated


def compute_manifest_hash(
    train_recs: list[dict[str, Any]],
    val_recs: list[dict[str, Any]],
    test_recs: list[dict[str, Any]],
) -> str:
    """Compute a deterministic SHA-256 Merkle root hash across all splits."""
    hasher = hashlib.sha256()
    for split_name, recs in (("train", train_recs), ("val", val_recs), ("test", test_recs)):
        for rec in recs:
            rec_id = str(rec.get("id", ""))
            c_sha = str(rec.get("content_sha256", ""))
            hasher.update(f"{split_name}:{rec_id}:{c_sha}\n".encode())
    return hasher.hexdigest()
