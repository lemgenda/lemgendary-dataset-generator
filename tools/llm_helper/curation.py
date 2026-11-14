"""SOTA LLM Data Curation Pipeline: Quality Filtering, MinHash/Semantic Dedup,
Decontamination, Metadata Enrichment, and Deterministic Lineage.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

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

    def to_dict(self) -> dict[str, int]:
        """Return stats as a JSON-serializable dictionary."""
        return {
            "input_count": self.input_count,
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
    """Enrich a record with license, language, difficulty, quality_score, and content_sha256."""
    enriched = dict(record)
    quality, difficulty = score_record_quality(enriched)
    content_sha = hashlib.sha256(_record_text(enriched).encode("utf-8")).hexdigest()
    enriched.setdefault("license", default_license)
    enriched.setdefault("language", default_language)
    enriched["difficulty"] = difficulty
    enriched["quality_score"] = quality
    enriched["content_sha256"] = content_sha
    return enriched


def heuristic_quality_filter(
    records: list[dict[str, Any]],
    min_quality: float = 0.30,
    max_repetition: float = 0.65,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter records by quality score and n-gram repetition, returning (accepted, rejected)."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rec in records:
        enriched = enrich_record_metadata(rec)
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


def _minhash_signature(shingles: set[str], num_perm: int = 32) -> tuple[int, ...]:
    if not shingles:
        return (0,) * num_perm
    sig: list[int] = []
    for seed in range(num_perm):
        min_val = _PRIME
        for sh in shingles:
            h = int.from_bytes(
                hashlib.blake2b(f"{seed}:{sh}".encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            min_val = min(min_val, h)
        sig.append(min_val)
    return tuple(sig)


def _cosine_ngram_similarity(a_grams: set[str], b_grams: set[str]) -> float:
    if not a_grams or not b_grams:
        return 0.0
    inter = len(a_grams & b_grams)
    return inter / math.sqrt(float(len(a_grams) * len(b_grams)))


def minhash_semantic_dedup(
    records: list[dict[str, Any]],
    jaccard_threshold: float = 0.85,
    cosine_threshold: float = 0.92,
    num_perm: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate records using MinHash Jaccard + n-gram cosine similarity."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    kept_sigs: list[tuple[tuple[int, ...], set[str]]] = []

    for rec in records:
        grams = _char_ngrams(_record_text(rec), n=5)
        sig = _minhash_signature(grams, num_perm=num_perm)
        is_dup = False
        for k_sig, k_grams in kept_sigs:
            jacc = sum(1 for a, b in zip(sig, k_sig) if a == b) / float(num_perm)
            cos_sim = _cosine_ngram_similarity(grams, k_grams)
            if jacc >= jaccard_threshold or cos_sim >= cosine_threshold:
                is_dup = True
                break
        if is_dup:
            rejected.append({**rec, "rejection_reason": "near_or_semantic_duplicate"})
        else:
            accepted.append(rec)
            kept_sigs.append((sig, grams))
    return accepted, rejected


def decontaminate_splits(
    train_recs: list[dict[str, Any]],
    val_recs: list[dict[str, Any]],
    test_recs: list[dict[str, Any]],
    overlap_threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove train records with high n-gram/semantic overlap against val/test splits."""
    eval_grams = [_char_ngrams(_record_text(r), n=8) for r in (val_recs + test_recs)]
    clean_train: list[dict[str, Any]] = []
    contaminated: list[dict[str, Any]] = []
    for rec in train_recs:
        t_grams = _char_ngrams(_record_text(rec), n=8)
        leaked = any(
            _cosine_ngram_similarity(t_grams, e_grams) >= overlap_threshold
            for e_grams in eval_grams
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
            hasher.update(f"{split_name}:{rec_id}:{c_sha}\n".encode("utf-8"))
    return hasher.hexdigest()
