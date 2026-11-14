"""Unit and end-to-end tests for the SOTA LLM curation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tools.llm_helper.curation import (
    compute_manifest_hash,
    decontaminate_splits,
    enrich_record_metadata,
    heuristic_quality_filter,
    minhash_semantic_dedup,
)
from tools.llm_helper.metadata_compiler import LLMManifoldDescriptor, generate_llm_metadata


def _make_record(rec_id: str, user_text: str, assistant_text: str) -> dict:
    return {
        "id": rec_id,
        "category": "architecture_overview",
        "messages": [
            {"role": "system", "content": "You are LemGendary AI Assistant."},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def test_heuristic_quality_filter_and_enrichment() -> None:
    """Verify low-quality/repetitive records are rejected and valid ones enriched."""
    good = _make_record(
        "r_good",
        "Explain how ChunkedCausalLMLoss reduces VRAM on 4 GB GPUs.",
        (
            "ChunkedCausalLMLoss partitions active tokens into chunks of 256 and computes "
            "linear cross-entropy without materializing full [B, S, V] logit tensors. "
            "For Qwen2.5 (V=151,936), this reduces peak output-layer memory by over 94%."
        ),
    )
    bad_short = _make_record("r_short", "Hi", "Ok")
    bad_rep = _make_record("r_rep", "Repeat test", " ".join(["spam"] * 40))

    accepted, rejected = heuristic_quality_filter([good, bad_short, bad_rep], min_quality=0.30)
    assert len(accepted) == 1
    assert accepted[0]["id"] == "r_good"
    assert accepted[0]["license"] == "CC-BY-NC-4.0"
    assert accepted[0]["language"] == "en"
    assert accepted[0]["difficulty"] in ("basic", "intermediate", "advanced")
    assert len(accepted[0]["content_sha256"]) == 64
    assert len(rejected) == 2


def test_minhash_semantic_dedup_removes_near_duplicates() -> None:
    """Verify MinHash + n-gram cosine deduplication prunes near-duplicate records."""
    r1 = _make_record(
        "r1",
        "How does Kaggle TPU v5e-8 execute PyTorch/XLA models?",
        "PyTorch/XLA compiles static HLO graphs via PJRT across 8 TPU v5e cores with 128 GB HBM.",
    )
    r2 = _make_record(
        "r2",
        "How does Kaggle TPU v5e-8 execute PyTorch/XLA models?",
        "PyTorch/XLA compiles static HLO graphs via PJRT across 8 TPU v5e cores with 128 GB HBM!",
    )
    r3 = _make_record(
        "r3",
        "What is the port of the Dataset Compiler REST API?",
        "The Dataset Compiler FastAPI sidecar daemon listens with Origin Guard.",
    )
    accepted, rejected = minhash_semantic_dedup([r1, r2, r3])
    assert [r["id"] for r in accepted] == ["r1", "r3"]
    assert [r["id"] for r in rejected] == ["r2"]


def test_decontaminate_splits_and_manifest_hash() -> None:
    """Verify train-to-eval decontamination and deterministic Merkle manifest hashing."""
    train = [
        enrich_record_metadata(
            _make_record(
                "t1",
                "Explain LemGendary Dataset Compiler WebP transcoding.",
                "Zero-intermediate WebP transcoding streams directly from zip archives to shards.",
            )
        ),
        enrich_record_metadata(
            _make_record(
                "t2",
                "Explain Sawtooth Governor OOM recovery.",
                "Sawtooth Governor halves batch size and doubles accumulation steps on OOM.",
            )
        ),
    ]
    val = [
        enrich_record_metadata(
            _make_record(
                "v1",
                "Explain LemGendary Dataset Compiler WebP transcoding.",
                "Zero-intermediate WebP transcoding streams directly from zip archives to shards.",
            )
        )
    ]
    clean_train, leaked = decontaminate_splits(train, val, [], overlap_threshold=0.75)
    assert [r["id"] for r in clean_train] == ["t2"]
    assert [r["id"] for r in leaked] == ["t1"]

    h1 = compute_manifest_hash(clean_train, val, [])
    h2 = compute_manifest_hash(clean_train, val, [])
    assert h1 == h2 and len(h1) == 64


def test_metadata_artifacts_emit_rejected_and_lineage() -> None:
    """Verify generate_llm_metadata writes rejected_records.jsonl and manifest_sha256."""
    rec = enrich_record_metadata(
        _make_record(
            "m1",
            "What is unified_data.yaml?",
            "unified_data.yaml is the single source of truth declaring 21 dataset manifolds.",
        )
    )
    rej = {"id": "m_bad", "rejection_reason": "low_quality_or_high_repetition"}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        generate_llm_metadata(
            out,
            "TestLLM",
            [rec],
            [],
            descriptor=LLMManifoldDescriptor("llm", ["ai_helper"], ["docs"]),
            rejected_records=[rej],
            curation_stats={"input_count": 2, "quality_rejected": 1, "output_count": 1},
            seed=42,
        )
        bld = json.loads((out / "build_manifest.json").read_text(encoding="utf-8"))
        qlt = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
        rej_lines = (out / "rejected_records.jsonl").read_text(encoding="utf-8").splitlines()
        assert bld["seed"] == 42
        assert len(bld["manifest_sha256"]) == 64
        assert qlt["curation_stats"]["quality_rejected"] == 1
        assert len(rej_lines) == 1
