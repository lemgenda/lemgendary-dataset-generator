"""Unit and integration tests for the state-grounded QLoRA copilot LLM compiler."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from api.routes.llm import LLMValidationRequest, validate_llm_corpus

from core.ecosystem import kaggle_owner
from core.manifold_compile import _create_layout, _is_llm_dataset
from tests.llm_records import chat_record
from tools.llm_helper.base_compiler import BaseLLMCompiler
from tools.llm_helper.copilot_grounding import (
    GUI_TABS,
    SYSTEM_SYNTHESIS_PROMPT,
    UNSUPPORTED_CONTAINERS,
    GuiStateOptions,
    build_gui_state_snapshot,
    build_retrieved_docs_snippet,
    load_registry_context,
    synthesize_boundary_and_negative_records,
    synthesize_state_grounded_copilot_records,
)
from tools.llm_helper.curation import (
    CurationStats,
    filter_by_token_budget,
    heuristic_quality_filter,
    minhash_semantic_dedup,
)
from tools.llm_helper.knowledge_synthesizer import (
    compile_knowledge_corpus,
    synthesize_instructions_from_section,
)
from tools.llm_helper.quality_validator import QualityValidator, TokenValidationConfig
from tools.llm_helper.response_formatter import (
    ReplyOptions,
    distill_section_to_structured_parts,
    format_structured_reply,
)
from tools.llm_helper.taxonomy import sidecar_ports
from tools.llm_helper.token_budget import (
    ASSISTANT_HARD_MAX_TOKENS,
    ASSISTANT_TARGET_MAX_TOKENS,
    ASSISTANT_TARGET_MIN_TOKENS,
    MAX_TOTAL_TOKENS,
    SYNTHESIS_MAX_TOTAL_TOKENS,
    count_record_tokens,
    count_tokens,
)


def _assistant_text(rec: dict) -> str:
    return next(m["content"] for m in rec["messages"] if m.get("role") == "assistant")


def _system_text(rec: dict) -> str:
    return next(m["content"] for m in rec["messages"] if m.get("role") == "system")


def test_token_budget_fallback_and_tiktoken_mock() -> None:
    """Verify fallback len/3.5 estimator and tiktoken encoder delegation."""
    sample = "LemGendary AI Studio QLoRA token budget verification."
    with patch("tools.llm_helper.token_budget._load_tiktoken_encoding", return_value=None):
        assert count_tokens(sample) == math.ceil(len(sample) / 3.5)
        assert count_tokens("") == 0

    fake_encoder = SimpleNamespace(encode=lambda text, **_kw: list(range(len(text.split()))))
    with patch(
        "tools.llm_helper.token_budget._load_tiktoken_encoding",
        return_value=fake_encoder,
    ):
        assert count_tokens(sample) == len(sample.split())


def test_gui_state_snapshot_matches_zustand_schema() -> None:
    """Verify [GUI_STATE] snapshot mirrors Zustand store fields and valid tabs."""
    env_p, comp_p, train_p = sidecar_ports()
    snap = build_gui_state_snapshot(
        GuiStateOptions(
            active_tab="tab-training",
            active_training={
                "model_key": "nafnet_debluring",
                "status": "TRAINING",
                "epoch": 12,
                "budget": 80,
                "rung": 384,
                "fraction": 0.6,
                "governor": "STABILIZATION_COOLDOWN",
                "metrics": {"psnr": 28.4},
                "targets": {"psnr": 30.2},
            },
        )
    )
    assert snap["active_tab"] in GUI_TABS
    assert set(snap["hardware"].keys()) == {
        "gpu",
        "vram_mb",
        "backend",
        "turing_no_tensor_cores",
    }
    assert set(snap["sidecars"].keys()) == {
        f"env_{env_p}",
        f"compiler_{comp_p}",
        f"training_{train_p}",
    }
    assert set(snap.keys()) == {
        "active_tab",
        "hardware",
        "sidecars",
        "active_training",
        "active_compile",
        "recent_errors",
    }
    assert "STABILIZATION_COOLDOWN" in SYSTEM_SYNTHESIS_PROMPT
    assert "vram_mb: 4096" in SYSTEM_SYNTHESIS_PROMPT


def test_retrieved_docs_snippet_sentence_and_token_bounds() -> None:
    """Verify [RETRIEVED_DOCS] keeps 1-3 clean sentences within token cap."""
    _env_p, comp_p, _train_p = sidecar_ports()
    prose = (
        f"First sentence documents port {comp_p}.\n"
        "Second sentence explains WebDataset (.data) shards.\n"
        "Third sentence describes Sawtooth Governor.\n"
        "Fourth sentence should be excluded."
    )
    snippet = build_retrieved_docs_snippet(prose, max_sentences=3, max_tokens=100)
    assert "First sentence" in snippet
    assert "Third sentence" in snippet
    assert "Fourth sentence" not in snippet


def test_boundary_and_negative_covers_all_three_archetypes() -> None:
    """Verify boundary generator covers offline/idle, unsupported containers, missing SOTA."""
    _env_p, _comp_p, train_p = sidecar_ports()
    workspace_root = Path(__file__).resolve().parents[2]
    ctx = load_registry_context(workspace_root)
    recs = synthesize_boundary_and_negative_records(ctx, "Base sys.", target_count=36)
    assert len(recs) == 36
    assert all(r["category"] == "boundary_and_negative" for r in recs)

    combined = "\n".join(_assistant_text(r) for r in recs)
    assert str(train_p) in combined
    for fmt in UNSUPPORTED_CONTAINERS:
        assert fmt in combined
    assert "unified_models_v2.yaml" in combined
    assert "Target Metrics (SOTA)" in combined

    kept, _rejected = minhash_semantic_dedup(recs)
    assert len(kept) / len(recs) >= 0.70

    validator = QualityValidator(token_config=TokenValidationConfig(enforce_copilot_structure=True))
    is_valid, issues = validator.validate_corpus(recs)
    assert is_valid, f"Boundary issues: {issues}"


def _assert_corpus_invariants(corpus: list[dict]) -> None:
    """Check grounding ratio, boundary survival ratio, token ceilings, and validator."""
    grounded = sum(
        1
        for r in corpus
        if "[GUI_STATE]" in _system_text(r) or "[RETRIEVED_DOCS]" in _system_text(r)
    )
    boundary = sum(1 for r in corpus if r["category"] == "boundary_and_negative")
    assert grounded / len(corpus) >= 0.60
    assert 0.07 <= (boundary / len(corpus)) <= 0.14

    for rec in corpus:
        total_tok, asst_tok = count_record_tokens(rec)
        assert total_tok <= SYNTHESIS_MAX_TOTAL_TOKENS
        assert total_tok <= MAX_TOTAL_TOKENS
        assert ASSISTANT_TARGET_MIN_TOKENS <= asst_tok <= ASSISTANT_TARGET_MAX_TOKENS
        assert asst_tok <= ASSISTANT_HARD_MAX_TOKENS

    validator = QualityValidator(token_config=TokenValidationConfig(enforce_copilot_structure=True))
    is_valid, issues = validator.validate_corpus(corpus)
    errors = [iss for iss in issues if iss.severity == "ERROR"]
    warnings = [iss for iss in issues if iss.severity == "WARNING" and iss.field != "prompt"]
    assert is_valid, f"Critical validation errors: {errors[:5]}"
    assert not warnings, f"Unexpected copilot structure/token warnings: {warnings[:5]}"


def test_synthesized_corpus_grounding_ratio_structure_and_token_bounds() -> None:
    """Verify >=60% grounded, ~10% boundary, 150-350 assistant tokens, <=1450 total."""
    _env_p, comp_p, _train_p = sidecar_ports()
    workspace_root = Path(__file__).resolve().parents[2]
    ctx = load_registry_context(workspace_root)
    assert len(ctx.models) >= 20
    assert len(ctx.datasets) >= 18
    assert len(ctx.gui_controls) >= 50

    with tempfile.TemporaryDirectory() as tmp_dir:
        md_file = Path(tmp_dir) / "MANUAL_TEST.md"
        md_file.write_text(
            "# Operations Guide\n\n"
            "## Dataset Compilation\n\n"
            "Select a manifold in comp-select-manifold and click comp-btn-compile "
            f"to compile WebDataset shards on port {comp_p}. Verify schema integrity with "
            "compiler-btn-validate-schema before queuing large builds.\n",
            encoding="utf-8",
        )
        assert len(synthesize_state_grounded_copilot_records(ctx, "Base sys.")) >= 40
        sec_recs = synthesize_instructions_from_section(
            "MANUAL_TEST.md",
            "Operations Guide > Dataset Compilation",
            md_file.read_text(encoding="utf-8"),
            "Base sys.",
            1,
        )
        assert len(sec_recs) >= 1
        assert "[GUI_STATE]" in _system_text(sec_recs[0])

    corpus = compile_knowledge_corpus(
        knowledge_sources=[
            {"type": "markdown_file", "path": "lemgendary-docs/MD-Papers/MANUAL_CLI.md"},
            {
                "type": "markdown_file",
                "path": "lemgendary-docs/MD-Papers/GUI_CONTROL_REGISTRY.md",
            },
            {
                "type": "yaml_file",
                "path": "lemgendary-training-suite/unified_models_v2.yaml",
            },
            {"type": "yaml_file", "path": "lemgendary-datasets/unified_data.yaml"},
        ],
        system_prompt="You are the LemGendary AI Studio Copilot.",
        workspace_root=workspace_root,
    )
    assert len(corpus) > 100
    _assert_corpus_invariants(corpus)


def test_quality_validator_token_ceilings_and_emoji_gates() -> None:
    """Verify hard rejection above 1,536 total tokens, 420 assistant tokens, and emojis."""
    validator = QualityValidator(token_config=TokenValidationConfig(enforce_copilot_structure=True))
    assert len(QualityValidator().validate_record(chat_record("legacy_1"))) == 0

    over_assistant = {
        "id": "over_asst",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Help."},
            {"role": "assistant", "content": "word " * 500},
        ],
    }
    issues = validator.validate_record(over_assistant)
    assert any(i.severity == "ERROR" and i.field == "assistant" for i in issues)

    over_total = {
        "id": "over_tot",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": "sys " * 1500},
            {"role": "user", "content": "Help."},
            {
                "role": "assistant",
                "content": (
                    "[CONFIRMED] Direct answer.\n\n"
                    "- Bullet one with enough detail.\n"
                    "- Bullet two with enough detail.\n"
                    "- Bullet three with enough detail.\n\n"
                    "Next step: validate schema."
                ),
            },
        ],
    }
    tot_issues = validator.validate_record(over_total)
    assert any(i.severity == "ERROR" and i.field == "tokens" for i in tot_issues)

    dingbat_rec = {
        "id": "dingbat_1",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Check status \u2705"},
            {"role": "assistant", "content": "Ok."},
        ],
    }
    assert any("emoji" in i.issue.lower() for i in validator.validate_record(dingbat_rec))

    ctrl_rec = {
        "id": "ctrl_1",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Check \x07lert."},
            {"role": "assistant", "content": "Ok."},
        ],
    }
    assert any(
        "control characters" in i.issue.lower() for i in validator.validate_record(ctrl_rec)
    )


def _sample_curation_records() -> tuple[dict, dict, dict]:
    """Build two valid copilot records and one oversized record for curation tests."""
    comp_p = sidecar_ports()[1]
    good_asst = (
        "[CONFIRMED] Manifold `nima_aesthetic` compiles to WebDataset `.data` shards.\n\n"
        "- Select `nima_aesthetic` in `comp-select-manifold` on `tab-datasets`.\n"
        f"- Click `compiler-btn-validate-schema` to verify port {comp_p} schema contracts.\n"
        "- Click `comp-btn-compile` (Compile Manifold) or run `python cli.py compile`.\n"
        "- Monitor `WS /api/ws/events` until `shard_offsets.json` is emitted cleanly "
        "and confirms zero corrupt WebP or JSON sidecar members in shards.\n\n"
        "Next step: run `python cli.py verify` after compilation finishes."
    )
    good = {
        "id": "cur_good",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": "Copilot [GUI_STATE]{}[/GUI_STATE]"},
            {"role": "user", "content": "How do I compile nima_aesthetic on tab-datasets?"},
            {"role": "assistant", "content": good_asst},
        ],
    }
    val_asst = (
        "[CONFIRMED] Stage 1 triage confirms `turing_no_tensor_cores: true` on GTX 1650.\n\n"
        "- Stage 2 diagnosis checks `tab-training` badge and disables BF16 AMP precision.\n"
        "- Stage 3 remediation keeps `fp32` math via Turing override before `Local Training`.\n"
        "- Click `Blend Soup` (`soup-btn-execute`) only after passing the Monotonic Ratchet.\n"
        "- Export finalized ONNX checkpoints via `lemtrain export` after telemetry stabilizes.\n\n"
        "Next step: inspect `train_metrics.json` inside the `tab-health` sidecar panel."
    )
    good_val = {
        "id": "cur_good_2",
        "category": "troubleshooting_diagnostics",
        "messages": [
            {"role": "system", "content": "Copilot [GUI_STATE]{}[/GUI_STATE]"},
            {
                "role": "user",
                "content": "Why did BF16 mixed precision disable automatically on GTX 1650?",
            },
            {"role": "assistant", "content": val_asst},
        ],
    }
    too_long = {
        "id": "cur_long",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Explain everything."},
            {"role": "assistant", "content": "token " * 500},
        ],
    }
    return good, good_val, too_long


def test_curation_token_budget_gate() -> None:
    """Verify curation rejects oversized records and records token metadata."""
    good, _good_val, too_long = _sample_curation_records()
    acc, rej = filter_by_token_budget([good, too_long])
    assert [r["id"] for r in acc] == ["cur_good"]
    assert [r["id"] for r in rej] == ["cur_long"]
    assert rej[0]["rejection_reason"] == "exceeds_max_assistant_tokens"

    acc_q, rej_q = heuristic_quality_filter([good, too_long])
    assert [r["id"] for r in acc_q] == ["cur_good"]
    assert acc_q[0]["total_tokens"] <= MAX_TOTAL_TOKENS
    assert acc_q[0]["assistant_tokens"] <= ASSISTANT_HARD_MAX_TOKENS
    assert len(rej_q) == 1

    stats = CurationStats(
        input_count=2,
        token_budget_rejected=1,
        quality_rejected=0,
        dedup_rejected=0,
        decontam_rejected=0,
        output_count=1,
    )
    assert stats.to_dict()["token_budget_rejected"] == 1


def test_end_to_end_compile_manifold_curation() -> None:
    """Verify BaseLLMCompiler.compile_manifold propagates token_budget_rejected stats."""
    good, good_val, too_long = _sample_curation_records()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        cfg_file = tmp / "unified_data.yaml"
        cfg_file.write_text(
            "_registry_metadata:\n"
            "  version: '4.3.0'\n"
            "  name_prefix: LemGendized\n"
            "  name_suffix: ''\n"
            "  output_folder_name: ../LemGendaryDatasets\n"
            "datasets:\n"
            "  test_llm:\n"
            "    name: TestLLM\n"
            "    task: ai_helper\n"
            "    dataset_type: llm\n"
            "    modernized_folder: TestLLM\n"
            f"    kaggle_ref: kaggle://{kaggle_owner()}/testllm\n"
            "    canonical_format: jsonl\n"
            "    val_split: 0.2\n",
            encoding="utf-8",
        )
        compiler = BaseLLMCompiler(config_path=cfg_file, workspace_root=tmp)
        res = compiler.compile_manifold("test_llm", custom_records=[good, good_val, too_long])
        assert res["total_records"] == 2
        q_report = json.loads((Path(res["target_dir"]) / "quality_report.json").read_text("utf-8"))
        assert q_report["curation_stats"]["token_budget_rejected"] == 1
        readme_text = (Path(res["target_dir"]) / "README.md").read_text(encoding="utf-8")
        assert "Total Samples:** 2" in readme_text
        meta_json = json.loads(
            (Path(res["target_dir"]) / "dataset-metadata.json").read_text("utf-8")
        )
        assert any(r["path"] == "data/test_llm_train.jsonl" for r in meta_json["resources"])
        assert meta_json["id"] == f"{kaggle_owner()}/testllm"
        assert "kaggle://" not in readme_text


def test_distill_balanced_parens_and_stage_alignment() -> None:
    """Verify numbered headings, wrapped Markdown, JSON lines, and diag stage order."""
    _env_p, _comp_p, train_p = sidecar_ports()
    body = (
        "Symptom: Sidecar status stays stuck in RUNNING after job completion.\n"
        "Wrapped continuation explains websocket EOF missed.\n\n"
        f'{{"status": "RUNNING", "port": {train_p}}}\n\n'
        "Root Cause: Event loop close skips terminal status broadcast.\n\n"
        "Remediation: Flush terminal event frame and verify via lem-env validate."
    )
    opening, bullets, next_step, tag = distill_section_to_structured_parts(
        doc_id="AI_HELPER_TROUBLESHOOTING_KNOWLEDGE.md",
        header="Doc > 3. Core Diagnostics > Scenario 12 [CONFIRMED]",
        clean_text=body,
        category="troubleshooting_diagnostic",
        gui_state=build_gui_state_snapshot(GuiStateOptions(active_tab="tab-health")),
    )
    reply = format_structured_reply(
        opening,
        bullets,
        next_step,
        ReplyOptions(category="troubleshooting_diagnostic", epistemic_tag=tag, is_diagnostic=True),
    )
    first_line = reply.splitlines()[0]
    assert first_line.count("(") == first_line.count(")")
    assert '{"status":' not in reply
    assert "Stage 1 (Initial Triage)" in reply
    assert "Stage 2 (Targeted Diagnosis)" in reply
    assert "Stage 3 (Bounded Remediation)" in reply


def test_llm_layout_and_api_validate_parity() -> None:
    """Verify LLM layout skips image dirs and API validate enforces copilot rules."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "LemGendizedAIHelperLemGend"
        cfg = {"dataset_type": "llm", "canonical_format": "jsonl"}
        assert _is_llm_dataset(cfg)
        _create_layout(out_dir, cfg, "ai_helper")
        assert not (out_dir / "images").exists()

    non_copilot = chat_record(
        record_id="api_val_1",
        turns=(
            "System [GUI_STATE]{}[/GUI_STATE]",
            "Where do I click?",
            "Short answer without three bullets or next step.",
        ),
        category="gui_procedure",
    )
    resp = validate_llm_corpus(LLMValidationRequest(records=[non_copilot]))
    assert resp["issues_count"] >= 1
    assert any(
        "bullet" in iss["issue"].lower() or "next-step" in iss["issue"].lower()
        for iss in resp["issues"]
    )
