"""Comprehensive Unit and Verification Tests for Universal LLM Dataset Compiler.

Validates:
1. TaxonomyEngine dynamic taxonomy and authority tier resolution.
2. EvolPromptSynthesizer multi-persona query mutation across 4 personas.
3. MultiTurnDialogueSynthesizer stateful multi-turn diagnostic flows.
4. ToolCallingFormatter structured Hermes/Qwen function calling schemas.
5. DPOSynthesizer paired chosen/rejected preference generation.
6. AnchorMixer ratio blending and clean record validation.
"""

from __future__ import annotations

from tools.llm_helper.anchor_mixer import AnchorMixer
from tools.llm_helper.dialogue_synthesizer import (
    DiagnosticScenario,
    MultiTurnDialogueSynthesizer,
)
from tools.llm_helper.dpo_synthesizer import DPOSynthesizer
from tools.llm_helper.evol_synthesizer import EvolPromptSynthesizer
from tools.llm_helper.quality_validator import QualityValidator
from tools.llm_helper.taxonomy import TaxonomyEngine
from tools.llm_helper.tool_calling import ToolCallingFormatter, ToolExecutionPayload


def test_taxonomy_engine_dynamic_configuration() -> None:
    """TaxonomyEngine classifies categories and authority tiers dynamically."""
    custom_categories = ["k8s_manifest", "incident_response", "api_endpoint"]
    custom_tiers = ["core_spec", "runbook", "discussion"]
    engine = TaxonomyEngine(
        domain_name="cloud_infrastructure",
        categories=custom_categories,
        authority_hierarchy=custom_tiers,
        custom_rules=[("k8s_manifest", ("deploy", "pod"), ("yaml", "manifest"))],
    )

    cat = engine.classify_category("deploy-spec", "pod manifest")
    assert cat == "k8s_manifest"

    tier1 = engine.resolve_authority_tier("core_schema.yaml")
    assert tier1 == "core_spec"

    tier2 = engine.resolve_authority_tier("oncall_handbook.md")
    assert tier2 == "runbook"


def test_evol_prompt_synthesizer_multi_personas() -> None:
    """EvolPromptSynthesizer mutates queries across 4 distinct personas."""
    synthesizer = EvolPromptSynthesizer(seed=123)
    base = "How do I configure memory limits in the container?"

    pool = synthesizer.generate_persona_pool(base, context_keywords=["container", "memory"])
    assert len(pool) == 4

    personas = {p.persona for p in pool}
    assert personas == {
        "novice_exploratory",
        "staff_architect",
        "adversarial_boundary",
        "operations_diagnostic",
    }

    for p in pool:
        assert len(p.prompt) > 0
        assert not any(ord(c) > 0x10000 for c in p.prompt)  # No emojis


def test_multi_turn_dialogue_synthesizer_and_validator() -> None:
    """MultiTurnDialogueSynthesizer generates compliant 4-turn stateful dialogues."""
    synth = MultiTurnDialogueSynthesizer(system_prompt="You are an expert diagnostic assistant.")
    scenario = DiagnosticScenario(
        topic="Worker Process",
        symptom="Process exits with 10048 code.",
        diagnosis="Stale socket listener.",
        remediation_steps=[
            "Identify listener PID via netstat.",
            "Terminate stale tree via taskkill.",
        ],
        epistemic_tag="[CONFIRMED]",
    )
    record = synth.build_diagnostic_dialogue(
        record_id="diag_001",
        category="troubleshooting_diagnostic",
        scenario=scenario,
    )

    assert record["id"] == "diag_001"
    assert len(record["messages"]) == 5  # system, user, assistant, user, assistant

    validator = QualityValidator()
    issues = validator.validate_record(record)
    assert len(issues) == 0


def test_tool_calling_formatter() -> None:
    """ToolCallingFormatter structures tools, invocations, and return messages."""
    formatter = ToolCallingFormatter(format_style="hermes")
    tool_def = {
        "type": "function",
        "function": {
            "name": "stop_service",
            "description": "Stop a background sidecar service",
            "parameters": {
                "type": "object",
                "properties": {"service_id": {"type": "string"}},
                "required": ["service_id"],
            },
        },
    }

    payload = ToolExecutionPayload(
        tool_def=tool_def,
        function_name="stop_service",
        call_arguments={"service_id": "dataset-compiler"},
        mock_result={"status": "stopped", "message": "Service terminated"},
        final_assistant_summary="The dataset compiler sidecar has been stopped.",
    )
    record = formatter.build_tool_execution_record(
        record_id="tool_call_001",
        system_prompt="You are an automated copilot.",
        user_query="Stop the dataset compiler immediately.",
        payload=payload,
    )

    assert record["metadata"]["has_tool_call"] is True
    assert record["metadata"]["tool_name"] == "stop_service"
    assert len(record["messages"]) == 5

    validator = QualityValidator()
    issues = validator.validate_record(record)
    assert len(issues) == 0


def test_dpo_preference_pair_generator() -> None:
    """DPOSynthesizer constructs chosen/rejected pairs enforcing negative alignment."""
    dpo = DPOSynthesizer()
    prompt = "What is the consequence of exceeding 1536 sequence tokens?"
    chosen = "[CONFIRMED] Tokens exceeding 1536 are clamped to prevent CUDA out-of-memory errors."

    pair = dpo.build_preference_pair(
        prompt=prompt,
        chosen_response=chosen,
        strategy="uncalibrated_overconfidence",
    )

    assert pair["prompt"] == prompt
    assert pair["chosen"] == chosen
    assert "100% certainty" in pair["rejected"]
    assert pair["strategy"] == "uncalibrated_overconfidence"


def test_anchor_mixer_blending() -> None:
    """AnchorMixer blends general anchors into domain splits while keeping purity."""
    domain_records = [
        {
            "id": f"dom_{i}",
            "category": "general_knowledge",
            "messages": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}."},
            ],
        }
        for i in range(10)
    ]

    anchor_records = [
        {
            "id": f"anc_{i}",
            "category": "general_knowledge",
            "messages": [
                {"role": "user", "content": f"Solve math {i}?"},
                {"role": "assistant", "content": f"Calculated answer {i}."},
            ],
        }
        for i in range(10)
    ]

    mixer = AnchorMixer(mix_ratio=0.2, seed=42)
    blended = mixer.blend_records(domain_records, anchor_records)

    assert len(blended) == 12  # 10 domain + 2 anchors (20%)
    anchor_ids = [r["id"] for r in blended if r["id"].startswith("anc_")]
    assert len(anchor_ids) == 2
