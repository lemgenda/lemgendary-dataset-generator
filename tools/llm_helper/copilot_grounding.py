"""State-grounded copilot synthesis, RAG snippet formatting, and registry loader.

Grounds ChatML instruction pairs in live Zustand [GUI_STATE] snapshots and
1-3 sentence [RETRIEVED_DOCS] excerpts sourced dynamically from
``unified_models_v2.yaml``, ``unified_data.yaml``, and
``GUI_CONTROL_REGISTRY.md``, and delegates ``boundary_and_negative`` synthesis
to ``boundary_synthesizer.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .boundary_synthesizer import UNSUPPORTED_CONTAINERS
from .boundary_synthesizer import (
    synthesize_boundary_and_negative_records as _synthesize_boundary_records,
)
from .knowledge_ingestor import KnowledgeDocument
from .response_formatter import ReplyOptions, format_structured_reply, strip_emojis
from .taxonomy import HARDWARE_PROFILES, sidecar_ports
from .token_budget import (
    ASSISTANT_HARD_MAX_TOKENS,
    SYNTHESIS_MAX_TOTAL_TOKENS,
    count_record_tokens,
    truncate_to_token_limit,
)

logger = logging.getLogger("lemdata.llm.copilot_grounding")

__all__ = [
    "GUI_TABS",
    "HARDWARE_PROFILES",
    "SYSTEM_SYNTHESIS_PROMPT",
    "UNSUPPORTED_CONTAINERS",
    "GuiStateOptions",
    "RegistryContext",
    "build_gui_state_snapshot",
    "build_retrieved_docs_snippet",
    "format_grounded_system_turn",
    "load_registry_context",
    "synthesize_boundary_and_negative_records",
    "synthesize_state_grounded_copilot_records",
]

SYSTEM_SYNTHESIS_PROMPT = (
    "You are synthesizing high-precision ChatML instruction data for "
    "`ai_helper_lemgend` (Qwen2.5-Coder-1.5B-Instruct, INT4 ONNX), an "
    "embedded copilot inside the LemGendary AI Studio Desktop GUI.\n\n"
    "Convert the provided documentation chunk or registry entry into a JSON "
    "object:\n"
    '{"id": "<deterministic_id>", "category": "<category>", "messages": '
    '[{"role": "system", "content": "..."}, {"role": "user", "content": '
    '"..."}, {"role": "assistant", "content": "..."}]}\n\n'
    "### STRICT CONSTRAINTS:\n"
    "1. ZERO EMOJIS: Never emit any Unicode emoji, pictograph, or dingbat "
    "in any turn.\n"
    "2. STATE & RAG GROUNDING (>= 60% of samples):\n"
    "   - Include a compact `[GUI_STATE]...[/GUI_STATE]` JSON block and/or a "
    "1-3 sentence `[RETRIEVED_DOCS]...[/RETRIEVED_DOCS]` snippet from "
    "`unified_models_v2.yaml`, `unified_data.yaml`, or "
    "`GUI_CONTROL_REGISTRY.md` inside the `system` turn.\n"
    "   - Make the `user` query depend on reading specific fields from "
    "`[GUI_STATE]` (such as `vram_mb: 4096`, `turing_no_tensor_cores: true`, "
    '`governor: "STABILIZATION_COOLDOWN"`, or `training_8200: "OFFLINE"`).\n'
    "3. BOUNDARY & NEGATIVE REFUSALS (~10% of samples):\n"
    "   - When `active_training` is `null`, `training_8200` is `\"OFFLINE\"`, "
    "or the user asks about unsupported formats (`TFRecord`, `HDF5`, `LMDB`, "
    "`RecordIO`), the assistant MUST refuse to guess, state what is missing "
    "or unsupported, and give the exact GUI/CLI remediation step.\n"
    "4. CONCISE 3-PART ASSISTANT FORMAT (150-350 tokens max):\n"
    "   - Opening: 1 direct sentence answering the user's question.\n"
    "   - Body: 3-4 short bullet points referencing exact GUI controls "
    "(`Compile Manifold`, `Local Training`, `Blend Soup`, `Config & Secrets`), "
    "CLI commands (`lemtrain`, `lem-env`, `lem`), or governor invariants "
    "(Monotonic Ratchet, 4-Pillar Invariant, Sawtooth Governor).\n"
    "   - Closing: 1 proactive next-step recommendation.\n"
    "   - For troubleshooting records, include epistemic tags (`[CONFIRMED]`, "
    "`[PROBABLE]`, `[ILLUSTRATIVE]`) and the 3-stage intake protocol (Initial "
    "Triage, Targeted Diagnosis, Bounded Remediation) within the 350-token "
    "limit.\n"
    "5. TOKEN BUDGET:\n"
    "   - Total combined tokens across all turns MUST stay under 1,450 tokens "
    "so `max_seq_length: 2048` never truncates the final assistant completion "
    "during prompt masking."
)

GUI_TABS: tuple[str, ...] = (
    "tab-dashboard",
    "tab-datasets",
    "tab-training",
    "tab-pipeline",
    "tab-health",
)

_GOVERNOR_STATES: tuple[str, ...] = (
    "NOMINAL_CRUISE",
    "STABILIZATION_COOLDOWN",
    "SAWTOOTH_VRAM_CLAMP",
    "LADDER_HOLD",
    "SWA_BLEND_READY",
)

_GUI_CONTROL_LINE_RE = re.compile(
    r"^-\s+`(?P<cid>[^`]+)`\s+\(`(?P<ctype>[^`]+)`\):\s+"
    r"(?:\"(?P<qlabel>[^\"]+)\"|(?P<ulabel>[^.]+))\.\s*(?P<rest>.*)$"
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class RegistryContext:
    """In-memory snapshot of ecosystem models, datasets, and GUI controls."""

    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    gui_controls: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class GuiStateOptions:
    """Structured options for building a Zustand store ``[GUI_STATE]`` snapshot."""

    active_tab: str = "tab-training"
    hardware: dict[str, Any] | None = None
    sidecars: dict[str, str] | None = None
    active_training: dict[str, Any] | None = None
    active_compile: dict[str, Any] | None = None
    recent_errors: list[str] | None = None


def _extract_models_from_yaml(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        k: v
        for k, v in data.items()
        if not k.startswith("_") and isinstance(v, dict) and "name" in v
    }


def _parse_gui_controls(markdown_text: str) -> list[dict[str, str]]:
    """Extract control IDs, labels, and full descriptions from GUI_CONTROL_REGISTRY.md."""
    entries: list[dict[str, str]] = []
    cur_entry: dict[str, str] | None = None
    line_re = re.compile(r"^\s*-\s+`(?P<cid>[^`]+)`\s+\(`(?P<ctype>[^`]+)`\):\s*(?P<rest>.*)$")

    for raw in markdown_text.splitlines():
        stripped = raw.strip()
        match = line_re.match(raw)
        if match:
            if cur_entry:
                entries.append(cur_entry)
            cur_entry = {
                "id": match.group("cid"),
                "type": match.group("ctype"),
                "raw_desc": match.group("rest").strip(),
            }
        elif cur_entry:
            if stripped.startswith("- ") or stripped.startswith("#"):
                entries.append(cur_entry)
                cur_entry = None
            elif stripped:
                cur_entry["raw_desc"] = f"{cur_entry['raw_desc']} {stripped}".strip()

    if cur_entry:
        entries.append(cur_entry)

    controls: list[dict[str, str]] = []
    for entry in entries:
        desc = entry["raw_desc"]
        q_match = re.match(r'^"([^"]+)"\.?\s*(.*)$', desc, re.DOTALL)
        if q_match:
            label, summary = q_match.group(1), q_match.group(2)
        else:
            u_match = re.match(r"^([^.]+)\.\s*(.*)$", desc)
            if u_match:
                label, summary = u_match.group(1), u_match.group(2)
            else:
                label, summary = "", desc
        controls.append(
            {
                "id": entry["id"],
                "type": entry["type"],
                "label": label.strip(),
                "summary": strip_emojis(summary.strip()),
            }
        )
    return controls


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not load YAML registry %s: %s", path, exc)
        return {}


def load_registry_context(
    workspace_root: Path | None = None,
    docs: list[KnowledgeDocument] | None = None,
) -> RegistryContext:
    """Build ``RegistryContext`` from ingested documents with workspace relative fallback."""
    ctx = RegistryContext()
    for doc in docs or ():
        structured = (doc.metadata or {}).get("structured_data")
        if doc.doc_id == "unified_models_v2.yaml" and isinstance(structured, dict):
            ctx.models = _extract_models_from_yaml(structured)
        elif doc.doc_id == "unified_data.yaml" and isinstance(structured, dict):
            datasets = structured.get("datasets")
            if isinstance(datasets, dict):
                ctx.datasets = {k: v for k, v in datasets.items() if isinstance(v, dict)}
        elif doc.doc_id == "GUI_CONTROL_REGISTRY.md":
            ctx.gui_controls = _parse_gui_controls(doc.content)

    root = workspace_root or Path(__file__).resolve().parents[3]
    if not ctx.models:
        models_yaml = _load_yaml_file(
            root / "lemgendary-training-suite" / "unified_models_v2.yaml"
        )
        ctx.models = _extract_models_from_yaml(models_yaml)
    if not ctx.datasets:
        data_yaml = _load_yaml_file(root / "lemgendary-datasets" / "unified_data.yaml")
        datasets = data_yaml.get("datasets")
        if isinstance(datasets, dict):
            ctx.datasets = {k: v for k, v in datasets.items() if isinstance(v, dict)}
    if not ctx.gui_controls:
        reg_path = root / "lemgendary-docs" / "MD-Papers" / "GUI_CONTROL_REGISTRY.md"
        if reg_path.is_file():
            try:
                ctx.gui_controls = _parse_gui_controls(reg_path.read_text(encoding="utf-8"))
            except OSError as exc:
                logger.warning("Could not read GUI registry %s: %s", reg_path, exc)

    return ctx


def build_gui_state_snapshot(options: GuiStateOptions | None = None) -> dict[str, Any]:
    """Construct a compact Zustand store snapshot matching the GUI store schema."""
    opts = options or GuiStateOptions()
    env_p, comp_p, train_p = sidecar_ports()
    return {
        "active_tab": opts.active_tab if opts.active_tab in GUI_TABS else "tab-dashboard",
        "hardware": opts.hardware or dict(HARDWARE_PROFILES[0]),
        "sidecars": opts.sidecars
        or {
            f"env_{env_p}": "ONLINE",
            f"compiler_{comp_p}": "ONLINE",
            f"training_{train_p}": "ONLINE",
        },
        "active_training": opts.active_training,
        "active_compile": opts.active_compile,
        "recent_errors": list(opts.recent_errors or []),
    }


def _clean_markdown_formatting(text: str) -> str:
    """Strip markdown formatting (#, *, `, |) while preserving identifiers."""
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"(?<!\w)[#*`|]+|[#*`|]+(?!\w)", " ", cleaned)
    return re.sub(r"\s+", " ", strip_emojis(cleaned)).strip()


def build_retrieved_docs_snippet(text: str, max_sentences: int = 3, max_tokens: int = 110) -> str:
    """Extract a 1-3 sentence clean excerpt for ``[RETRIEVED_DOCS]`` RAG grounding."""
    cleaned = _clean_markdown_formatting(text)
    if not cleaned:
        return ""
    sentences = [s.strip() for s in _SENTENCE_RE.split(cleaned) if len(s.strip()) >= 15]
    if not sentences:
        return truncate_to_token_limit(cleaned, max_tokens)
    selected = " ".join(sentences[:max_sentences])
    return truncate_to_token_limit(selected, max_tokens)


def format_grounded_system_turn(
    base_system_prompt: str,
    gui_state: dict[str, Any] | None = None,
    retrieved_docs: str | None = None,
) -> str:
    """Append compact ``[GUI_STATE]`` and/or ``[RETRIEVED_DOCS]`` blocks to system turn."""
    parts = [strip_emojis(base_system_prompt).strip()]
    if gui_state is not None:
        state_json = json.dumps(gui_state, separators=(",", ":"), ensure_ascii=True)
        parts.append(f"[GUI_STATE]{state_json}[/GUI_STATE]")
    if retrieved_docs:
        parts.append(f"[RETRIEVED_DOCS]{strip_emojis(retrieved_docs).strip()}[/RETRIEVED_DOCS]")
    return "\n\n".join(p for p in parts if p)


def _deterministic_id(prefix: str, key: str) -> str:
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _sample_active_metrics(targets: dict[str, Any], seed_idx: int) -> dict[str, float]:
    """Synthesize plausible in-progress metric readings relative to SOTA targets."""
    metrics: dict[str, float] = {}
    delta_factor = 0.90 + (seed_idx % 4) * 0.025
    lower_is_better = ("loss", "lpips", "fid", "mae", "nme", "drawdown", "perplexity")
    for metric_name, raw_val in list(targets.items())[:3]:
        if not isinstance(raw_val, (int, float)):
            continue
        val = float(raw_val)
        if any(tok in metric_name.lower() for tok in lower_is_better):
            metrics[metric_name] = round(val * (1.22 - (seed_idx % 4) * 0.05), 4)
        else:
            metrics[metric_name] = round(val * delta_factor, 4)
    return metrics


def _build_model_run_snapshot(
    model_key: str, model_info: dict[str, Any], idx: int
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Build Zustand snapshot, hardware dict, and governor label for one model sample."""
    hw = dict(HARDWARE_PROFILES[idx % len(HARDWARE_PROFILES)])
    governor = _GOVERNOR_STATES[idx % len(_GOVERNOR_STATES)]
    opt = model_info.get("optimization") or {}
    category = str(model_info.get("category") or "").lower()
    if category == "llm" or "llm" in model_key.lower():
        ladder = opt.get("seq_ladder") or [model_info.get("max_seq_length") or 2048]
    else:
        ladder = opt.get("res_ladder") or [model_info.get("val_resolution") or 256]
    rung = ladder[idx % len(ladder)]
    targets = dict(model_info.get("sota_targets") or {})
    recent_errors: list[str] = []
    if governor == "SAWTOOTH_VRAM_CLAMP":
        recent_errors.append(f"CUDA OOM guard clamped batch for {model_key} at {rung}px")
    elif governor == "STABILIZATION_COOLDOWN":
        recent_errors.append(f"Plateau detected on {model_key}; LR cooled by 0.85")

    gui_state = build_gui_state_snapshot(
        GuiStateOptions(
            active_tab="tab-training",
            hardware=hw,
            active_training={
                "model_key": model_key,
                "status": "TRAINING",
                "epoch": 12 + (idx % 40),
                "budget": 60 + (idx % 4) * 20,
                "rung": rung,
                "fraction": round(min(1.0, 0.2 * ((idx % 5) + 1)), 2),
                "governor": governor,
                "metrics": _sample_active_metrics(targets, idx),
                "targets": targets,
            },
            recent_errors=recent_errors,
        )
    )
    return gui_state, hw, governor


def _model_bullets(
    model_key: str,
    run: dict[str, Any],
    hw: dict[str, Any],
    ladder: list[Any],
    primary_metric: str,
) -> list[str]:
    """Build the 4 grounded bullet points for a live model training copilot reply."""
    turing_note = (
        "Turing no-Tensor-Core FP32 override is active for GTX 1650 (4096 MB VRAM)"
        if hw["turing_no_tensor_cores"]
        else f"{hw['gpu']} ({hw['vram_mb']} MB, backend={hw['backend']}) runs native precision"
    )
    return [
        f"State & Hardware: {turing_note}; Monotonic Ratchet locks rung `{run['rung']}` "
        f"and fraction `{run['fraction']}` against backward steps.",
        f"Governor & Metrics: `{run['governor']}` monitors `{primary_metric}` against "
        f"`sota_targets` {json.dumps(run['targets'])} across ladder `{ladder}`.",
        f"GUI & CLI Control: inspect `training-telemetry-tabs-tab-local` or trigger "
        f"`train-btn-eval-{model_key}` (Evaluate Model) / `lemtrain train {model_key}`.",
        f"Checkpoints: after convergence, click `train-btn-soup-{model_key}` (Blend Soup) "
        f"or run `lemtrain soup {model_key}` to ensemble vault checkpoints.",
    ]


def _model_prompt_variant(
    model_key: str,
    run: dict[str, Any],
    hw: dict[str, Any],
    primary_metric: str,
    variant: int,
) -> tuple[str, str, str]:
    """Select user query, opening sentence, and next step for ``variant``."""
    cur_val = run["metrics"].get(primary_metric, "N/A")
    tgt_val = run["targets"].get(primary_metric, "N/A")
    gap_note = (
        f"{round(float(tgt_val) - float(cur_val), 4)}"
        if isinstance(cur_val, (int, float)) and isinstance(tgt_val, (int, float))
        else "pending evaluation"
    )
    if variant == 1:
        prec_mode = (
            "forces FP32 compute via Turing override (no BF16/FP16 AMP overflow)"
            if hw["turing_no_tensor_cores"]
            else f"uses native `{hw['backend']}` precision within `{hw['vram_mb']}` MB VRAM"
        )
        next_action = (
            f"maintain steady monitoring while `{run['governor']}` operates normally"
            if run["governor"] == "NOMINAL_CRUISE"
            else f"wait for `{run['governor']}` to clear"
        )
        return (
            f"Given [GUI_STATE] on `{hw['gpu']}` (vram_mb={hw['vram_mb']}, "
            f"turing_no_tensor_cores={hw['turing_no_tensor_cores']}), how does "
            f"`{model_key}` handle precision and Sawtooth VRAM safety at rung `{run['rung']}`?",
            f"[CONFIRMED] `{model_key}` on `{hw['gpu']}` (`vram_mb={hw['vram_mb']}`) "
            f"{prec_mode} at epoch `{run['epoch']}/{run['budget']}` under `{run['governor']}`.",
            f"Next step: monitor `training-telemetry-tabs-tab-local` for `{model_key}` "
            f"and {next_action} before bumping rung `{run['rung']}`.",
        )
    if variant == 2:
        return (
            f"In [GUI_STATE], `{model_key}` is at fraction `{run['fraction']}` and rung "
            f"`{run['rung']}` under `{run['governor']}`. Can I decrease the rung or "
            "run Blend Soup now?",
            f"[CONFIRMED] `{model_key}` cannot step down from rung `{run['rung']}` "
            f"(fraction `{run['fraction']}`) under the Monotonic Ratchet, and Blend Soup "
            f"waits until `{primary_metric}` converges toward `{tgt_val}`.",
            f"Next step: evaluate `{model_key}` with `train-btn-eval-{model_key}` and run "
            f"`train-btn-soup-{model_key}` (Blend Soup) once plateau settles.",
        )
    wait_action = (
        f"while `{run['governor']}` operates normally"
        if run["governor"] == "NOMINAL_CRUISE"
        else f"until `{run['governor']}` clears cooldown"
    )
    return (
        f"Look at [GUI_STATE] for {model_key} at epoch {run['epoch']}/{run['budget']} "
        f"(rung={run['rung']}, fraction={run['fraction']}, governor={run['governor']}, "
        f"vram_mb={hw['vram_mb']}). What is my gap to target on "
        f"{primary_metric}, and what should I do next?",
        f"[CONFIRMED] `{model_key}` is at epoch {run['epoch']}/{run['budget']} on rung "
        f"`{run['rung']}` (fraction `{run['fraction']}`) under `{run['governor']}` with "
        f"`{primary_metric}={cur_val}` versus SOTA target `{tgt_val}` (delta `{gap_note}`).",
        f"Next step: keep `{model_key}` training {wait_action}, then run "
        f"`train-btn-eval-{model_key}` and Blend Soup.",
    )


def _model_turns(
    model_key: str,
    model_info: dict[str, Any],
    gui_state: dict[str, Any],
    idx: int,
    variant: int = 0,
) -> tuple[str, str, str]:
    """Build (retrieved, user, assistant) turns for a live model training copilot record."""
    run = gui_state["active_training"]
    targets = run["targets"]
    hw = gui_state["hardware"]
    ladder = (model_info.get("optimization") or {}).get("res_ladder") or [run["rung"]]
    retrieved = build_retrieved_docs_snippet(
        f"{model_info.get('name', model_key)} ({model_key}) uses "
        f"{model_info.get('architecture_type', 'Neural Architecture')} with loss "
        f"{model_info.get('loss_fn', 'cross_entropy')}. Resolution ladder: {ladder}. "
        f"SOTA targets: {json.dumps(targets)}."
    )
    primary_metric = next(iter(targets), "eval_loss")
    user_turn, opening, next_step = _model_prompt_variant(
        model_key, run, hw, primary_metric, variant
    )
    assistant_turn = format_structured_reply(
        opening,
        _model_bullets(model_key, run, hw, ladder, primary_metric),
        next_step,
        ReplyOptions(
            category="model_pipeline",
            epistemic_tag="[CONFIRMED]",
            seed_key=f"{model_key}:{idx}:v{variant}:{run['governor']}",
        ),
    )
    return retrieved, user_turn, assistant_turn


def _make_model_state_record(
    model_key: str,
    model_info: dict[str, Any],
    idx: int,
    system_prompt: str,
    variant: int = 0,
) -> dict[str, Any]:
    """Build one live training copilot record grounded in ``unified_models_v2.yaml``."""
    gui_state, _hw, governor = _build_model_run_snapshot(model_key, model_info, idx)
    run = gui_state["active_training"]
    retrieved, user_turn, assistant_turn = _model_turns(
        model_key, model_info, gui_state, idx, variant=variant
    )
    sys_turn = format_grounded_system_turn(
        system_prompt, gui_state=gui_state, retrieved_docs=retrieved
    )
    return {
        "id": _deterministic_id(
            "copilot_train", f"{model_key}:{idx}:v{variant}:{governor}:{run['rung']}"
        ),
        "category": "model_pipeline",
        "source_doc": "unified_models_v2.yaml",
        "section_title": f"State Grounded Training: {model_key}",
        "heading_path": f"unified_models_v2.yaml > {model_key}",
        "authority_tier": "tier_1_manifest",
        "messages": [
            {"role": "system", "content": sys_turn},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": assistant_turn},
        ],
    }


def _dataset_prompt_variant(
    dataset_key: str,
    ds_info: dict[str, Any],
    comp_p: int,
    variant: int,
) -> tuple[str, str, str]:
    """Select user query, opening, and next step for dataset compile ``variant``."""
    fmt = str(ds_info.get("canonical_format", "webdataset"))
    folder = str(ds_info.get("modernized_folder", dataset_key))
    val_split = ds_info.get("val_split", 0.12)
    task = ds_info.get("task", "vision")
    if variant == 1:
        return (
            f"In [GUI_STATE] on `tab-datasets`, `{dataset_key}` targets `{fmt}` in "
            f"`{folder}` (val_split={val_split}). What validation and sidecar contracts apply?",
            f"[CONFIRMED] `{dataset_key}` enforces `{fmt}` container validation, "
            f"`val_split={val_split}`, and sidecar routing on port `{comp_p}` for `{folder}`.",
            f"Next step: run `python cli.py verify --manifold {folder}` (or `python cli.py llm "
            f"validate` for JSONL) before publishing `{dataset_key}`.",
        )
    return (
        f"On tab-datasets, I selected `{dataset_key}` (format `{fmt}`, folder `{folder}`). "
        "Which GUI controls and CLI commands validate and compile this manifold?",
        f"[CONFIRMED] Manifold `{dataset_key}` compiles into `{folder}` using canonical "
        f"format `{fmt}` for `{task}` with validation split `{val_split}`.",
        f"Next step: click `compiler-btn-validate-schema`, then trigger `comp-btn-compile` "
        f"to build `{folder}` on port {comp_p}.",
    )


def _dataset_turns(
    dataset_key: str, ds_info: dict[str, Any], idx: int, variant: int = 0
) -> tuple[str, str, str]:
    """Build (retrieved_docs, user_turn, assistant_turn) for a dataset manifold record."""
    _env_p, comp_p, _train_p = sidecar_ports()
    fmt = str(ds_info.get("canonical_format", "webdataset"))
    folder = str(ds_info.get("modernized_folder", dataset_key))
    retrieved = build_retrieved_docs_snippet(
        f"Manifold {dataset_key} targets {folder} for task {ds_info.get('task', 'vision')} "
        f"in canonical format {fmt} with val_split {ds_info.get('val_split', 0.12)}. "
        f"Dataset Compiler listens on port {comp_p}."
    )
    user_turn, opening, next_step = _dataset_prompt_variant(dataset_key, ds_info, comp_p, variant)
    bullets = [
        f"GUI Pre-Flight: click `compiler-btn-validate-schema` (Validate Schema) on port "
        f"{comp_p} before selecting `{dataset_key}` in `comp-select-manifold`.",
        "GUI Compile: verify `preset-select` and `comp-select-shard-ext` (`.data` default "
        "for WebDataset), then click `comp-btn-compile` (Compile Manifold).",
        f"CLI Equivalent: run `python cli.py compile --model {dataset_key}` (or `lem` / "
        f"`python cli.py llm compile --dataset {dataset_key}` for JSONL).",
        "Verification Gate: after `WS /api/ws/events` finishes, confirm `shard_offsets.json` "
        "and run `python cli.py verify` to enforce the 4-Pillar Invariant.",
    ]
    assistant_turn = format_structured_reply(
        opening,
        bullets,
        next_step,
        ReplyOptions(
            category="gui_procedure",
            epistemic_tag="[CONFIRMED]",
            seed_key=f"ds:{dataset_key}:{idx}:v{variant}",
        ),
    )
    return retrieved, user_turn, assistant_turn


def _make_dataset_state_record(
    dataset_key: str,
    ds_info: dict[str, Any],
    idx: int,
    system_prompt: str,
    variant: int = 0,
) -> dict[str, Any]:
    """Build one live dataset compilation copilot record grounded in ``unified_data.yaml``."""
    fmt = str(ds_info.get("canonical_format", "webdataset"))
    gui_state = build_gui_state_snapshot(
        GuiStateOptions(
            active_tab="tab-datasets",
            hardware=dict(HARDWARE_PROFILES[idx % len(HARDWARE_PROFILES)]),
            active_compile={
                "dataset_key": dataset_key,
                "status": "READY" if variant == 0 else "COMPILING",
                "format": fmt,
                "folder": str(ds_info.get("modernized_folder", dataset_key)),
                "progress_pct": 0 if variant == 0 else 64,
            },
        )
    )
    retrieved, user_turn, assistant_turn = _dataset_turns(
        dataset_key, ds_info, idx, variant=variant
    )
    sys_turn = format_grounded_system_turn(
        system_prompt, gui_state=gui_state, retrieved_docs=retrieved
    )
    return {
        "id": _deterministic_id("copilot_data", f"{dataset_key}:{idx}:v{variant}:{fmt}"),
        "category": "gui_procedure",
        "source_doc": "unified_data.yaml",
        "section_title": f"State Grounded Compile: {dataset_key}",
        "heading_path": f"unified_data.yaml > {dataset_key}",
        "authority_tier": "tier_1_manifest",
        "messages": [
            {"role": "system", "content": sys_turn},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": assistant_turn},
        ],
    }


def _resolve_control_tab(cid: str, idx: int) -> str:
    if cid.startswith(("comp-", "compiler-", "custom-")):
        return "tab-datasets"
    if cid.startswith(("train-", "soup-", "kaggle-")):
        return "tab-training"
    if cid.startswith("health-"):
        return "tab-health"
    if cid.startswith("pipeline-"):
        return "tab-pipeline"
    return GUI_TABS[idx % len(GUI_TABS)]


def _gui_control_bullets(control: dict[str, str], tab: str) -> list[str]:
    """Build control-specific grounded bullets citing registry contract and SSOT ports."""
    env_p, comp_p, train_p = sidecar_ports()
    cid = control["id"]
    ctype = control["type"]
    label = control["label"]
    summary = control["summary"].strip() or f"Operates `{cid}` inside `{tab}`."
    bullets = [
        f"Registry Contract: `{cid}` (`{ctype}`, label \"{label}\") in "
        "GUI_CONTROL_REGISTRY.md sends `X-LemGendary-Client: ai-studio-gui`.",
        f"Behavior Scope: {truncate_to_token_limit(summary, 55)}.",
    ]
    if tab == "tab-datasets":
        bullets.append(
            f"Compiler Sidecar: routes via `sidecarFetch` to port {comp_p} and works "
            "with `comp-select-manifold`, `compiler-btn-validate-schema`, and `comp-btn-compile`."
        )
    elif tab == "tab-training":
        bullets.append(
            f"Training Sidecar: coordinates port {train_p} (`Local Training`, "
            "Monotonic Ratchet, and `Blend Soup`) with telemetry streamed on `WS /api/ws/events`."
        )
    else:
        bullets.append(
            f"Sidecar Mesh: `{tab}` monitors Environment Manager ({env_p}), Dataset "
            f"Compiler ({comp_p}), and Training Suite ({train_p}) via `header-btn-test-modal`."
        )
    return bullets


def _resolve_behavior_clause(summary: str) -> str:
    """Return the control behavior description derived from its summary text."""
    _none_dot_prefix = "None."
    _none_prefix = "None"
    if summary.startswith(_none_dot_prefix):
        behavior_text = summary.removeprefix(_none_dot_prefix).strip()
    elif summary.startswith(_none_prefix):
        behavior_text = summary.removeprefix(_none_prefix).strip(" :.,")
    else:
        return f"executes: {summary}"
    if behavior_text:
        return f"performs client-side GUI action: {behavior_text}"
    return "manages local GUI state"


def _make_gui_control_record(
    control: dict[str, str],
    idx: int,
    system_prompt: str,
) -> dict[str, Any]:
    """Build a grounded GUI control workflow record from ``GUI_CONTROL_REGISTRY.md``."""
    cid = control["id"]
    label = control["label"]
    summary = control["summary"]
    tab = _resolve_control_tab(cid, idx)

    gui_state = build_gui_state_snapshot(
        GuiStateOptions(
            active_tab=tab,
            hardware=dict(HARDWARE_PROFILES[idx % len(HARDWARE_PROFILES)]),
        )
    )
    retrieved = build_retrieved_docs_snippet(
        f"Control `{cid}` ({control['type']}) labeled \"{label}\": {summary}"
    )
    sys_turn = format_grounded_system_turn(
        system_prompt, gui_state=gui_state, retrieved_docs=retrieved
    )
    user_turn = (
        f"On `{tab}`, what does the `{cid}` control (\"{label}\") do and how does it "
        "interact with the sidecar mesh?"
    )
    behavior_clause = _resolve_behavior_clause(summary)
    opening = (
        f"[CONFIRMED] Control `{cid}` (\"{label}\") on `{tab}` "
        f"{truncate_to_token_limit(behavior_clause, 32)}."
    )
    next_step = (
        f"Next step: confirm required sidecar dots are online on `tab-dashboard` "
        f"before triggering `{cid}` (\"{label}\") on `{tab}`."
    )
    assistant_turn = format_structured_reply(
        opening,
        _gui_control_bullets(control, tab),
        next_step,
        ReplyOptions(
            category="gui_procedure",
            epistemic_tag="[CONFIRMED]",
            seed_key=f"ctrl:{cid}:{idx}",
        ),
    )
    return {
        "id": _deterministic_id("copilot_gui", f"{cid}:{idx}"),
        "category": "gui_procedure",
        "source_doc": "GUI_CONTROL_REGISTRY.md",
        "section_title": f"GUI Control: {cid}",
        "heading_path": f"GUI_CONTROL_REGISTRY.md > {cid}",
        "authority_tier": "tier_2_operational_manual",
        "messages": [
            {"role": "system", "content": sys_turn},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": assistant_turn},
        ],
    }


def synthesize_state_grounded_copilot_records(
    ctx: RegistryContext,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Synthesize live state-grounded records across models, datasets, and GUI controls."""
    records: list[dict[str, Any]] = []
    for idx, (m_key, m_info) in enumerate(sorted(ctx.models.items())):
        for variant in (0, 1, 2):
            records.append(
                _make_model_state_record(
                    m_key, m_info, idx + variant * 3, system_prompt, variant=variant
                )
            )

    for idx, (d_key, d_info) in enumerate(sorted(ctx.datasets.items())):
        for variant in (0, 1):
            records.append(
                _make_dataset_state_record(d_key, d_info, idx, system_prompt, variant=variant)
            )

    for idx, ctrl in enumerate(ctx.gui_controls):
        records.append(_make_gui_control_record(ctrl, idx, system_prompt))

    return [r for r in records if _within_synthesis_budget(r)]


def synthesize_boundary_and_negative_records(
    ctx: RegistryContext,
    system_prompt: str,
    target_count: int,
) -> list[dict[str, Any]]:
    """Delegate diverse boundary and negative refusal synthesis."""
    return _synthesize_boundary_records(ctx.models, ctx.datasets, system_prompt, target_count)


def _within_synthesis_budget(record: dict[str, Any]) -> bool:
    total_tokens, assistant_tokens = count_record_tokens(record)
    return (
        assistant_tokens <= ASSISTANT_HARD_MAX_TOKENS
        and total_tokens <= SYNTHESIS_MAX_TOTAL_TOKENS
    )
