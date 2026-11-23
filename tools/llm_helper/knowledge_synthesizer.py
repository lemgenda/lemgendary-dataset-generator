"""Generic Knowledge Ingestion & Instruction Synthesizer for LLM Datasets.

Ingests technical documentation (such as Markdown papers, operations manuals,
and specifications) and compiles them into structured instruction pairs conforming
to the 3-turn ChatML format, zero-emoji invariants, state/RAG copilot grounding,
boundary conditioning, and strict QLoRA sequence-length bounds.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .chunker import RecursiveChunker
from .copilot_grounding import (
    HARDWARE_PROFILES,
    SYSTEM_SYNTHESIS_PROMPT,
    GuiStateOptions,
    RegistryContext,
    build_gui_state_snapshot,
    build_retrieved_docs_snippet,
    format_grounded_system_turn,
    load_registry_context,
    synthesize_boundary_and_negative_records,
    synthesize_state_grounded_copilot_records,
)
from .knowledge_ingestor import KnowledgeDocument, KnowledgeIngestor
from .response_formatter import (
    ReplyOptions,
    distill_section_to_structured_parts,
    format_structured_reply,
    strip_emojis,
)
from .taxonomy import classify_category as _classify_category
from .taxonomy import resolve_authority_tier as _authority_tier
from .token_budget import (
    ASSISTANT_HARD_MAX_TOKENS,
    SYNTHESIS_MAX_TOTAL_TOKENS,
    count_record_tokens,
    count_tokens,
    truncate_to_token_limit,
)

logger = logging.getLogger("lemdata.llm.synthesizer")

__all__ = [
    "SYSTEM_SYNTHESIS_PROMPT",
    "_authority_tier",
    "_classify_category",
    "_iter_sections",
    "compile_knowledge_corpus",
    "synthesize_instruction_from_section",
    "synthesize_instructions_from_section",
]

_PROMPT_TEMPLATES: dict[str, tuple[str, str]] = {
    "cli_command": (
        "What are the CLI operations and flags for {h} in the LemGendary ecosystem?",
        "Which CLI verification flags and safety checks apply to {h} in `{doc_id}`?",
    ),
    "gui_procedure": (
        "How do I operate {h} in the LemGendary AI Studio GUI?",
        "Which GUI controls, sidecar endpoints, and checks govern {h} in `{doc_id}`?",
    ),
    "api_procedure": (
        "What is the specification and procedure for {h} in the LemGendary REST API?",
        "What are the request constraints and status responses for {h} in `{doc_id}`?",
    ),
    "terminology_concept": (
        "Explain the concept and mathematical/architectural definition of {h} in LemGendary AI.",
        "How does {h} constrain training or validation behavior according to `{doc_id}`?",
    ),
    "troubleshooting_diagnostic": (
        "How does LemGendary AI diagnose and remediate: {h}?",
        "Walk through the 3-stage intake protocol and bounded remediation for {h} in `{doc_id}`.",
    ),
}
_DEFAULT_TEMPLATES: tuple[str, str] = (
    "Explain the architectural design and implementation details of {h} from {doc_base}.",
    "What are the operational constraints, verification steps, or key parameters for "
    "{h} in `{doc_id}`?",
)

_HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$")


@dataclass(frozen=True)
class _SectionSynthesisContext:
    """Parameters needed to synthesize grounded primary/operational pairs."""

    doc_id: str
    header: str
    clean_text: str
    system_prompt: str
    index: int
    registry_ctx: RegistryContext | None = None
    ground_with_state: bool | None = None


@dataclass(frozen=True)
class _SectionSource:
    """Source section identity and metadata for ChatML pair generation."""

    doc_id: str
    header: str
    clean_text: str
    base_meta: dict[str, str]
    rec_id: str


@dataclass(frozen=True)
class _PairGrounding:
    """Prompt and state/RAG grounding bundle for one ChatML pair."""

    system_prompt: str
    user_prompt: str
    gui_state: dict[str, Any] | None
    retrieved_docs: str | None
    operational_mode: bool = False


def _clean_content(text: str) -> str:
    """Strip emojis, raw HTML tags, HTML comments, and markdown TOC anchors."""
    clean = strip_emojis(text)
    clean = re.sub(r"<[^>]+>", "", clean)
    clean = re.sub(r"<!--.*?-->", "", clean, flags=re.DOTALL)
    clean = "\n".join(line for line in clean.splitlines() if not line.strip().startswith("* ["))
    return clean.strip()


def _select_section_gui_state(
    category: str,
    index: int,
    registry_ctx: RegistryContext | None,
) -> dict[str, Any]:
    """Build a lightweight Zustand snapshot matched to the section category."""
    hw = dict(HARDWARE_PROFILES[index % len(HARDWARE_PROFILES)])
    model_keys = (
        sorted(registry_ctx.models.keys())
        if registry_ctx and registry_ctx.models
        else ["nafnet_debluring", "codeformer", "ai_helper_lemgend"]
    )
    dataset_keys = (
        sorted(registry_ctx.datasets.keys())
        if registry_ctx and registry_ctx.datasets
        else ["nima_aesthetic", "ai_helper_lemgend"]
    )
    mkey = model_keys[index % len(model_keys)]
    dkey = dataset_keys[index % len(dataset_keys)]

    tab = "tab-training"
    active_training: dict[str, Any] | None = None
    active_compile: dict[str, Any] | None = None
    recent_errors: list[str] = []

    if category in ("gui_procedure", "api_procedure") and index % 2 == 0:
        tab = "tab-datasets"
        active_compile = {"dataset_key": dkey, "status": "READY", "format": "webdataset"}
    elif category == "cli_command":
        tab = "tab-health"
    elif category in ("model_pipeline", "troubleshooting_diagnostic", "general_knowledge"):
        m_info = (registry_ctx.models.get(mkey) if registry_ctx else None) or {}
        targets = dict(m_info.get("sota_targets") or {"eval_loss": 1.15})
        is_diag = category == "troubleshooting_diagnostic"
        active_training = {
            "model_key": mkey,
            "status": "TRAINING",
            "epoch": 10 + (index % 25),
            "budget": 80,
            "rung": 256,
            "fraction": 0.4,
            "governor": "STABILIZATION_COOLDOWN" if is_diag else "NOMINAL_CRUISE",
            "metrics": {},
            "targets": targets,
        }
        if is_diag:
            recent_errors.append(f"Telemetry alert on {mkey}: inspect metrics.csv and governor.")

    return build_gui_state_snapshot(
        GuiStateOptions(
            active_tab=tab,
            hardware=hw,
            active_training=active_training,
            active_compile=active_compile,
            recent_errors=recent_errors,
        )
    )


def _enforce_record_token_budget(
    system_prompt: str,
    gui_state: dict[str, Any] | None,
    retrieved_docs: str | None,
    user_prompt: str,
    assistant_reply: str,
) -> tuple[str, str, str]:
    """Ensure combined ChatML turns remain strictly <= SYNTHESIS_MAX_TOTAL_TOKENS."""
    sys_turn = format_grounded_system_turn(
        system_prompt, gui_state=gui_state, retrieved_docs=retrieved_docs
    )
    total = count_tokens(sys_turn) + count_tokens(user_prompt) + count_tokens(assistant_reply)
    if total <= SYNTHESIS_MAX_TOTAL_TOKENS:
        return sys_turn, user_prompt, assistant_reply

    allowed_sys = max(
        120,
        SYNTHESIS_MAX_TOTAL_TOKENS - count_tokens(user_prompt) - count_tokens(assistant_reply),
    )
    if retrieved_docs:
        sys_without_rag = format_grounded_system_turn(
            system_prompt, gui_state=gui_state, retrieved_docs=None
        )
        rem = allowed_sys - count_tokens(sys_without_rag) - 40
        if rem > 25:
            trimmed_rag = truncate_to_token_limit(retrieved_docs, rem)
            sys_turn = format_grounded_system_turn(
                system_prompt, gui_state=gui_state, retrieved_docs=trimmed_rag
            )
        else:
            sys_turn = sys_without_rag

    if count_tokens(sys_turn) > allowed_sys:
        sys_turn = truncate_to_token_limit(sys_turn, allowed_sys)
    return sys_turn, user_prompt, assistant_reply


def _build_chatml_record(
    meta: dict[str, str],
    turns: tuple[str, str, str],
) -> dict[str, Any]:
    return {
        **meta,
        "messages": [
            {"role": "system", "content": turns[0]},
            {"role": "user", "content": turns[1]},
            {"role": "assistant", "content": turns[2]},
        ],
    }


def _make_section_pair(source: _SectionSource, grounding: _PairGrounding) -> dict[str, Any]:
    """Synthesize one formatted, token-bounded ChatML record for a document section."""
    category = source.base_meta["category"]
    opening, bullets, next_step, epistemic_tag = distill_section_to_structured_parts(
        source.doc_id,
        source.header,
        source.clean_text,
        category,
        gui_state=grounding.gui_state,
    )
    if grounding.operational_mode:
        clean_h = re.sub(r"^[\d\.\s\-]+", "", source.header).strip()
        opening = f"Operational contract for `{source.doc_id}` ({clean_h}): {opening}"
        if len(bullets) >= 2:
            bullets = [*bullets[1:], bullets[0]]

    assistant_reply = format_structured_reply(
        opening,
        bullets,
        next_step,
        ReplyOptions(
            category=category,
            epistemic_tag=epistemic_tag,
            is_diagnostic=category == "troubleshooting_diagnostic",
            seed_key=source.rec_id,
        ),
    )
    turns = _enforce_record_token_budget(
        grounding.system_prompt,
        grounding.gui_state,
        grounding.retrieved_docs,
        strip_emojis(grounding.user_prompt),
        assistant_reply,
    )
    return _build_chatml_record({**source.base_meta, "id": source.rec_id}, turns)


def _is_dense_section(clean_text: str, tier: str) -> bool:
    return (
        len(clean_text.split()) >= 140
        or "```" in clean_text
        or "$$" in clean_text
        or "\\(" in clean_text
        or tier in ("tier_1_manifest", "tier_2_operational_manual")
    )


def _build_section_meta(doc_id: str, header: str) -> dict[str, str]:
    """Build provenance metadata mapping for a document section."""
    return {
        "category": _classify_category(doc_id, header),
        "source_doc": doc_id,
        "section_title": header.rsplit(" > ", maxsplit=1)[-1] if " > " in header else header,
        "heading_path": header,
        "authority_tier": _authority_tier(doc_id),
    }


def _synthesize_operational_pair(
    ctx: _SectionSynthesisContext,
    base_meta: dict[str, str],
    op_tpl: str,
    fields: dict[str, str],
) -> dict[str, Any]:
    """Generate the operational-constraints second variant for dense sections."""
    doc_base = fields["doc_base"].lower()
    op_raw = f"{ctx.doc_id}:{ctx.header}:{ctx.index}:op".encode()
    op_id = f"gen_{doc_base}_{hashlib.sha256(op_raw).hexdigest()[:8]}"
    op_state = _select_section_gui_state(base_meta["category"], ctx.index + 1, ctx.registry_ctx)
    op_user = (
        f"In [GUI_STATE] on `{op_state['active_tab']}` "
        f"(backend={op_state['hardware']['backend']}), {op_tpl.format(**fields)}"
    )
    return _make_section_pair(
        _SectionSource(ctx.doc_id, ctx.header, ctx.clean_text, base_meta, op_id),
        _PairGrounding(
            ctx.system_prompt,
            op_user,
            op_state,
            build_retrieved_docs_snippet(ctx.clean_text, max_sentences=2, max_tokens=90),
            operational_mode=True,
        ),
    )


def _synthesize_from_context(ctx: _SectionSynthesisContext) -> list[dict[str, Any]]:
    """Generate 1 or 2 ChatML records from a cleaned section context."""
    base_meta = _build_section_meta(ctx.doc_id, ctx.header)
    clean_h = re.sub(r"^[\d\.\s\-]+", "", ctx.header).strip()
    primary_tpl, op_tpl = _PROMPT_TEMPLATES.get(base_meta["category"], _DEFAULT_TEMPLATES)
    fields = {"h": clean_h, "doc_id": ctx.doc_id, "doc_base": Path(ctx.doc_id).stem}

    should_ground = (ctx.index % 3 != 0) if ctx.ground_with_state is None else ctx.ground_with_state
    gui_state = (
        _select_section_gui_state(base_meta["category"], ctx.index, ctx.registry_ctx)
        if should_ground
        else None
    )
    primary_user = primary_tpl.format(**fields)
    if gui_state is not None:
        hw = gui_state["hardware"]
        primary_user = (
            f"Given [GUI_STATE] (tab={gui_state['active_tab']}, vram_mb={hw['vram_mb']}, "
            f"turing_no_tensor_cores={hw['turing_no_tensor_cores']}), {primary_user}"
        )

    primary_id = (
        f"gen_{fields['doc_base'].lower()}_"
        f"{hashlib.sha256(f'{ctx.doc_id}:{ctx.header}:{ctx.index}'.encode()).hexdigest()[:8]}"
    )
    records = [
        _make_section_pair(
            _SectionSource(ctx.doc_id, ctx.header, ctx.clean_text, base_meta, primary_id),
            _PairGrounding(
                ctx.system_prompt,
                primary_user,
                gui_state,
                build_retrieved_docs_snippet(ctx.clean_text) if should_ground else None,
            ),
        )
    ]

    if _is_dense_section(ctx.clean_text, base_meta["authority_tier"]):
        records.append(_synthesize_operational_pair(ctx, base_meta, op_tpl, fields))

    return records


def synthesize_instructions_from_section(
    doc_id: str,
    header: str,
    content: str,
    system_prompt: str,
    index: int,
) -> list[dict[str, Any]]:
    """Convert an ingested document section into 1 or 2 state/RAG-grounded ChatML records."""
    clean_text = _clean_content(content)
    if len(clean_text) < 100:
        return []
    return _synthesize_from_context(
        _SectionSynthesisContext(
            doc_id=doc_id,
            header=header,
            clean_text=clean_text,
            system_prompt=system_prompt,
            index=index,
        )
    )


def synthesize_instruction_from_section(
    doc_id: str,
    header: str,
    content: str,
    system_prompt: str,
    index: int,
) -> dict[str, Any] | None:
    """Convert an ingested document section into a structured 3-turn ChatML record."""
    recs = synthesize_instructions_from_section(doc_id, header, content, system_prompt, index)
    return recs[0] if recs else None


def _ingest_sources(knowledge_sources: list[dict[str, Any]], root: Path) -> list[KnowledgeDocument]:
    """Ingest every local or remote source declared in knowledge_sources."""
    ingestor = KnowledgeIngestor()
    all_docs: list[KnowledgeDocument] = []
    for src in knowledge_sources:
        src_type = src.get("type", "")
        if src_type in ("markdown_dir", "text_dir", "file", "dir"):
            all_docs.extend(
                ingestor.ingest_path(
                    root / src.get("path", ""),
                    glob_pattern=src.get("glob", "*.*"),
                    url_prefix=src.get("url_prefix"),
                )
            )
        elif src_type in ("markdown_file", "yaml_file", "json_file", "text_file", "csv_file"):
            docs = ingestor.ingest_path(
                root / src.get("path", ""),
                url_prefix=src.get("url_prefix"),
            )
            if src_type == "yaml_file":
                for d in docs:
                    d.content = _convert_yaml_to_markdown_sections(d.content)
            all_docs.extend(docs)
        elif src_type == "remote_url":
            url = src.get("url")
            if url:
                doc = ingestor.ingest_url(
                    url,
                    tag=src.get("tag"),
                    ttl_hours=int(src.get("cache_ttl_hours", 168)),
                )
                if doc:
                    all_docs.append(doc)
    return all_docs


def _convert_yaml_to_markdown_sections(content: str) -> str:
    """Format structured YAML specifications into sectioned Markdown documents."""
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        return content
    if not isinstance(data, dict):
        return content

    md_blocks: list[str] = []
    if any(k in data for k in ("forex_predictor", "nima_aesthetic", "parsenet", "codeformer")):
        _append_model_manifest_sections(data, md_blocks)
    elif "datasets" in data and isinstance(data.get("datasets"), dict):
        _append_dataset_manifest_sections(data["datasets"], md_blocks)
    elif "platform_capabilities" in data or "desktop_client_baseline" in data:
        _append_capabilities_sections(data, md_blocks)

    return "\n\n".join(md_blocks) if md_blocks else content


def _append_dataset_manifest_sections(
    datasets: dict[str, Any], md_blocks: list[str]
) -> None:
    """Format dataset manifolds from unified_data.yaml into markdown."""
    for d_key, d_info in datasets.items():
        if not isinstance(d_info, dict):
            continue
        name = d_info.get("name", d_key)
        title = d_info.get("title", name)
        task = d_info.get("task", "vision")
        fmt = d_info.get("canonical_format", "webdataset")
        folder = d_info.get("modernized_folder", d_key)
        subtitle = d_info.get("subtitle", "").strip()
        sec = (
            f"## Dataset Manifold Specification: {title} ({d_key})\n\n"
            f"The **{title}** (`{d_key}`) manifold is stored in `{folder}`.\n"
            f"- Task: `{task}`\n"
            f"- Canonical Format: `{fmt}`\n"
            f"- Validation Split: {d_info.get('val_split', 0.12)}\n\n"
        )
        if subtitle:
            sec += f"{subtitle}\n\n"
        md_blocks.append(sec)


def _append_model_manifest_sections(data: dict[str, Any], md_blocks: list[str]) -> None:
    """Format model architectures from unified_models_v2 into markdown."""
    for m_key, m_info in data.items():
        if m_key.startswith("_") or not isinstance(m_info, dict):
            continue
        _append_single_model_section(m_key, m_info, md_blocks)


def _append_single_model_section(m_key: str, m_info: dict[str, Any], md_blocks: list[str]) -> None:
    """Format one model entry from unified_models_v2."""
    name = m_info.get("name", m_key)
    arch = m_info.get("architecture_type", "Neural Architecture")
    params = m_info.get("parameters_m", "N/A")
    loss = m_info.get("loss_fn", "N/A")
    desc = m_info.get("description", "").strip()
    chkpt = m_info.get("checkpoint", "N/A")
    cat = m_info.get("category", "general")
    sec = (
        f"## Neural Model Architecture: {name} ({m_key})\n\n"
        f"The **{name}** (`{m_key}`) is a native architecture in `{cat}`.\n"
        f"- Architecture: {arch}\n"
        f"- Parameters: {params}M\n"
        f"- Loss: {loss}\n"
        f"- Checkpoint: {chkpt}\n\n"
    )
    if desc:
        sec += f"### Description and Operational Scope for {name}\n\n{desc}\n\n"
    if m_info.get("optimization"):
        sec += (
            f"### Optimization Dynamics for {name}\n\n"
            "This model uses Sawtooth Governor training optimization parameters.\n\n"
        )
    md_blocks.append(sec)


def _append_capabilities_sections(data: dict[str, Any], md_blocks: list[str]) -> None:
    """Format governance baselines and platform capabilities into markdown."""
    s_ver = data.get("schema_version")
    d_ver = data.get("document_baseline")
    md_blocks.append(
        "## Canonical System Baselines and Governance\n\n"
        f"Schema Version: {s_ver}\nDocument Baseline: {d_ver}\n\n"
    )
    for m in data.get("models", []):
        _append_capability_model(m, md_blocks)
    for cap in data.get("platform_capabilities", []):
        _append_capability_entry(cap, md_blocks)


def _append_capability_model(m: dict[str, Any], md_blocks: list[str]) -> None:
    """Format model capability entry."""
    mid = m.get("id", "model")
    mname = m.get("name", mid)
    mdom = m.get("domain", "general")
    march = m.get("architecture_type", "Neural Architecture")
    ctrl = m.get("control_planes", {})
    cli_cmd = ctrl.get("cli", {}).get("command", "N/A")
    api_route = ctrl.get("api", {}).get("route", "N/A")
    sec = (
        f"## Canonical Model Specification: {mname} ({mid})\n\n"
        f"Domain: `{mdom}`\nArchitecture: {march}\n"
        f"CLI Command: `{cli_cmd}`\nAPI Route: `{api_route}`\n\n"
    )
    md_blocks.append(sec)


def _append_capability_entry(cap: dict[str, Any], md_blocks: list[str]) -> None:
    """Format single platform capability entry."""
    cid = cap.get("id", "cap")
    cname = cap.get("name", cid)
    owner = cap.get("owner_project", "ecosystem")
    desc = cap.get("delivery", cap.get("request", ""))
    sec = (
        f"## Platform Capability: {cname} ({cid})\n\n"
        f"Maintained by `{owner}`.\n\n"
        f"### Operational Mechanics\n\n{desc}\n\n"
    )
    md_blocks.append(sec)


def _iter_sections(content: str) -> Iterator[tuple[str, str, str]]:
    """Yield ``(leaf_header, full_heading_path, body)`` respecting code fences."""
    in_code_fence = False
    heading_stack: list[str] = []
    current_leaf: str | None = None
    current_path: str | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_code_fence = not in_code_fence
            current_lines.append(line)
            continue

        if in_code_fence:
            current_lines.append(line)
            continue

        m = _HEADER_PATTERN.match(trimmed)
        if not m:
            current_lines.append(line)
            continue

        level = len(m.group(1))
        title = m.group(2).strip()

        if current_leaf and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                yield current_leaf, current_path or current_leaf, body

        while len(heading_stack) >= level:
            heading_stack.pop()
        heading_stack.append(title)

        current_leaf = title
        current_path = " > ".join(heading_stack)
        current_lines = []

    if current_leaf and current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            yield current_leaf, current_path or current_leaf, body


def _synthesize_records(
    docs: list[KnowledgeDocument],
    system_prompt: str,
    registry_ctx: RegistryContext | None = None,
) -> list[dict[str, Any]]:
    """Turn every usable document section into records, chunking oversized sections."""
    chunker = RecursiveChunker(chunk_size=1200, chunk_overlap=200)
    raw_records: list[dict[str, Any]] = []
    for doc in docs:
        for _leaf, heading_path, text in _iter_sections(doc.content):
            if len(text) > 2400:
                sub_doc = KnowledgeDocument(
                    doc_id=doc.doc_id,
                    content=text,
                    source_type=doc.source_type,
                    source_path=doc.source_path,
                    canonical_url=doc.canonical_url,
                )
                for c_idx, chunk in enumerate(chunker.chunk_document(sub_doc), start=1):
                    clean_text = _clean_content(chunk.text)
                    if len(clean_text) < 100:
                        continue
                    raw_records.extend(
                        _synthesize_from_context(
                            _SectionSynthesisContext(
                                doc_id=doc.doc_id,
                                header=f"{heading_path} (Part {c_idx})",
                                clean_text=clean_text,
                                system_prompt=system_prompt,
                                index=len(raw_records),
                                registry_ctx=registry_ctx,
                            )
                        )
                    )
            else:
                clean_text = _clean_content(text)
                if len(clean_text) < 100:
                    continue
                raw_records.extend(
                    _synthesize_from_context(
                        _SectionSynthesisContext(
                            doc_id=doc.doc_id,
                            header=heading_path,
                            clean_text=clean_text,
                            system_prompt=system_prompt,
                            index=len(raw_records),
                            registry_ctx=registry_ctx,
                        )
                    )
                )
    return raw_records


def _round_robin_select(
    raw_records: list[dict[str, Any]], max_samples: int | None = None
) -> list[dict[str, Any]]:
    """Order records deterministically by category and id; optionally cap at ``max_samples``."""
    if max_samples is None:
        return sorted(raw_records, key=lambda x: (str(x["category"]), str(x["id"])))

    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in raw_records:
        by_category.setdefault(r["category"], []).append(r)
    queues = [sorted(by_category[k], key=lambda x: str(x["id"])) for k in sorted(by_category)]

    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < max_samples and any(depth < len(q) for q in queues):
        for queue in queues:
            if len(selected) >= max_samples:
                break
            if depth < len(queue):
                selected.append(queue[depth])
        depth += 1

    selected.sort(key=lambda x: str(x["id"]))
    return selected


def _log_corpus_telemetry(records: list[dict[str, Any]]) -> None:
    """Log copilot grounding ratio, boundary ratio, and token envelope."""
    total = len(records)
    if total == 0:
        return
    grounded = 0
    boundary = 0
    max_total = 0
    max_assistant = 0
    for rec in records:
        if rec.get("category") == "boundary_and_negative":
            boundary += 1
        sys_msg = next(
            (
                str(m.get("content", ""))
                for m in (rec.get("messages") or [])
                if isinstance(m, dict) and m.get("role") == "system"
            ),
            "",
        )
        if "[GUI_STATE]" in sys_msg or "[RETRIEVED_DOCS]" in sys_msg:
            grounded += 1
        t_tok, a_tok = count_record_tokens(rec)
        max_total = max(max_total, t_tok)
        max_assistant = max(max_assistant, a_tok)

    logger.info(
        "Copilot synthesis stats: total=%d grounded=%.1f%% boundary=%.1f%% "
        "max_assistant_tokens=%d (cap=%d) max_total_tokens=%d (cap=%d)",
        total,
        (grounded / total) * 100.0,
        (boundary / total) * 100.0,
        max_assistant,
        ASSISTANT_HARD_MAX_TOKENS,
        max_total,
        SYNTHESIS_MAX_TOTAL_TOKENS,
    )


def compile_knowledge_corpus(
    knowledge_sources: list[dict[str, Any]],
    system_prompt: str,
    max_samples: int | None = None,
    workspace_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Synthesize an instruction dataset from declared knowledge and live state."""
    root = workspace_root or Path(__file__).resolve().parents[3]
    all_docs = _ingest_sources(knowledge_sources, root)
    logger.info("Ingested %d knowledge documents across sources.", len(all_docs))

    registry_ctx = load_registry_context(root, all_docs)
    section_records = _synthesize_records(all_docs, system_prompt, registry_ctx=registry_ctx)
    state_records = synthesize_state_grounded_copilot_records(registry_ctx, system_prompt)
    base_records = section_records + state_records
    if not base_records:
        return []

    boundary_target = max(12, round(len(base_records) / 8.5))
    boundary_records = synthesize_boundary_and_negative_records(
        registry_ctx,
        system_prompt,
        target_count=boundary_target,
    )
    raw_records = base_records + boundary_records
    logger.info(
        "Synthesized %d candidate records (%d section, %d state-grounded, %d boundary).",
        len(raw_records),
        len(section_records),
        len(state_records),
        len(boundary_records),
    )

    selected_records = _round_robin_select(raw_records, max_samples)
    _log_corpus_telemetry(selected_records)
    logger.info("Compiled %d instruction records.", len(selected_records))
    return selected_records
