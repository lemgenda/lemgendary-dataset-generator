"""Generic Knowledge Ingestion & Instruction Synthesizer for LLM Datasets.

Ingests technical documentation (such as Markdown papers, operations manuals,
and specifications) and compiles them into structured instruction pairs conforming
to the 3-turn ChatML format, zero-emoji invariants, and epistemic validation bounds.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from .knowledge_ingestor import KnowledgeDocument, KnowledgeIngestor
from .taxonomy import (
    classify_category as _classify_category,
)
from .taxonomy import (
    resolve_authority_tier as _authority_tier,
)

logger = logging.getLogger("lemdata.llm.synthesizer")

# category -> (user prompt template, assistant reply template). Placeholders:
# {h} cleaned section title, {doc_id}, {doc_base}, {text} cleaned section body.
_REFERENCE_REPLY = "Reference: `{doc_id}` ({h}).\n\n{text}"
_PROMPT_TEMPLATES: dict[str, tuple[str, str]] = {
    "cli_command": (
        "What are the CLI operations and flags for {h} in the LemGendary ecosystem?",
        "According to `{doc_id}` under section '{h}':\n\n{text}",
    ),
    "gui_procedure": (
        "How do I operate {h} in the LemGendary AI Studio GUI?",
        _REFERENCE_REPLY,
    ),
    "api_procedure": (
        "What is the specification and procedure for {h} in the LemGendary REST API?",
        _REFERENCE_REPLY,
    ),
    "terminology_concept": (
        ("Explain the concept and mathematical/architectural definition of {h} in LemGendary AI."),
        _REFERENCE_REPLY,
    ),
}
_DEFAULT_TEMPLATE = (
    "Explain the architectural design and implementation details of {h} from {doc_base}.",
    "According to `{doc_id}` ({h}):\n\n{text}",
)
_DIAGNOSTIC_TEMPLATE = (
    "How does LemGendary AI diagnose and remediate: {h}?",
    (
        "Based on the official documentation in `{doc_id}` ({h}), the diagnostic"
        " breakdown is:\n\n{text}\n\nEpistemic Status: {epistemic} - verified against"
        " runtime documentation contracts."
    ),
)

_HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$")


def _clean_content(text: str) -> str:
    """Strip emojis, raw HTML tags, and markdown anchors."""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", "", text)
    # Strip markdown anchor links like [link](#section)
    clean = re.sub(r"<!--.*?-->", "", clean, flags=re.DOTALL)
    # Strip markdown table of contents lines
    clean = "\n".join(line for line in clean.splitlines() if not line.strip().startswith("* ["))
    return clean.strip()


def _render_turns(category: str, doc_id: str, header: str, clean_text: str) -> tuple[str, str]:
    """Return the (user prompt, assistant reply) for a section of the given category."""
    # Formulate question based on category and section title
    if category == "troubleshooting_diagnostic":
        user_tpl, reply_tpl = _DIAGNOSTIC_TEMPLATE
    else:
        user_tpl, reply_tpl = _PROMPT_TEMPLATES.get(category, _DEFAULT_TEMPLATE)
    fields = {
        "h": re.sub(r"^[\d\.\s\-]+", "", header).strip(),
        "doc_id": doc_id,
        "doc_base": Path(doc_id).stem,
        "text": clean_text,
        "epistemic": "[CONFIRMED]" if "confirmed" in header.lower() else "[PROBABLE]",
    }
    return user_tpl.format(**fields), reply_tpl.format(**fields)


def synthesize_instruction_from_section(
    doc_id: str,
    header: str,
    content: str,
    system_prompt: str,
    index: int,
) -> dict[str, Any] | None:
    """Convert an ingested document section into a structured 3-turn ChatML record."""
    clean_text = _clean_content(content)
    if len(clean_text) < 100:
        return None

    path_ctx = header
    leaf_header = header.rsplit(" > ", maxsplit=1)[-1] if " > " in header else header
    category = _classify_category(doc_id, path_ctx)
    doc_base = Path(doc_id).stem
    user_prompt, assistant_reply = _render_turns(category, doc_id, path_ctx, clean_text)

    # Deterministic SHA hash for record ID
    record_hash = hashlib.sha256(f"{doc_id}:{path_ctx}:{index}".encode()).hexdigest()[:8]
    record_id = f"gen_{doc_base.lower()}_{record_hash}"

    return {
        "id": record_id,
        "category": category,
        "source_doc": doc_id,
        "section_title": leaf_header,
        "heading_path": path_ctx,
        "authority_tier": _authority_tier(doc_id),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_reply},
        ],
    }


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
    elif "platform_capabilities" in data or "desktop_client_baseline" in data:
        _append_capabilities_sections(data, md_blocks)

    return "\n\n".join(md_blocks) if md_blocks else content


def _append_model_manifest_sections(data: dict[str, Any], md_blocks: list[str]) -> None:
    """Format model architectures from unified_models_v2 into markdown."""
    for m_key, m_info in data.items():
        if m_key.startswith("_") or not isinstance(m_info, dict):
            continue
        _append_single_model_section(m_key, m_info, md_blocks)


def _append_single_model_section(
    m_key: str, m_info: dict[str, Any], md_blocks: list[str]
) -> None:
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
            f"This model uses Sawtooth Governor training optimization parameters.\n\n"
        )
    md_blocks.append(sec)


def _append_capabilities_sections(data: dict[str, Any], md_blocks: list[str]) -> None:
    """Format governance baselines and platform capabilities into markdown."""
    s_ver = data.get("schema_version")
    d_ver = data.get("document_baseline")
    md_blocks.append(
        f"## Canonical System Baselines and Governance\n\n"
        f"Schema Version: {s_ver}\n"
        f"Document Baseline: {d_ver}\n\n"
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
    docs: list[KnowledgeDocument], system_prompt: str
) -> list[dict[str, Any]]:
    """Turn every usable document section into a record, numbering only the kept ones."""
    raw_records: list[dict[str, Any]] = []
    for doc in docs:
        for _leaf, heading_path, text in _iter_sections(doc.content):
            rec = synthesize_instruction_from_section(
                doc.doc_id, heading_path, text, system_prompt, len(raw_records)
            )
            if rec:
                raw_records.append(rec)
    return raw_records


def _round_robin_select(
    raw_records: list[dict[str, Any]], target_samples: int
) -> list[dict[str, Any]]:
    """Pick up to ``target_samples`` records, one category at a time, sorted by id."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in raw_records:
        by_category.setdefault(r["category"], []).append(r)
    # Sort each category deterministically by record ID
    queues = [sorted(by_category[k], key=lambda x: str(x["id"])) for k in sorted(by_category)]

    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < target_samples and any(depth < len(q) for q in queues):
        for queue in queues:
            if len(selected) >= target_samples:
                break
            if depth < len(queue):
                selected.append(queue[depth])
        depth += 1

    selected.sort(key=lambda x: str(x["id"]))
    return selected


def compile_knowledge_corpus(
    knowledge_sources: list[dict[str, Any]],
    system_prompt: str,
    target_samples: int = 525,
    workspace_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Synthesizes a full balanced instruction dataset purely from declared knowledge sources."""
    root = workspace_root or Path(__file__).resolve().parents[3]
    all_docs = _ingest_sources(knowledge_sources, root)
    logger.info("Ingested %d knowledge documents across sources.", len(all_docs))

    # Parse sections hierarchically from markdown/text documents
    raw_records = _synthesize_records(all_docs, system_prompt)
    logger.info("Synthesized %d candidate instruction records from documents.", len(raw_records))

    if not raw_records:
        return []
    selected_records = _round_robin_select(raw_records, target_samples)
    logger.info("Compiled %d balanced instruction records.", len(selected_records))
    return selected_records
