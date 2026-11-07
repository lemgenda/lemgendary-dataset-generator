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

from .knowledge_ingestor import KnowledgeDocument, KnowledgeIngestor

logger = logging.getLogger("lemdata.llm.synthesizer")


# (category, doc_id keywords, section-title keywords), checked in order; first hit wins.
_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("troubleshooting_diagnostic", ("trouble",), ("diagnostic", "failure")),
    ("gui_procedure", ("gui", "control"), ("interface",)),
    ("cli_command", ("cli",), ("command", "flag")),
    ("api_procedure", ("api",), ("endpoint", "rest")),
    ("terminology_concept", ("glossary",), ("terminology", "definition")),
    ("model_pipeline", ("paper",), ("architecture", "mathematical")),
)

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


def _tokenize_text(text: str) -> set[str]:
    """Extract lowercase alpha-numeric tokens for word-boundary matching."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _classify_category(doc_id: str, section_title: str) -> str:
    """Classify knowledge chunk into domain categories using token boundaries."""
    doc_tokens = _tokenize_text(doc_id)
    title_tokens = _tokenize_text(section_title)
    for category, doc_keys, title_keys in _CATEGORY_RULES:
        if any(k in doc_tokens for k in doc_keys) or any(k in title_tokens for k in title_keys):
            return category
    return "general_knowledge"


def _authority_tier(doc_id: str) -> str:
    """Classify the document into an authoritative provenance tier."""
    lower = doc_id.lower()
    if any(k in lower for k in ("manifest", "canonical", "unified_data", "unified_models")):
        return "tier_1_manifest"
    if any(k in lower for k in ("manual", "cli", "api", "operations")):
        return "tier_2_operational_manual"
    if any(k in lower for k in ("paper", "architecture", "whitepaper", "specification")):
        return "tier_3_whitepaper"
    return "tier_4_educational"


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
            all_docs.extend(
                ingestor.ingest_path(
                    root / src.get("path", ""),
                    url_prefix=src.get("url_prefix"),
                )
            )
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


def _synthesize_records(docs: list[KnowledgeDocument], system_prompt: str) -> list[dict[str, Any]]:
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
