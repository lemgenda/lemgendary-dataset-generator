"""Concise 3-part assistant response formatter for LemGendary AI Studio Copilot.

Enforces zero-emoji sanitization, the 3-part reply structure (1-sentence direct
answer, 3-4 grounded bullet points, 1 proactive next-step recommendation), and
the 150-350 token target band (hard cap 420 tokens) for QLoRA fine-tuning.
"""

from __future__ import annotations

import functools
import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .taxonomy import sidecar_ports
from .token_budget import (
    ASSISTANT_TARGET_MAX_TOKENS,
    ASSISTANT_TARGET_MIN_TOKENS,
    count_tokens,
    truncate_to_token_limit,
)

EMOJI_STRIP_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\uFE0F]",
    flags=re.UNICODE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[`])")
_CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_MULTI_WS_RE = re.compile(r"\s+")
_LEADING_BULLET_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
_JSON_KV_RE = re.compile(r'"[^"]+"\s*:\s*')
_NUMBERED_PREFIX_RE = re.compile(r"(?:^|[\s(>\[`])\d+(?:\.\d+)*$")
_ABBREV_RE = re.compile(r"\b(?:e\.g|i\.e|etc|vs|fig|sec|ch|v\d+(?:\.\d+)*)$", re.I)

_EFFECTIVE_MIN_FALLBACK_TOKENS = ASSISTANT_TARGET_MIN_TOKENS + 30
_SHRINK_TARGET_TOKENS = ASSISTANT_TARGET_MAX_TOKENS - 85


def _balance_parens(text: str) -> str:
    """Drop stray closing ')' and any unclosed trailing '(' clause cleanly."""
    cleaned_chars: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            cleaned_chars.append(ch)
        elif ch == ")":
            if depth > 0:
                depth -= 1
                cleaned_chars.append(ch)
        else:
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    if depth == 0:
        return cleaned

    open_idx = len(cleaned)
    cur_depth = 0
    for idx, ch in enumerate(cleaned):
        if ch == "(":
            if cur_depth == 0:
                open_idx = idx
            cur_depth += 1
        elif ch == ")":
            cur_depth = max(0, cur_depth - 1)
    prefix = cleaned[:open_idx].rstrip(" ,;:-")
    if len(prefix) >= 24:
        return prefix
    return cleaned.replace("(", "").replace(")", "")


def _split_safe_sentences(text: str) -> list[str]:
    """Split prose on sentence boundaries, ignoring parens, decimals, and numbering."""
    sentences: list[str] = []
    start = 0
    paren_depth = 0
    n = len(text)
    for idx, ch in enumerate(text):
        if ch == "(":
            paren_depth += 1
            continue
        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            continue
        if ch not in ".!?" or paren_depth > 0:
            continue
        if idx + 1 >= n or not text[idx + 1].isspace():
            continue
        nxt = text[idx + 1:].lstrip()
        if not nxt or not (nxt[0].isupper() or nxt[0] in "[`\""):
            continue
        prefix = text[start:idx].rstrip()
        if ch == "." and (_NUMBERED_PREFIX_RE.search(prefix) or _ABBREV_RE.search(prefix)):
            continue
        piece = text[start : idx + 1].strip()
        if piece:
            sentences.append(piece)
        start = idx + 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _looks_like_raw_code_or_json(line: str) -> bool:
    """Return True if a candidate line is raw JSON/YAML payload rather than prose."""
    s = line.strip()
    if not s:
        return True
    if s[0] in "{}[]" or _JSON_KV_RE.search(s):
        return True
    if '",.' in s or "\\n" in s or "```" in s:
        return True
    punct = sum(1 for c in s if c in "{}[]\";")
    return punct >= max(4, len(s) // 8)


def _clean_leaf_header(header: str) -> str:
    """Extract a concise leaf section title free of hierarchical numbered breadcrumbs."""
    leaf = header.rsplit(" > ", maxsplit=1)[-1] if " > " in header else header
    cleaned = re.sub(r"^[\d\.\s\-]+", "", leaf).strip()
    cleaned = re.sub(r"\s*\(Part \d+\)$", "", cleaned).strip()
    return strip_emojis(cleaned or leaf).replace("`", "")


@functools.lru_cache(maxsize=1)
def _category_grounding_pool() -> dict[str, tuple[str, ...]]:
    """Build category grounding notes with SSOT sidecar ports from ecosystem.json."""
    env_p, comp_p, train_p = sidecar_ports()
    return {
        "gui_procedure": (
            f"GUI requests route through sidecarFetch with X-LemGendary-Client: "
            f"ai-studio-gui across ports {env_p}, {comp_p}, and {train_p}.",
            "Confirm sidecar health dots in the sidebar before invoking "
            "comp-btn-compile or train-btn-start-{model_key}.",
            "Inspect config-tabs-tab-manifests via sidebar-btn-config (Config & "
            "Secrets) when verifying registry defaults.",
        ),
        "cli_command": (
            "Verify dataset schemas with python cli.py llm validate and compile "
            "manifolds via python cli.py compile or lem.",
            "Check environment alignment with lem-env validate and launch local "
            "training runs via lemtrain train MODEL_KEY.",
            "All Typer CLI commands share the in-process service layer with the "
            "FastAPI sidecar routes.",
        ),
        "api_procedure": (
            f"Sidecar REST APIs bind to {env_p} (Env Manager), {comp_p} (Dataset "
            f"Compiler), and {train_p} (Training Suite) with OpenAPI 3.1 contracts.",
            "Write endpoints require Origin Guard headers and return structured "
            "422 validation errors when preset or ref rules fail.",
            f"Inspect live telemetry via WS /ws/log ({env_p}) or WS /api/ws/events "
            f"({comp_p}/{train_p}) for terminal job status.",
        ),
        "troubleshooting_diagnostic": (
            f"Stage 1 Initial Triage: confirm sidecar ports {env_p}/{comp_p}/"
            f"{train_p} online and verify hardware telemetry before mutating state.",
            "Stage 2 Targeted Diagnosis: inspect training-telemetry-tabs-tab-local "
            "logs, metrics.csv, and active Sawtooth Governor clamps.",
            "Stage 3 Bounded Remediation: apply Monotonic Ratchet holds (never "
            "retreat rung or fraction) and verify with lem-env validate.",
        ),
        "model_pipeline": (
            "Sawtooth Governor preserves the Monotonic Ratchet pair rule: never "
            "step spatial resolution or dataset fraction backwards.",
            "Turing GPUs without Tensor Cores (turing_no_tensor_cores: true) "
            "enforce FP32 compute to prevent NaN gradient overflow.",
            "Release checkpoints support Blend Soup (train-btn-soup-{model_key}) "
            "across vault_psnr and vault_lpips donors.",
        ),
        "terminology_concept": (
            "Authority Hierarchy resolves conflicts strictly as Code Manifests > "
            "Master Manuals > Architectural Whitepapers > Guides.",
            "The 4-Pillar Invariant binds schema validation, split integrity, "
            "epistemic tagging, and zero-emoji compliance.",
            "QLoRA checkpoints optimize r=16, alpha=32 adapters over NF4 base "
            "weights for INT4 ONNX Studio deployment.",
        ),
        "boundary_and_negative": (
            f"Refuse unverified guesses when active_training is null, port {train_p} "
            "is OFFLINE, or sota_targets are absent from context.",
            "Only native containers (WebDataset .data, Apache Parquet .parquet, "
            "LitData, MDS, JSONL) are accepted by Compile Manifold.",
            "Check unified_models_v2.yaml in Config & Secrets (sidebar-btn-config) "
            "or the Target Metrics (SOTA) popover for exact targets.",
        ),
        "general_knowledge": (
            "Consult unified_models_v2.yaml and unified_data.yaml as Tier-1 "
            "Single Source of Truth before overriding defaults.",
            "Use tab-datasets (comp-btn-compile) for manifold builds and "
            "tab-training (Local Training) for Sawtooth-governed runs.",
            "Validate workspace parity with health-btn-validate-ecosystem or "
            "lem-env validate across all sibling projects.",
        ),
    }


@dataclass(frozen=True)
class ReplyOptions:
    """Formatting options for 3-part structured assistant replies."""

    category: str = "general_knowledge"
    epistemic_tag: str | None = None
    is_diagnostic: bool = False
    seed_key: str = ""


@dataclass(frozen=True)
class _ReplyParts:
    """Normalized 3-part reply components before token-band enforcement."""

    opening: str
    bullets: list[str]
    next_step: str


def strip_emojis(text: str) -> str:
    """Remove all Unicode emojis, pictographs, and dingbats from ``text``."""
    if not text:
        return ""
    return EMOJI_STRIP_PATTERN.sub("", text)


def _clean_line(text: str) -> str:
    """Collapse internal whitespace, strip emojis, and strip bullet prefixes."""
    cleaned = strip_emojis(_CODE_FENCE_RE.sub(" ", text))
    cleaned = cleaned.replace("|", " ").replace("`", "")
    cleaned = _MULTI_WS_RE.sub(" ", cleaned).strip()
    cleaned = _LEADING_BULLET_RE.sub("", cleaned).strip()
    return _balance_parens(cleaned)


def _ensure_single_sentence(text: str, epistemic_tag: str | None = None) -> str:
    """Normalize opening to a single direct sentence, optionally prefixed with a tag."""
    cleaned = _clean_line(text)
    if epistemic_tag:
        cleaned = re.sub(r"^\[(CONFIRMED|PROBABLE|ILLUSTRATIVE)\]\s*", "", cleaned).strip()
    sentences = _split_safe_sentences(cleaned) if cleaned else []
    first = sentences[0].strip() if sentences and sentences[0].strip() else cleaned
    first = _balance_parens(first).rstrip(" ,;:")
    if not first:
        first = "Follow the documented LemGendary AI Studio workflow for this operation."
    if first[-1] not in ".!?":
        first += "."
    if epistemic_tag and epistemic_tag not in first:
        return f"{epistemic_tag} {first}"
    return first


def _normalize_next_step(text: str) -> str:
    """Normalize the closing proactive recommendation into one clean line."""
    cleaned = _clean_line(text)
    if not cleaned:
        cleaned = (
            "Next step: verify sidecar health on tab-dashboard and run "
            "compiler-btn-validate-schema before launching."
        )
    if not cleaned.lower().startswith("next step:"):
        cleaned = f"Next step: {cleaned}"
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _normalize_bullets(
    bullets: list[str],
    options: ReplyOptions,
    fallback_seed: str,
) -> list[str]:
    """Ensure 3 to 4 clean single-line bullet points grounded in category rules."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in bullets:
        line = _clean_line(raw)
        if len(line) < 25 or _looks_like_raw_code_or_json(line):
            continue
        if line[-1] not in ".!?":
            line += "."
        norm = line.lower()
        if norm not in seen:
            seen.add(norm)
            cleaned.append(line)
        if len(cleaned) == 4:
            break

    pools = _category_grounding_pool()
    pool = pools.get(options.category, pools["general_knowledge"])
    if options.is_diagnostic and len(cleaned) < 3:
        pool = pools["troubleshooting_diagnostic"]

    seed = options.seed_key or fallback_seed
    offset = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    idx = 0
    while len(cleaned) < 3 and idx < len(pool) * 2:
        candidate = pool[(offset + idx) % len(pool)]
        idx += 1
        if candidate.lower() not in seen:
            seen.add(candidate.lower())
            cleaned.append(candidate)

    return cleaned[:4]


def _assemble_reply(parts: _ReplyParts) -> str:
    """Assemble the canonical 3-part ChatML assistant response."""
    bullet_block = "\n".join(f"- {b}" for b in parts.bullets)
    return f"{parts.opening}\n\n{bullet_block}\n\n{parts.next_step}"


def _safe_truncate_clause(text: str, limit: int) -> str:
    """Truncate ``text`` cleanly without collapsing over an early parenthesis."""
    raw = truncate_to_token_limit(text, limit)
    balanced = _balance_parens(raw)
    if len(balanced) < max(24, int(len(raw) * 0.7)) or count_tokens(balanced) < 16:
        balanced = raw.replace("(", "").replace(")", "")
    cleaned = balanced.rstrip(" .")
    return f"{cleaned}." if cleaned else text


def _shrink_to_target(parts: _ReplyParts, max_tokens: int) -> str:
    """Condense longest bullets then margins until reply fits ``max_tokens``."""
    cur_bullets = list(parts.bullets)
    cur_opening = parts.opening
    cur_next = parts.next_step

    for _ in range(12):
        reply = _assemble_reply(_ReplyParts(cur_opening, cur_bullets, cur_next))
        tokens = count_tokens(reply)
        if tokens <= max_tokens:
            break
        longest_idx = max(range(len(cur_bullets)), key=lambda i: count_tokens(cur_bullets[i]))
        cur_toks = count_tokens(cur_bullets[longest_idx])
        target_toks = max(18, cur_toks - max(12, tokens - max_tokens))
        cur_bullets[longest_idx] = _safe_truncate_clause(cur_bullets[longest_idx], target_toks)
        if cur_toks <= 20:
            break

    reply = _assemble_reply(_ReplyParts(cur_opening, cur_bullets, cur_next))
    if count_tokens(reply) > max_tokens:
        cur_opening = _safe_truncate_clause(cur_opening, 55)
        cur_next = _safe_truncate_clause(cur_next, 55)
        reply = _assemble_reply(_ReplyParts(cur_opening, cur_bullets, cur_next))
    if count_tokens(reply) < ASSISTANT_TARGET_MIN_TOKENS:
        return _append_pad_notes(cur_opening, cur_bullets, cur_next)
    return reply


def _append_bullet_supplement(
    cur_bullets: list[str],
    parts: _ReplyParts,
    supplement: str,
    target_idx: int,
    max_tokens: int,
) -> bool:
    """Try appending a grounding note to one bullet if within ``max_tokens``."""
    if supplement.lower() in cur_bullets[target_idx].lower():
        return False
    candidate = f"{cur_bullets[target_idx].rstrip('.')} ({supplement.rstrip('.')})."
    trial = [*cur_bullets[:target_idx], candidate, *cur_bullets[target_idx + 1:]]
    if count_tokens(_assemble_reply(_ReplyParts(parts.opening, trial, parts.next_step))) <= (
        max_tokens
    ):
        cur_bullets[target_idx] = candidate
        return True
    return False


_PAD_NOTES: tuple[str, ...] = (
    " Always validate schema consistency via compiler-btn-validate-schema "
    "and confirm environment parity with lem-env validate before promotion.",
    " Inspect telemetry under tab-health or tab-dashboard to verify sidecar "
    "ports and hardware constraints match the active manifest.",
    " Consult unified_models_v2.yaml and unified_data.yaml in Config & Secrets "
    "before dispatching local or cloud jobs.",
    " Retain Monotonic Ratchet discipline by never reducing resolution rung "
    "or dataset fraction across curriculum restarts.",
    " Review quality_report.json and curation stats after every manifold "
    "build to guarantee split integrity and sequence compliance.",
)


def _append_pad_notes(opening: str, cur_bullets: list[str], next_step: str) -> str:
    """Append concise operational notes until reply satisfies minimum token floor."""
    cur_next = next_step
    reply = _assemble_reply(_ReplyParts(opening, cur_bullets, cur_next))
    for note in _PAD_NOTES:
        if count_tokens(reply) >= _EFFECTIVE_MIN_FALLBACK_TOKENS:
            break
        candidate = _assemble_reply(
            _ReplyParts(opening, cur_bullets, f"{cur_next.rstrip('.')}.{note}")
        )
        if count_tokens(candidate) > _SHRINK_TARGET_TOKENS and (
            count_tokens(reply) >= ASSISTANT_TARGET_MIN_TOKENS
        ):
            break
        cur_next = f"{cur_next.rstrip('.')}.{note}"
        reply = candidate
    return reply


def _expand_to_minimum(parts: _ReplyParts, options: ReplyOptions) -> str:
    """Add deterministic operational detail until ``ASSISTANT_TARGET_MIN_TOKENS`` is met."""
    cur_bullets = list(parts.bullets)
    pools = _category_grounding_pool()
    pool = [*pools.get(options.category, ()), *pools["general_knowledge"]]
    offset = int(hashlib.sha256((options.seed_key or parts.opening).encode()).hexdigest()[:8], 16)
    existing_lower = {b.lower() for b in cur_bullets}

    pool_idx = 0
    while (
        count_tokens(_assemble_reply(_ReplyParts(parts.opening, cur_bullets, parts.next_step)))
        < _EFFECTIVE_MIN_FALLBACK_TOKENS
    ):
        if len(cur_bullets) < 4 and pool_idx < len(pool):
            extra = pool[(offset + pool_idx) % len(pool)]
            pool_idx += 1
            if extra.lower() not in existing_lower:
                cur_bullets.append(extra)
                existing_lower.add(extra.lower())
                continue

        if pool_idx >= len(pool) * 2:
            break
        supplement = pool[(offset + pool_idx) % len(pool)]
        pool_idx += 1
        _append_bullet_supplement(
            cur_bullets,
            parts,
            supplement,
            pool_idx % len(cur_bullets),
            _SHRINK_TARGET_TOKENS,
        )

    return _append_pad_notes(parts.opening, cur_bullets, parts.next_step)


def format_structured_reply(
    opening: str,
    bullets: list[str],
    next_step: str,
    options: ReplyOptions | None = None,
) -> str:
    """Format a 3-part ChatML assistant response within the 150-350 token target band."""
    opts = options or ReplyOptions()
    clean_opening = _ensure_single_sentence(opening, epistemic_tag=opts.epistemic_tag)
    clean_next = _normalize_next_step(next_step)
    norm_bullets = _normalize_bullets(bullets, opts, fallback_seed=clean_opening)
    parts = _ReplyParts(clean_opening, norm_bullets, clean_next)

    reply = _assemble_reply(parts)
    tokens = count_tokens(reply)
    if tokens < _EFFECTIVE_MIN_FALLBACK_TOKENS:
        reply = _expand_to_minimum(parts, opts)
    elif tokens > _SHRINK_TARGET_TOKENS:
        reply = _shrink_to_target(parts, max_tokens=_SHRINK_TARGET_TOKENS)
    return strip_emojis(reply)


def _reflow_markdown_blocks(clean_text: str) -> list[str]:
    """Strip code fences and join wrapped prose lines into complete logical blocks."""
    stripped = strip_emojis(_CODE_FENCE_RE.sub("\n\n", clean_text))
    blocks: list[str] = []
    para_lines: list[str] = []

    def _flush() -> None:
        if para_lines:
            joined = _MULTI_WS_RE.sub(" ", " ".join(para_lines)).strip()
            if joined:
                blocks.append(joined)
            para_lines.clear()

    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            _flush()
            continue
        if _looks_like_raw_code_or_json(line):
            _flush()
            continue
        if line.startswith("|") and line.endswith("|"):
            _flush()
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            if cells and not all(set(c) <= set("-:") for c in cells):
                blocks.append(": ".join(cells[:3]))
            continue
        if _LEADING_BULLET_RE.match(line):
            _flush()
            para_lines.append(line)
            continue
        para_lines.append(line)
    _flush()
    return blocks


def _extract_candidate_facts(clean_text: str) -> list[str]:
    """Extract clean, complete sentences from reflowed markdown prose blocks."""
    facts: list[str] = []
    for block in _reflow_markdown_blocks(clean_text):
        cleaned = _clean_line(block)
        if len(cleaned) < 28 or _looks_like_raw_code_or_json(cleaned):
            continue
        for sent in _split_safe_sentences(cleaned):
            s = _balance_parens(sent.strip()).rstrip(" ,;:")
            if (
                len(s) < 28
                or len(s.split()) < 5
                or s[0].islower()
                or _looks_like_raw_code_or_json(s)
            ):
                continue
            if s[-1] not in ".!?":
                s += "."
            facts.append(s)
    return facts


def _select_next_step(category: str, clean_h: str, gui_state: dict[str, Any] | None) -> str:
    """Build a tailored proactive next-step recommendation aligned to topic and state."""
    _env_p, _comp_p, train_p = sidecar_ports()
    if gui_state:
        sidecars = gui_state.get("sidecars") or {}
        if sidecars.get(f"training_{train_p}") == "OFFLINE":
            return (
                f"Next step: start the Model Training Suite sidecar on port {train_p} "
                "from tab-dashboard before dispatching training or evaluation."
            )
        active_train = gui_state.get("active_training")
        if (
            category == "model_pipeline"
            and isinstance(active_train, dict)
            and active_train.get("model_key")
        ):
            mkey = active_train["model_key"]
            return (
                f"Next step: monitor {mkey} under training-telemetry-tabs-tab-local "
                f"and use train-btn-eval-{mkey} or Blend Soup once targets converge."
            )
        if gui_state.get("active_tab") == "tab-datasets":
            return (
                "Next step: run compiler-btn-validate-schema and trigger "
                "comp-btn-compile (Compile Manifold) on tab-datasets."
            )

    category_next_steps = {
        "troubleshooting_diagnostic": (
            f"Next step: capture telemetry around {clean_h} and re-run "
            "lem-env validate or test-modal-btn-run-training-unit to confirm recovery."
        ),
        "cli_command": (
            f"Next step: verify flags with python cli.py --help and execute "
            f"lem-env validate after applying {clean_h}."
        ),
        "api_procedure": (
            f"Next step: inspect OpenAPI contracts and sidecar responses for {clean_h} "
            "via WS /api/ws/events or WS /ws/log."
        ),
        "gui_procedure": (
            f"Next step: locate the control in GUI_CONTROL_REGISTRY.md and check "
            f"sidecar port status before triggering {clean_h}."
        ),
    }
    return category_next_steps.get(
        category,
        f"Next step: cross-check {clean_h} against unified_models_v2.yaml or "
        "unified_data.yaml in Config & Secrets (sidebar-btn-config).",
    )


_STAGE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Stage 1 (Initial Triage)", ("symptom", "observed", "signature", "failure", "issue")),
    ("Stage 2 (Targeted Diagnosis)", ("root cause", "cause", "diagnosis", "why", "trigger")),
    (
        "Stage 3 (Bounded Remediation)",
        ("remediation", "fix", "resolution", "mitigation", "workaround", "verify"),
    ),
)
_STAGE_STRIP_RE = re.compile(
    r"^(?:symptom|observed|signature|root cause|cause|diagnosis|"
    r"remediation|fix|resolution|mitigation|verification)\s*:?\s*",
    re.IGNORECASE,
)


def _match_stage_line(first: str, low: str, assigned: list[str]) -> bool:
    """Assign ``first`` to the first matching empty diagnostic stage slot."""
    for idx, (label, cues) in enumerate(_STAGE_CUES):
        if not assigned[idx] and any(c in low for c in cues):
            body = _STAGE_STRIP_RE.sub("", first).strip()
            if len(body) >= 24:
                assigned[idx] = f"{label}: {body}"
                return True
    return False


def _extract_troubleshooting_stages(clean_text: str) -> list[str]:
    """Extract ordered Stage 1/2/3/Guardrail items from troubleshooting prose."""
    diag_pool = _category_grounding_pool()["troubleshooting_diagnostic"]
    assigned: list[str] = ["", "", ""]
    guardrail = ""
    for block in _reflow_markdown_blocks(clean_text):
        cleaned = _clean_line(block)
        if len(cleaned) < 28 or _looks_like_raw_code_or_json(cleaned):
            continue
        low = cleaned.lower()
        if any(
            low.startswith(k)
            for k in ("model:", "dataset:", "hardware:", "tool:", "component:")
        ):
            continue
        first = _split_safe_sentences(cleaned)[0]
        if len(first) < 28:
            continue
        if not _match_stage_line(first, low, assigned) and not guardrail and any(
            k in low for k in ("governor", "ratchet", "invariant", "prevent")
        ):
            guardrail = f"Governor Guardrail: {first}"

    result = [
        assigned[idx] or f"{_STAGE_CUES[idx][0]}: {diag_pool[idx].split(': ', 1)[-1]}"
        for idx in range(3)
    ]
    if guardrail:
        result.append(guardrail)
    return result


def distill_section_to_structured_parts(
    doc_id: str,
    header: str,
    clean_text: str,
    category: str,
    gui_state: dict[str, Any] | None = None,
) -> tuple[str, list[str], str, str | None]:
    """Distill raw documentation prose into concise 3-part copilot components."""
    leaf_h = _clean_leaf_header(header)
    facts = _extract_candidate_facts(clean_text)

    epistemic_tag: str | None = None
    if category == "troubleshooting_diagnostic":
        epistemic_tag = "[CONFIRMED]" if "confirmed" in header.lower() else "[PROBABLE]"
    elif gui_state is not None:
        epistemic_tag = "[CONFIRMED]"

    opening = (
        f"In `{doc_id}` ({leaf_h}), {facts[0]}"
        if facts
        else f"In `{doc_id}`, {leaf_h} defines the canonical studio specification."
    )
    if category == "troubleshooting_diagnostic":
        raw_bullets = _extract_troubleshooting_stages(clean_text)
    else:
        raw_bullets = facts[1:5] if len(facts) > 1 else [f"{leaf_h} specification rule."]

    return opening, raw_bullets, _select_next_step(category, leaf_h, gui_state), epistemic_tag
