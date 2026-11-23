"""Quality and Invariant Validator for LLM Instruction Datasets.

Enforces zero-emoji rules, ChatML conversation schemas, 3-Stage Diagnostic
Intake Protocols with epistemic tagging, 3-part concise copilot reply structure,
sequence length alignment (1,536 total / 420 assistant hard ceiling), and
duplicate ID detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .token_budget import (
    ASSISTANT_HARD_MAX_TOKENS,
    ASSISTANT_TARGET_MAX_TOKENS,
    ASSISTANT_TARGET_MIN_TOKENS,
    MAX_TOTAL_TOKENS,
    count_record_tokens,
)

EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\uFE0F]",
    flags=re.UNICODE,
)
EPISTEMIC_TAG_PATTERN = re.compile(r"\[(CONFIRMED|PROBABLE|ILLUSTRATIVE)\]")
_BULLET_LINE_RE = re.compile(r"^\s*[-*]\s+\S")


# Each stage of the Diagnostic Intake Protocol is present when any of its markers is.
_DIAG_STAGE_MARKERS = (
    ("stage 1", "triage"),
    ("stage 2", "diagnosis"),
    ("stage 3", "remediation"),
)


@dataclass(frozen=True)
class TokenValidationConfig:
    """Sequence-length and copilot structure thresholds for ``QualityValidator``."""

    max_total_tokens: int = MAX_TOTAL_TOKENS
    max_assistant_tokens: int = ASSISTANT_HARD_MAX_TOKENS
    target_assistant_min_tokens: int = ASSISTANT_TARGET_MIN_TOKENS
    target_assistant_max_tokens: int = ASSISTANT_TARGET_MAX_TOKENS
    enforce_copilot_structure: bool = False


@dataclass(frozen=True)
class ValidationIssue:
    """One invariant violation: the record, the offending field, and what is wrong."""

    record_id: str
    field: str
    issue: str
    severity: str = "ERROR"

    def __repr__(self) -> str:
        return f"[{self.severity}] {self.record_id} ({self.field}): {self.issue}"


def _collect_message_text(
    rec_id: str, messages: list[Any], issues: list[ValidationIssue]
) -> tuple[str, str, str]:
    """Flag malformed or empty turns; return (all text, system text, assistant text)."""
    full_text = ""
    system_text = ""
    assistant_text = ""
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            issues.append(
                ValidationIssue(rec_id, f"messages[{idx}]", "Message must be a dictionary")
            )
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or not content.strip():
            issues.append(ValidationIssue(rec_id, f"messages[{idx}]", "Message content is empty"))
        role = m.get("role")
        full_text += " " + str(content)
        if role == "system":
            system_text += " " + str(content)
        elif role == "assistant":
            assistant_text += ("\n" if assistant_text else "") + str(content)
    return full_text, system_text, assistant_text


def _copilot_structure_issues(rec_id: str, assistant_text: str) -> list[ValidationIssue]:
    """Warn if a copilot reply deviates from the 1-opening / 3-4 bullets / 1-next-step layout."""
    lines = [line.strip() for line in assistant_text.splitlines() if line.strip()]
    if not lines:
        return []

    bullet_indices = [i for i, ln in enumerate(lines) if _BULLET_LINE_RE.match(ln)]
    issues: list[ValidationIssue] = []
    if not 3 <= len(bullet_indices) <= 4:
        issues.append(
            ValidationIssue(
                rec_id,
                "assistant",
                f"Expected 3-4 bullet points in copilot reply, found {len(bullet_indices)}",
                severity="WARNING",
            )
        )
        return issues

    first_bullet = bullet_indices[0]
    last_bullet = bullet_indices[-1]
    if first_bullet == 0:
        issues.append(
            ValidationIssue(
                rec_id,
                "assistant",
                "Missing 1-sentence direct opening before bullet points",
                severity="WARNING",
            )
        )
    if last_bullet >= len(lines) - 1:
        issues.append(
            ValidationIssue(
                rec_id,
                "assistant",
                "Missing proactive next-step closing recommendation after bullet points",
                severity="WARNING",
            )
        )
    return issues


class QualityValidator:
    """Enforces strict schema and content invariants over LLM dialogue records."""

    def __init__(
        self,
        zero_emojis: bool = True,
        enforce_epistemic_tags: bool = True,
        enforce_diag_stages: bool = True,
        token_config: TokenValidationConfig | None = None,
    ) -> None:
        self.zero_emojis = zero_emojis
        self.enforce_epistemic_tags = enforce_epistemic_tags
        self.enforce_diag_stages = enforce_diag_stages
        self.token_config = token_config or TokenValidationConfig()

    def validate_record(self, record: dict[str, Any]) -> list[ValidationIssue]:
        """Return the quality issues found in one LLM training record."""
        issues: list[ValidationIssue] = []
        rec_id = str(record.get("id", "UNKNOWN_ID"))

        if not record.get("id"):
            issues.append(ValidationIssue(rec_id, "id", "Missing required record id"))

        if not record.get("category"):
            issues.append(ValidationIssue(rec_id, "category", "Missing required category"))

        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            issues.append(ValidationIssue(rec_id, "messages", "Must have at least 2 message turns"))
            return issues

        roles = [m.get("role") for m in messages if isinstance(m, dict)]
        if roles not in (["system", "user", "assistant"], ["user", "assistant"]):
            issues.append(ValidationIssue(rec_id, "messages", f"Unexpected role sequence: {roles}"))

        full_text, system_text, assistant_text = _collect_message_text(rec_id, messages, issues)

        if self.zero_emojis:
            emojis = EMOJI_PATTERN.findall(full_text)
            if emojis:
                issues.append(
                    ValidationIssue(
                        rec_id, "content", f"Found {len(emojis)} illegal emoji characters: {emojis}"
                    )
                )

        issues.extend(self._token_budget_issues(record, rec_id, system_text, assistant_text))

        category = str(record.get("category", "")).strip().lower()
        if category in ("troubleshooting_diagnostic", "troubleshooting_diagnostics"):
            issues.extend(self._diagnostic_issues(rec_id, assistant_text))

        return issues

    def _token_budget_issues(
        self,
        record: dict[str, Any],
        rec_id: str,
        system_text: str,
        assistant_text: str,
    ) -> list[ValidationIssue]:
        """Enforce sequence alignment ceilings and copilot reply target structure."""
        issues: list[ValidationIssue] = []
        cfg = self.token_config
        total_tokens, assistant_tokens = count_record_tokens(record)

        if total_tokens > cfg.max_total_tokens:
            issues.append(
                ValidationIssue(
                    rec_id,
                    "tokens",
                    f"Total sequence tokens ({total_tokens}) exceed hard limit "
                    f"({cfg.max_total_tokens})",
                )
            )
        if assistant_tokens > cfg.max_assistant_tokens:
            issues.append(
                ValidationIssue(
                    rec_id,
                    "assistant",
                    f"Assistant turn tokens ({assistant_tokens}) exceed hard ceiling "
                    f"({cfg.max_assistant_tokens})",
                )
            )

        category = str(record.get("category", "")).strip().lower()
        is_copilot_record = (
            cfg.enforce_copilot_structure
            or "[GUI_STATE]" in system_text
            or "[RETRIEVED_DOCS]" in system_text
            or category == "boundary_and_negative"
        )
        if not is_copilot_record:
            return issues

        if assistant_tokens < cfg.target_assistant_min_tokens:
            issues.append(
                ValidationIssue(
                    rec_id,
                    "assistant",
                    f"Assistant turn tokens ({assistant_tokens}) below target minimum "
                    f"({cfg.target_assistant_min_tokens})",
                    severity="WARNING",
                )
            )
        elif assistant_tokens > cfg.target_assistant_max_tokens:
            issues.append(
                ValidationIssue(
                    rec_id,
                    "assistant",
                    f"Assistant turn tokens ({assistant_tokens}) above target maximum "
                    f"({cfg.target_assistant_max_tokens})",
                    severity="WARNING",
                )
            )

        issues.extend(_copilot_structure_issues(rec_id, assistant_text))
        return issues

    def _diagnostic_issues(self, rec_id: str, assistant_text: str) -> list[ValidationIssue]:
        """Warnings for a troubleshooting reply without epistemic tags or the 3 stages."""
        issues: list[ValidationIssue] = []
        if self.enforce_epistemic_tags and not EPISTEMIC_TAG_PATTERN.findall(assistant_text):
            issues.append(
                ValidationIssue(
                    rec_id,
                    "assistant",
                    "Missing epistemic tag in troubleshooting response",
                    severity="WARNING",
                )
            )

        lowered = assistant_text.lower()
        if self.enforce_diag_stages and not all(
            any(marker in lowered for marker in stage) for stage in _DIAG_STAGE_MARKERS
        ):
            issues.append(
                ValidationIssue(
                    rec_id,
                    "assistant",
                    "Incomplete 3-Stage Diagnostic Intake Protocol in troubleshooting"
                    " response",
                    severity="WARNING",
                )
            )
        return issues

    def validate_corpus(self, records: list[dict[str, Any]]) -> tuple[bool, list[ValidationIssue]]:
        """Validate an entire collection of records, checking uniqueness and compliance."""
        all_issues: list[ValidationIssue] = []
        seen_ids: set[str] = set()
        seen_prompts: dict[str, str] = {}

        for rec in records:
            r_id = str(rec.get("id", ""))
            if r_id in seen_ids:
                all_issues.append(ValidationIssue(r_id, "id", f"Duplicate record ID found: {r_id}"))
            seen_ids.add(r_id)

            messages = rec.get("messages") or []
            user_text = next(
                (
                    m.get("content", "").strip()
                    for m in messages
                    if isinstance(m, dict) and m.get("role") == "user"
                ),
                "",
            )
            if user_text:
                norm_prompt = re.sub(r"\s+", " ", user_text).lower()
                if norm_prompt in seen_prompts:
                    matched = seen_prompts[norm_prompt]
                    all_issues.append(
                        ValidationIssue(
                            r_id,
                            "prompt",
                            f"Duplicate prompt content found (matches {matched})",
                            severity="WARNING",
                        )
                    )
                else:
                    seen_prompts[norm_prompt] = r_id

            all_issues.extend(self.validate_record(rec))

        critical_errors = [iss for iss in all_issues if iss.severity == "ERROR"]
        is_valid = len(critical_errors) == 0
        return is_valid, all_issues
