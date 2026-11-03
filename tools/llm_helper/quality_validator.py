"""Quality and Invariant Validator for LLM Instruction Datasets.

Enforces zero-emoji rules, ChatML conversation schemas, 3-Stage Diagnostic
Intake Protocols with epistemic tagging, and duplicate ID detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
EPISTEMIC_TAG_PATTERN = re.compile(r"\[(CONFIRMED|PROBABLE|ILLUSTRATIVE)\]")


@dataclass(frozen=True)
class ValidationIssue:
    """One invariant violation: the record, the offending field, and what is wrong."""

    record_id: str
    field: str
    issue: str
    severity: str = "ERROR"

    def __repr__(self) -> str:
        return f"[{self.severity}] {self.record_id} ({self.field}): {self.issue}"


class QualityValidator:
    """Enforces strict schema and content invariants over LLM dialogue records."""

    def __init__(
        self,
        zero_emojis: bool = True,
        enforce_epistemic_tags: bool = True,
        enforce_diag_stages: bool = True,
    ) -> None:
        self.zero_emojis = zero_emojis
        self.enforce_epistemic_tags = enforce_epistemic_tags
        self.enforce_diag_stages = enforce_diag_stages

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

        full_text = ""
        assistant_text = ""
        for idx, m in enumerate(messages):
            if not isinstance(m, dict):
                issues.append(
                    ValidationIssue(rec_id, f"messages[{idx}]", "Message must be a dictionary")
                )
                continue
            content = m.get("content", "")
            if not isinstance(content, str) or not content.strip():
                issues.append(
                    ValidationIssue(rec_id, f"messages[{idx}]", "Message content is empty")
                )
            full_text += " " + str(content)
            if m.get("role") == "assistant":
                assistant_text += " " + str(content)

        if self.zero_emojis:
            emojis = EMOJI_PATTERN.findall(full_text)
            if emojis:
                issues.append(
                    ValidationIssue(
                        rec_id, "content", f"Found {len(emojis)} illegal emoji characters: {emojis}"
                    )
                )

        cat = str(record.get("category", ""))
        if cat == "troubleshooting_diagnostics":
            if self.enforce_epistemic_tags:
                matches = EPISTEMIC_TAG_PATTERN.findall(assistant_text)
                if not matches:
                    issues.append(
                        ValidationIssue(
                            rec_id,
                            "assistant",
                            "Missing epistemic tag in troubleshooting response",
                            severity="WARNING",
                        )
                    )

            if self.enforce_diag_stages:
                has_s1 = "stage 1" in assistant_text.lower() or "triage" in assistant_text.lower()
                has_s2 = (
                    "stage 2" in assistant_text.lower() or "diagnosis" in assistant_text.lower()
                )
                has_s3 = (
                    "stage 3" in assistant_text.lower() or "remediation" in assistant_text.lower()
                )
                if not (has_s1 and has_s2 and has_s3):
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

        for rec in records:
            r_id = str(rec.get("id", ""))
            if r_id in seen_ids:
                all_issues.append(ValidationIssue(r_id, "id", f"Duplicate record ID found: {r_id}"))
            seen_ids.add(r_id)
            all_issues.extend(self.validate_record(rec))

        critical_errors = [iss for iss in all_issues if iss.severity == "ERROR"]
        is_valid = len(critical_errors) == 0
        return is_valid, all_issues
