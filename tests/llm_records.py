"""Shared ChatML record builders for the LLM test modules."""

from __future__ import annotations

from typing import Any

DIAGNOSTIC_CATEGORY = "troubleshooting_diagnostics"
DIAGNOSTIC_REPLY = "[CONFIRMED] Stage 1 triage, Stage 2 diagnosis, Stage 3 remediation."


def chat_record(
    record_id: str,
    turns: tuple[str, str, str] = ("System prompt.", "Help me.", DIAGNOSTIC_REPLY),
    category: str = DIAGNOSTIC_CATEGORY,
) -> dict[str, Any]:
    """Return a 3-turn (system, user, assistant) record that passes every quality rule."""
    roles = ("system", "user", "assistant")
    messages = [{"role": role, "content": text} for role, text in zip(roles, turns, strict=True)]
    return {"id": record_id, "category": category, "messages": messages}
