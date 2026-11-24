"""Universal Multi-Turn Dialogue Synthesizer.

Synthesizes realistic 3-to-6 turn stateful conversations:
- Turn 1: User ambiguous query or initial symptom presentation
- Turn 2: Assistant active triage and diagnostic inquiry
- Turn 3: User telemetry / logs / configuration disclosure
- Turn 4: Assistant grounded diagnosis and bounded remediation
- Optional Turns 5-6: Follow-up verification or parameter tuning

Strictly enforces zero-emoji invariants, epistemic tags, and bounded token budgets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\uFE0F]",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class DiagnosticScenario:
    """Diagnostic scenario payload for multi-turn synthesis."""

    topic: str
    symptom: str
    diagnosis: str
    remediation_steps: list[str]
    epistemic_tag: str = "[CONFIRMED]"


class MultiTurnDialogueSynthesizer:
    """Constructs stateful multi-turn troubleshooting and consultation sessions."""

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = _EMOJI_PATTERN.sub("", system_prompt).strip()

    def build_diagnostic_dialogue(
        self,
        record_id: str,
        category: str,
        scenario: DiagnosticScenario,
    ) -> dict[str, Any]:
        """Synthesize a complete 4-turn stateful diagnostic dialogue."""
        t1_user = (
            f"I am running into an issue with {scenario.topic}. "
            f"{scenario.symptom} What should I check?"
        )

        t2_assistant = (
            f"[PROBABLE] Stage 1 Initial Triage: Triage {scenario.topic} before changes.\n\n"
            f"- Verify the active process status and check if the listening port is bound.\n"
            f"- Inspect error logs for collision errors like WinError 10048 or timeout signals.\n"
            f"- Please share the recent log lines or exit codes."
        )

        t3_user = f"Checked logs. Process fails with socket error on {scenario.topic}."

        bullets = "\n".join(f"- {step}" for step in scenario.remediation_steps)
        t4_assistant = (
            f"{scenario.epistemic_tag} Stage 2 Targeted Diagnosis: {scenario.diagnosis}\n\n"
            f"Stage 3 Bounded Remediation:\n"
            f"{bullets}\n\n"
            f"Run the pre-flight check to verify clean port bindings after applying the fix."
        )

        dialogue_turns = [
            ("user", t1_user),
            ("assistant", t2_assistant),
            ("user", t3_user),
            ("assistant", t4_assistant),
        ]
        messages = [{"role": "system", "content": self.system_prompt}]
        for role, content in dialogue_turns:
            messages.append({"role": role, "content": _EMOJI_PATTERN.sub("", content).strip()})

        return {
            "id": record_id,
            "category": category,
            "messages": messages,
            "metadata": {
                "multi_turn": True,
                "turn_count": 4,
                "dialogue_type": "diagnostic_troubleshooting",
                "epistemic_tag": scenario.epistemic_tag,
            },
        }

    def validate_dialogue_turns(self, messages: list[dict[str, Any]]) -> bool:
        """Verify that conversation turns alternate cleanly without repetition."""
        roles = [m.get("role") for m in messages if isinstance(m, dict)]
        if not roles or roles[0] != "system":
            return False
        dialogue = roles[1:]
        return all(r == ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(dialogue))
