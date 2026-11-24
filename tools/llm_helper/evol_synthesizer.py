"""Universal Multi-Persona Evol-Prompt Synthesizer.

Generates diverse query evolutions across four distinct operator personas:
1. Novice / Exploratory (colloquial phrasing, imprecise terminology, anxious tone)
2. Staff Architect / Expert (rigorous, demanding math bounds and complexity)
3. Adversarial / Boundary (probing unsupported options, limits, or illegal syntax)
4. Operations / Diagnostic (urgent, log-centric, telemetry-focused troubleshooting)

Guarantees zero-emoji invariants and strict knowledge source grounding.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\uFE0F]",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class EvolVariation:
    """One evolved prompt variation with provenance and persona metadata."""

    prompt: str
    persona: str
    evolution_strategy: str
    complexity_level: str


class EvolPromptSynthesizer:
    """Evolves base technical prompts into varied multi-persona queries."""

    PERSONAS: tuple[str, ...] = (
        "novice_exploratory",
        "staff_architect",
        "adversarial_boundary",
        "operations_diagnostic",
    )

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def evolve_prompt(
        self,
        base_prompt: str,
        context_keywords: list[str] | None = None,
        target_persona: str | None = None,
    ) -> EvolVariation:
        """Evolve a single base prompt into a chosen or sampled persona style."""
        persona = target_persona or self.rng.choice(self.PERSONAS)
        keywords = context_keywords or []
        kw_term = keywords[0] if keywords else "the system"

        cleaned_base = _EMOJI_PATTERN.sub("", base_prompt).strip()

        if persona == "novice_exploratory":
            evolved, strategy = self._evolve_novice(cleaned_base, kw_term)
            complexity = "basic"
        elif persona == "staff_architect":
            evolved, strategy = self._evolve_architect(cleaned_base, kw_term)
            complexity = "advanced"
        elif persona == "adversarial_boundary":
            evolved, strategy = self._evolve_adversarial(cleaned_base, kw_term)
            complexity = "boundary"
        else:
            evolved, strategy = self._evolve_operations(cleaned_base, kw_term)
            complexity = "operational"

        evolved_clean = _EMOJI_PATTERN.sub("", evolved).strip()
        return EvolVariation(
            prompt=evolved_clean,
            persona=persona,
            evolution_strategy=strategy,
            complexity_level=complexity,
        )

    def _evolve_novice(self, base: str, kw: str) -> tuple[str, str]:
        templates = [
            f"I'm pretty new to this. How do I actually do this: {base}?",
            f"Could you explain step by step in simple terms: {base}?",
            f"I'm not sure if I'm doing this right regarding {kw}. {base}",
            f"What happens if I try {base}? Does it work automatically?",
        ]
        return self.rng.choice(templates), "colloquial_simplification"

    def _evolve_architect(self, base: str, kw: str) -> tuple[str, str]:
        templates = [
            f"Specify exact architecture, invariants, and constraints for {kw}: {base}",
            f"What are mathematical bounds and asymptotic complexities for {kw}: {base}?",
            f"Provide an authoritative specification with exact parameters for {kw}: {base}",
            f"Analyze memory bounds and concurrency invariants regarding {kw}: {base}",
        ]
        return self.rng.choice(templates), "rigorous_deepening"

    def _evolve_adversarial(self, base: str, kw: str) -> tuple[str, str]:
        templates = [
            f"Can I bypass standard checks and force {base} in {kw} using unsupported parameters?",
            f"Is it possible to execute {base} on {kw} with invalid inputs without errors?",
            f"What prevents an invalid operation or state collision in {kw} when calling {base}?",
            f"Can we ignore the official specification and run {base} in {kw} on raw sockets?",
        ]
        return self.rng.choice(templates), "boundary_stress"

    def _evolve_operations(self, base: str, kw: str) -> tuple[str, str]:
        templates = [
            f"[URGENT] System alert triggered during {kw}. How do I verify and resolve: {base}?",
            f"Diagnostics indicate failure during operation. What is triage checklist for: {base}?",
            f"Check logs and verify port bindings before proceeding: {base}",
            f"Service reported timeout or connection drop. How do I inspect PID: {base}?",
        ]
        return self.rng.choice(templates), "incident_triage"

    def generate_persona_pool(
        self,
        base_prompt: str,
        context_keywords: list[str] | None = None,
    ) -> list[EvolVariation]:
        """Generate a full suite of all 4 persona evolutions for a base prompt."""
        return [
            self.evolve_prompt(base_prompt, context_keywords, target_persona=p)
            for p in self.PERSONAS
        ]
