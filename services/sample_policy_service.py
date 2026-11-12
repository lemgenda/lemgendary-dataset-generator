"""SamplePolicyService - apply the registry's sample policy to a compiled manifold.

Single Responsibility: resolve the manifold's sample rule from the registry
(:func:`tools.repair.sample_policy_fix.rule_for_manifold`) and plan or apply
it (:mod:`tools.repair.sample_policy_fix`).

- ``plan`` only reads; it is what ``repair sample-policy`` prints in its
  default dry run.
- ``apply`` rewrites the shards (verified, atomic, no copies kept), then
  index.json and dataset_info.yaml, and returns a fresh plan (normally
  nothing left to do).
"""

from __future__ import annotations

from pathlib import Path

from tools.repair.sample_policy_fix import (
    SamplePolicyPlan,
    SampleRule,
    apply_sample_policy,
    plan_sample_policy,
    rule_for_manifold,
)

__all__ = ["SamplePolicyService"]


class SamplePolicyService:
    """Plan and apply the registry's sample policy for one manifold."""

    def __init__(self, rule: SampleRule) -> None:
        self.rule = rule

    @classmethod
    def for_manifold(cls, manifold_dir: Path) -> SamplePolicyService | None:
        """The service for ``manifold_dir``'s registry rule; None when it has no policy."""
        rule = rule_for_manifold(manifold_dir)
        return cls(rule) if rule is not None else None

    def plan(self, manifold_dir: Path) -> SamplePolicyPlan:
        """Read-only: the drops, split moves and metadata updates the policy asks for."""
        return plan_sample_policy(manifold_dir, self.rule)

    def apply(self, plan: SamplePolicyPlan) -> SamplePolicyPlan:
        """Carry ``plan`` out and return a fresh plan of what is left."""
        return apply_sample_policy(plan)
