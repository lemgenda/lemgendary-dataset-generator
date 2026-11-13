"""Validate the labels inside WebDataset shards: JSON only, every document well formed.

The shard verifier's label pass (header checks live in
:mod:`utils.shard_audit`): a ``.txt`` member is a legacy text label (``repair
txt-labels`` converts it), and every ``<key>.json`` member must be a JSON
object that passes :func:`utils.label_json.label_problems` for the
manifold's label strategy.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from utils.label_json import label_problems
from utils.shard_audit import find_shards, split_member_name

__all__ = [
    "TEXT_LABEL_SUFFIX",
    "LabelAudit",
    "audit_labels",
    "label_audit_problems",
    "verify_shard_labels",
]

TEXT_LABEL_SUFFIX = ".txt"
_JSON_SUFFIX = ".json"
_SHOWN = 3


@dataclass
class LabelAudit:
    """Label members of a shard set: JSON documents checked, text labels, invalid ones."""

    json_labels: int = 0
    text_labels: list[str] = field(default_factory=list)
    invalid: dict[str, list[str]] = field(default_factory=dict)


def _check_json(audit: LabelAudit, name: str, payload: bytes, strategy: str | None) -> None:
    audit.json_labels += 1
    try:
        doc = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        audit.invalid[name] = [f"not JSON ({exc})"]
        return
    problems = label_problems(doc, strategy)
    if problems:
        audit.invalid[name] = problems


def _audit_shard(audit: LabelAudit, shard: Path, strategy: str | None) -> None:
    with tarfile.open(shard, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            suffix = split_member_name(member.name)[1]
            if suffix == TEXT_LABEL_SUFFIX:
                audit.text_labels.append(member.name)
            elif suffix == _JSON_SUFFIX:
                handle = tf.extractfile(member)
                _check_json(audit, member.name, handle.read() if handle else b"", strategy)


def audit_labels(shards_dir: Path, strategy: str | None = None) -> LabelAudit:
    """Read every label member of every shard under ``shards_dir``.

    Raises ``OSError`` / ``tarfile.TarError`` for an unreadable shard (the
    header audit reports those).
    """
    audit = LabelAudit()
    for shard in find_shards(shards_dir):
        _audit_shard(audit, shard, strategy)
    return audit


def label_audit_problems(audit: LabelAudit) -> list[str]:
    """Human-readable label defects of ``audit`` (empty: the labels are sound)."""
    problems = []
    if audit.text_labels:
        problems.append(
            f"{len(audit.text_labels)} text label members (.txt), e.g."
            f" {', '.join(audit.text_labels[:_SHOWN])}; JSON is the only label format"
            " (run repair txt-labels)"
        )
    if audit.invalid:
        examples = [f"{name}: {'; '.join(issues[:2])}"
                    for name, issues in list(audit.invalid.items())[:_SHOWN]]
        problems.append(f"{len(audit.invalid)} invalid JSON labels ({' | '.join(examples)})")
    return problems


def verify_shard_labels(shards_dir: Path, tag: str, strategy: str | None = None) -> bool:
    """Audit every label member under ``shards_dir``; print the defects, False when any."""
    try:
        problems = label_audit_problems(audit_labels(shards_dir, strategy))
    except (OSError, tarfile.TarError) as exc:
        problems = [f"unreadable shard ({exc})"]
    for problem in problems:
        print(f"[{tag}] {problem}")
    return not problems
