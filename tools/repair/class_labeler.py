"""Class labels from the sample key and a pseudo-label table (``repair class-labels``).

Each sample's class rule (:mod:`core.class_labels`, the first rule matching
its key) gives either a fixed class or a pseudo label. The JSON label gets
``class_id``, ``class_name`` and ``label_source``; a pseudo label also
``label_confidence`` (the model's top probability) and, with a check model,
``label_check_class`` (the check model's top class). Other JSON keys are kept.

Pseudo labels come from the table ``repair pseudo-labels`` wrote
(:func:`tools.repair.pseudo_labels.load_table`). A sample whose top
probability is below the rule's ``min_confidence`` is dropped (``DROPPED``):
an unconfident guess is worse than no sample. A sample without a rule, or a
pseudo-rule sample without a table row from the rule's model, is a problem
and the apply refuses. The plan tallies samples per class (``class <name>``),
dropped samples per predicted class, and agreement with the check model.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.class_labels import ClassLabelRules, ClassRule
from tools.repair.pseudo_labels import PseudoRecord, load_table
from tools.repair.source_label_fix import (
    DROPPED,
    LABELLED,
    UNCHANGED,
    SampleInput,
    SampleUpdate,
)

__all__ = ["NO_PSEUDO_LABEL", "NO_RULE", "ClassLabeler"]

NO_RULE = "no class rule"
NO_PSEUDO_LABEL = "no pseudo label"
_PSEUDO_FIELDS = ("label_confidence", "label_check_class")
_DIGITS = 4


@functools.cache
def _table(table_root: Path | None) -> dict[str, PseudoRecord]:
    """The pseudo-label table, read once per process."""
    return load_table(table_root) if table_root is not None else {}


def _top(probs: tuple[float, ...]) -> tuple[int, float]:
    best = max(range(len(probs)), key=lambda i: probs[i])
    return best, probs[best]


@dataclass(frozen=True)
class ClassLabeler:
    """Writes each sample's class from its key rule or its pseudo label."""

    rules: ClassLabelRules
    table_root: Path | None = None

    @property
    def file_suffix(self) -> str:
        """No per-file labels: the rules read the key."""
        return ""

    def info_fields(self) -> dict[str, Any]:
        """dataset_info.yaml ``classes``: the scheme's class names in id order."""
        return {"classes": list(self.rules.scheme.class_names)}

    def _fields(self, class_name: str, label_source: str) -> dict[str, Any]:
        return {"class_id": self.rules.scheme.class_id(class_name), "class_name": class_name,
                "label_source": label_source}

    def _update(self, sample: SampleInput, fields: dict[str, Any],
                tally: tuple[str, ...], metrics: dict[str, float]) -> SampleUpdate:
        meta = dict(sample.meta or {})
        stale = [k for k in _PSEUDO_FIELDS if k in meta and k not in fields]
        if not stale and all(meta.get(k) == v for k, v in fields.items()):
            return SampleUpdate(UNCHANGED, tally=tally)
        for key in stale:
            del meta[key]
        meta.update(fields)
        return SampleUpdate(LABELLED, meta=meta, tally=tally, metrics=metrics)

    def _pseudo(self, sample: SampleInput, rule: ClassRule) -> SampleUpdate:
        record = _table(self.table_root).get(sample.key)
        if record is None or record.model != rule.pseudo_model:
            found = f"model {record.model}" if record is not None else "no table row"
            return SampleUpdate(NO_PSEUDO_LABEL, problem=f"{rule.pseudo_model}: {found}")
        names = self.rules.scheme.class_names
        if len(record.probs) != len(names):
            return SampleUpdate(NO_PSEUDO_LABEL,
                                problem=f"{len(record.probs)} probabilities for {len(names)}")
        best, confidence = _top(record.probs)
        tally: tuple[str, ...] = ()
        fields = self._fields(names[best], rule.label_source)
        fields["label_confidence"] = round(confidence, _DIGITS)
        if record.check_probs is not None and len(record.check_probs) == len(names):
            check = _top(record.check_probs)[0]
            fields["label_check_class"] = names[check]
            tally = ("pseudo agree" if check == best else "pseudo disagree",)
        if confidence < rule.min_confidence:
            return SampleUpdate(DROPPED, tally=(*tally, f"dropped {names[best]}"),
                                problem=f"pseudo confidence {confidence:.3f} <"
                                        f" {rule.min_confidence}")
        return self._update(sample, fields, (*tally, f"class {names[best]}"),
                            {"pseudo_confidence": confidence})

    def label(self, sample: SampleInput) -> SampleUpdate:
        """The JSON label of ``sample`` with its class."""
        rule = self.rules.rule_for(sample.key)
        if rule is None:
            return SampleUpdate(NO_RULE, problem=sample.key)
        if rule.pseudo_model:
            return self._pseudo(sample, rule)
        class_name = str(rule.class_name)
        return self._update(sample, self._fields(class_name, rule.label_source),
                            (f"class {class_name}",), {})
