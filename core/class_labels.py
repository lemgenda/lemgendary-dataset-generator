"""Image class label schemes, key rules and pseudo-label models (SSOT).

A classification manifold whose shards carry no class labels gets them from
its registry entry's ``class_labels``: a class ``scheme`` and ordered
``rules``. Each rule matches the samples of one source (the ``<source>`` of a
``<Prefix>_<source>_<idx>`` key), optionally only those with ``idx < below``,
and gives them either a fixed class (``class_name``) or a pseudo label from an
image classifier (``pseudo_model``, kept when its top probability reaches
``min_confidence``; ``check_model`` is a second classifier whose agreement is
reported). The first matching rule wins. ``repair pseudo-labels`` runs the
models, ``repair class-labels`` writes ``class_id``, ``class_name`` and
``label_source`` (plus ``label_confidence`` and ``label_check_class`` for a
pseudo label) into each sample's JSON.

``nsfw_5``: the five classes of the GantMan nsfw_model, in its output order
(alphabetical): 0 drawings, 1 hentai, 2 neutral, 3 porn, 4 sexy.

Pseudo-label models (:data:`PSEUDO_MODELS`) are ONNX conversions of the
GantMan nsfw_model (MIT licence) published in the Hugging Face repository
``deepghs/imgutils-models`` (MIT), pinned to one revision: Inception V3
(299 px) and the MobileNet V2 model that nsfwjs ships (224 px). Both take
NHWC RGB pixels scaled to 0..1 after a nearest-neighbour resize (the Keras
``load_img`` default the models were trained with) and output softmax
probabilities in the scheme's order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from utils.naming import sample_key_source

__all__ = [
    "CLASS_SCHEMES",
    "NSFW_5",
    "PSEUDO_MODELS",
    "ClassLabelRules",
    "ClassRule",
    "ClassScheme",
    "PseudoModel",
    "pseudo_label_source",
]

NSFW_5: tuple[str, ...] = ("drawings", "hentai", "neutral", "porn", "sexy")
_PSEUDO_PREFIX = "pseudo:"
_DEFAULT_MIN_CONFIDENCE = 0.8


@dataclass(frozen=True)
class ClassScheme:
    """Class names in id order (id = position)."""

    name: str
    class_names: tuple[str, ...]

    def class_id(self, class_name: str) -> int:
        """The id of ``class_name``; ``ValueError`` when the scheme has no such class."""
        if class_name not in self.class_names:
            raise ValueError(f"class '{class_name}' is not in scheme {self.name}"
                             f" {list(self.class_names)}")
        return self.class_names.index(class_name)


@dataclass(frozen=True)
class PseudoModel:
    """An image classifier on the Hugging Face hub, pinned to one revision."""

    name: str
    repo_id: str
    filename: str
    revision: str
    input_size: int
    class_names: tuple[str, ...]


CLASS_SCHEMES: dict[str, ClassScheme] = {
    scheme.name: scheme for scheme in (ClassScheme("nsfw_5", NSFW_5),)
}
_IMGUTILS_REPO = "deepghs/imgutils-models"
_IMGUTILS_REVISION = "35e0ed8f35ce0dbea674568161a48b31ba7d2a99"
PSEUDO_MODELS: dict[str, PseudoModel] = {
    model.name: model for model in (
        PseudoModel("gantman_inception_v3", _IMGUTILS_REPO, "nsfw/inception_v3.onnx",
                    _IMGUTILS_REVISION, 299, NSFW_5),
        PseudoModel("gantman_mobilenet_v2", _IMGUTILS_REPO, "nsfw/nsfwjs.onnx",
                    _IMGUTILS_REVISION, 224, NSFW_5),
    )
}


def pseudo_label_source(model: str) -> str:
    """The ``label_source`` tag of a pseudo label from ``model``."""
    return f"{_PSEUDO_PREFIX}{model}"


@dataclass(frozen=True)
class ClassRule:
    """The class of one source's samples (optionally only ``idx < below``)."""

    source: str
    label_source: str
    class_name: str | None = None
    pseudo_model: str | None = None
    check_model: str | None = None
    below: int | None = None
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE

    def matches(self, source: str, idx: int) -> bool:
        """True when a sample of ``source`` with index ``idx`` falls under this rule."""
        return source.lower() == self.source.lower() and (self.below is None
                                                          or idx < self.below)

    @property
    def models(self) -> tuple[str, ...]:
        """The pseudo-label model and the check model, if any."""
        return tuple(m for m in (self.pseudo_model, self.check_model) if m)


def _model(name: Any, scheme: ClassScheme) -> str:
    model = PSEUDO_MODELS.get(str(name))
    if model is None:
        raise ValueError(f"pseudo model '{name}' is not one of {sorted(PSEUDO_MODELS)}")
    if model.class_names != scheme.class_names:
        raise ValueError(f"pseudo model {model.name} classes {list(model.class_names)} differ"
                         f" from scheme {scheme.name}")
    return model.name


def _rule(raw: Mapping[str, Any], scheme: ClassScheme) -> ClassRule:
    class_name, pseudo = raw.get("class_name"), raw.get("pseudo_model")
    if (class_name is None) == (pseudo is None):
        raise ValueError(f"class rule for {raw.get('source')}: give exactly one of"
                         " class_name and pseudo_model")
    below = raw.get("below")
    min_confidence = float(raw.get("min_confidence", _DEFAULT_MIN_CONFIDENCE))
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence {min_confidence} is not within 0..1")
    if class_name is not None:
        scheme.class_id(str(class_name))
        return ClassRule(source=str(raw["source"]), class_name=str(class_name),
                         label_source=str(raw.get("label_source") or raw["source"]),
                         below=int(below) if below is not None else None)
    model = _model(pseudo, scheme)
    check = raw.get("check_model")
    return ClassRule(source=str(raw["source"]), pseudo_model=model,
                     check_model=_model(check, scheme) if check else None,
                     label_source=str(raw.get("label_source") or pseudo_label_source(model)),
                     below=int(below) if below is not None else None,
                     min_confidence=min_confidence)


@dataclass(frozen=True)
class ClassLabelRules:
    """A class scheme and its ordered rules (the first matching rule wins)."""

    scheme: ClassScheme
    rules: tuple[ClassRule, ...]

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> ClassLabelRules:
        """Rules from a registry ``class_labels`` mapping; ``ValueError`` when invalid."""
        scheme = CLASS_SCHEMES.get(str(spec.get("scheme", "")))
        if scheme is None:
            raise ValueError(f"class_labels scheme '{spec.get('scheme')}' is not one of"
                             f" {sorted(CLASS_SCHEMES)}")
        raw_rules: Sequence[Mapping[str, Any]] = spec.get("rules") or ()
        if not raw_rules:
            raise ValueError("class_labels has no rules")
        return cls(scheme, tuple(_rule(raw, scheme) for raw in raw_rules))

    def rule_for(self, key: str) -> ClassRule | None:
        """The first rule matching sample ``key``; None when no rule does."""
        base = key.rpartition("/")[2]
        source = sample_key_source(base)
        idx_text = base.rpartition("_")[2]
        if source is None or not idx_text.isdigit():
            return None
        idx = int(idx_text)
        return next((rule for rule in self.rules if rule.matches(source, idx)), None)

    @property
    def models(self) -> tuple[str, ...]:
        """Every model the rules use (pseudo and check), each once, in rule order."""
        return tuple(dict.fromkeys(m for rule in self.rules for m in rule.models))
