"""R4-B2: class labels from key rules and pseudo labels (ClassificationMaster).

Tiny synthetic shards cover every rule of the registry entry (anime explicit
and general by index, food101, NSFW pseudo labels); the pseudo-label path
runs with fake classifiers (no model download).
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from pydantic import ValidationError

from core.class_labels import NSFW_5, ClassLabelRules, pseudo_label_source
from core.config_schema import DatasetEntry
from core.docs.generators.classes_txt import resolve_classes_list
from core.docs.generators.dataset_info_yaml import RECORDED_LABEL_FIELDS
from core.paths import UNIFIED_DATA
from services.pseudo_label_service import PseudoLabelService, plan_pseudo_labels
from services.source_label_service import SourceLabelService
from tools.repair.class_labeler import (
    DISAGREEMENT,
    LOW_CONFIDENCE,
    NO_PSEUDO_LABEL,
    ClassLabeler,
)
from tools.repair.pseudo_labels import (
    PoolDecoder,
    PseudoRecord,
    load_table,
    serial_decoder,
)
from tools.repair.source_label_fix import (
    DROPPED,
    SampleInput,
    SourceLabelError,
    every_key_resolver,
)
from utils.label_audit import audit_labels, label_audit_problems
from utils.sample_group import add_tar_member
from utils.shard_audit import audit_shard, find_shards

_FOLDER = "LemGendizedClassificationMaster"
_PREFIX = "ClassificationMasterManifold"
_EXPLICIT = 278119
_PORN, _SEXY, _NEUTRAL = (NSFW_5.index(n) for n in ("porn", "sexy", "neutral"))
# NSFW image colour -> (primary probabilities, check probabilities)
_RED, _GREEN, _BLUE = (200, 20, 20), (20, 200, 20), (20, 20, 200)


def _registry_rules() -> ClassLabelRules:
    data = yaml.safe_load(UNIFIED_DATA.read_text(encoding="utf-8"))
    return ClassLabelRules.from_spec(data["datasets"]["classification_master_manifold"]
                                     ["class_labels"])


def _key(source: str, idx: int) -> str:
    return f"{_PREFIX}_{source}_{idx:09d}"


def _webp(colour: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 10), colour).save(buf, format="WEBP", lossless=True)
    return buf.getvalue()


def _onehot(index: int, confidence: float) -> np.ndarray:
    probs = np.full(len(NSFW_5), (1.0 - confidence) / (len(NSFW_5) - 1))
    probs[index] = confidence
    return probs


class FakeClassifier:
    """Class from the dominant colour channel: red porn, green sexy, blue neutral."""

    def __init__(self, size: int, confidences: dict[int, float], classes: dict[int, int]) -> None:
        self._size = size
        self.confidences = confidences
        self.classes = classes
        self.calls = 0

    @property
    def input_size(self) -> int:
        """Square input side."""
        return self._size

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """One-hot-ish probabilities per image."""
        self.calls += 1
        assert batch.shape[1:] == (self._size, self._size, 3) and batch.dtype == np.uint8
        channels = batch.reshape(batch.shape[0], -1, 3).mean(axis=1).argmax(axis=1)
        return np.stack([_onehot(self.classes[int(c)], self.confidences[int(c)])
                         for c in channels])


def _classifiers() -> dict[str, FakeClassifier]:
    return {
        # red: confident porn; green: unconfident sexy (dropped); blue: confident neutral
        "gantman_inception_v3": FakeClassifier(299, {0: 0.95, 1: 0.55, 2: 0.9},
                                               {0: _PORN, 1: _SEXY, 2: _NEUTRAL}),
        # the check model disagrees on blue (sexy instead of neutral): dropped
        "gantman_mobilenet_v2": FakeClassifier(224, {0: 0.9, 1: 0.9, 2: 0.6},
                                               {0: _PORN, 1: _SEXY, 2: _SEXY}),
    }


_SAMPLES: list[tuple[str, str, int, tuple[int, int, int]]] = [
    ("train", "anime_dbrating", 5, _RED),
    ("train", "anime_dbrating", _EXPLICIT - 1, _RED),
    ("train", "anime_dbrating", _EXPLICIT, _RED),
    ("train", "anime_dbrating", 600000, _RED),
    ("train", "food101", 7, _GREEN),
    ("train", "NSFW", 1, _RED),
    ("train", "NSFW", 2, _GREEN),
    ("val", "NSFW", 3, _BLUE),
    ("val", "food101", 8, _BLUE),
]


def _manifold(root: Path) -> Path:
    manifold = root / _FOLDER
    for split in ("train", "val"):
        shard = manifold / f"shards/{split}/shard-00000.data"
        shard.parent.mkdir(parents=True)
        with tarfile.open(shard, "w") as tf:
            for s, source, idx, colour in _SAMPLES:
                if s != split:
                    continue
                add_tar_member(tf, f"{_key(source, idx)}.webp", _webp(colour))
                meta = {"split": split, "task": "restoration"}
                add_tar_member(tf, f"{_key(source, idx)}.json", json.dumps(meta).encode())
    index = [{"name": _key(source, idx), "split": s} for s, source, idx, _ in _SAMPLES]
    (manifold / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (manifold / "dataset_info.yaml").write_text(f"count: {len(_SAMPLES)}\n", encoding="utf-8")
    return manifold


def _labels(manifold: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for shard in find_shards(manifold / "shards"):
        with tarfile.open(shard) as tf:
            for member in tf.getmembers():
                if member.name.endswith(".json"):
                    handle = tf.extractfile(member)
                    assert handle is not None
                    out[member.name[:-5]] = json.loads(handle.read())
    return out


def _pseudo(manifold: Path, table: Path) -> dict[str, FakeClassifier]:
    rules = _registry_rules()
    fakes = _classifiers()
    service = PseudoLabelService(rules, table, lambda: dict(fakes), serial_decoder,
                                 batch_size=2)
    service.run(plan_pseudo_labels(manifold, rules, table))
    return fakes


class ClassRuleTests(unittest.TestCase):
    """The registry rules and their validation."""

    def test_rule_per_source(self) -> None:
        """Explicit anime below 278,119 is hentai, the rest drawings; food101 neutral; NSFW
        pseudo-labelled with a check model."""
        rules = _registry_rules()
        self.assertEqual(rules.scheme.class_names, NSFW_5)

        def rule(key: str) -> Any:
            found = rules.rule_for(key)
            assert found is not None
            return found
        self.assertEqual(rule(_key("anime_dbrating", _EXPLICIT - 1)).class_name, "hentai")
        self.assertEqual(rule(_key("anime_dbrating", _EXPLICIT)).class_name, "drawings")
        self.assertEqual(rule(_key("food101", 0)).class_name, "neutral")
        nsfw = rule(_key("NSFW", 0))
        self.assertEqual((nsfw.pseudo_model, nsfw.check_model, nsfw.min_confidence,
                          nsfw.require_check_agreement),
                         ("gantman_inception_v3", "gantman_mobilenet_v2", 0.8, True))
        self.assertEqual(nsfw.label_source, pseudo_label_source("gantman_inception_v3"))
        self.assertIsNone(rules.rule_for(_key("unknown", 0)))
        self.assertIsNone(rules.rule_for("no-index"))
        self.assertEqual(rules.models, ("gantman_inception_v3", "gantman_mobilenet_v2"))

    def test_invalid_specs_are_refused(self) -> None:
        """Both or neither of class_name/pseudo_model, unknown classes, models, schemes."""
        bad = [
            {"scheme": "nsfw_5", "rules": [{"source": "a", "class_name": "porn",
                                            "pseudo_model": "gantman_inception_v3"}]},
            {"scheme": "nsfw_5", "rules": [{"source": "a"}]},
            {"scheme": "nsfw_5", "rules": [{"source": "a", "class_name": "cat"}]},
            {"scheme": "nsfw_5", "rules": [{"source": "a", "pseudo_model": "nope"}]},
            {"scheme": "nope", "rules": [{"source": "a", "class_name": "porn"}]},
            {"scheme": "nsfw_5", "rules": []},
            {"scheme": "nsfw_5", "rules": [{"source": "a", "require_check_agreement": True,
                                            "pseudo_model": "gantman_inception_v3"}]},
        ]
        for spec in bad:
            with self.assertRaises(ValueError, msg=str(spec)):
                ClassLabelRules.from_spec(spec)
        with self.assertRaises(ValidationError):
            DatasetEntry.model_validate({"name": "x", "class_labels": bad[2]})

    def test_classes_txt_uses_the_class_scheme(self) -> None:
        """classes.txt lists the five classes in id order."""
        self.assertEqual(resolve_classes_list(_FOLDER, "classification"), list(NSFW_5))
        self.assertIn("classes", RECORDED_LABEL_FIELDS)


class PseudoLabelTests(unittest.TestCase):
    """The pseudo-label table (fake models)."""

    def test_table_holds_both_models_and_resumes(self) -> None:
        """Only NSFW samples are predicted; a second run finds every table written."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            table = Path(tmp) / "table"
            plan = plan_pseudo_labels(manifold, _registry_rules(), table)
            self.assertEqual([s.samples for s in plan.shards], [2, 1])
            fakes = _pseudo(manifold, table)
            records = load_table(table)
            self.assertEqual(sorted(records), sorted(_key("NSFW", i) for i in (1, 2, 3)))
            red = records[_key("NSFW", 1)]
            self.assertEqual((red.model, red.check_model), ("gantman_inception_v3",
                                                            "gantman_mobilenet_v2"))
            self.assertEqual(int(np.argmax(red.probs)), _PORN)
            again = plan_pseudo_labels(manifold, _registry_rules(), table)
            self.assertEqual(again.remaining, [])
            calls = fakes["gantman_inception_v3"].calls
            self.assertEqual(PseudoLabelService(_registry_rules(), table, lambda: dict(fakes),
                                                serial_decoder).run(again), [])
            self.assertEqual(fakes["gantman_inception_v3"].calls, calls)
            self.assertEqual(audit_shard(find_shards(manifold / "shards")[0]).samples, 7)

    def test_duplicate_key_is_refused(self) -> None:
        """A key in two table files is an error."""
        with tempfile.TemporaryDirectory() as tmp:
            row = PseudoRecord("k_a_1", "m", (1.0, 0.0)).to_json() + "\n"
            for name in ("a.jsonl", "b.jsonl"):
                (Path(tmp) / name).write_text(row, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_table(Path(tmp))

    def test_pool_decoder_matches_serial(self) -> None:
        """Worker processes prepare the same pixels, in order."""
        payloads = [_webp(c) for c in (_RED, _GREEN, _BLUE)]
        with PoolDecoder(2) as decode:
            pooled = decode(payloads, (8, 4))()
        serial = serial_decoder(payloads, (8, 4))()
        for got, want in zip(pooled, serial, strict=True):
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(got, want, strict=True)))


class ClassLabelRepairTests(unittest.TestCase):
    """repair class-labels: plan, apply, re-plan."""

    def _service(self, table: Path, rules: ClassLabelRules | None = None) -> SourceLabelService:
        return SourceLabelService(ClassLabeler(rules or _registry_rules(), table),
                                  every_key_resolver, 1, "class_label")

    def test_plan_tallies_classes_and_agreement(self) -> None:
        """The plan counts samples per class, drops per class and reason, and agreement."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            _pseudo(manifold, Path(tmp) / "table")
            plan = self._service(Path(tmp) / "table").plan(manifold)
            self.assertEqual(plan.problems, 0, plan.report())
            tallies = plan.tallies()
            self.assertEqual([tallies[f"class {n}"] for n in NSFW_5], [2, 2, 2, 1, 0])
            self.assertEqual((tallies["dropped sexy"], tallies["dropped neutral"]), (1, 1))
            self.assertEqual((tallies[LOW_CONFIDENCE], tallies[DISAGREEMENT]), (1, 1))
            self.assertEqual((tallies["pseudo agree"], tallies["pseudo disagree"]), (2, 1))

    def test_agreement_off_keeps_a_confident_disagreeing_label(self) -> None:
        """Without require_check_agreement only the confidence decides."""
        spec = yaml.safe_load(UNIFIED_DATA.read_text(encoding="utf-8"))["datasets"][
            "classification_master_manifold"]["class_labels"]
        for rule in spec["rules"]:
            rule.pop("require_check_agreement", None)
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            _pseudo(manifold, Path(tmp) / "table")
            plan = self._service(Path(tmp) / "table",
                                 ClassLabelRules.from_spec(spec)).plan(manifold)
            tallies = plan.tallies()
            self.assertEqual(tallies["class neutral"], 3)
            self.assertNotIn(DISAGREEMENT, tallies)

    def test_missing_check_probabilities_are_a_problem(self) -> None:
        """Under require_check_agreement a row without check probabilities refuses."""
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "table"
            table.mkdir()
            key = _key("NSFW", 9)
            row = PseudoRecord(key, "gantman_inception_v3", tuple(_onehot(_PORN, 0.99)))
            (table / "t.jsonl").write_text(row.to_json() + "\n", encoding="utf-8")
            sample = SampleInput(key, (12, 10), b"", {}, frozenset({"webp", "json"}), None)
            update = ClassLabeler(_registry_rules(), table).label(sample)
            self.assertEqual(update.status, NO_PSEUDO_LABEL)
            self.assertNotEqual(update.status, DROPPED)

    def test_apply_writes_classes_and_drops_unsure(self) -> None:
        """Rule classes, pseudo classes with confidence and check class; the unconfident
        and the disagreeing NSFW samples leave the shards, index.json and the count."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            _pseudo(manifold, Path(tmp) / "table")
            service = self._service(Path(tmp) / "table")
            remaining = service.apply(service.plan(manifold))
            self.assertFalse(remaining.changed, remaining.report())
            labels = _labels(manifold)
            self.assertNotIn(_key("NSFW", 2), labels)
            self.assertNotIn(_key("NSFW", 3), labels)

            def fields(source: str, idx: int, *names: str) -> tuple[Any, ...]:
                return tuple(labels[_key(source, idx)].get(n) for n in names)
            self.assertEqual(fields("anime_dbrating", _EXPLICIT - 1, "class_id", "class_name",
                                    "label_source", "split"),
                             (1, "hentai", "danbooru-explicit", "train"))
            self.assertEqual(fields("anime_dbrating", _EXPLICIT, "class_id", "label_source"),
                             (0, "danbooru-general"))
            self.assertEqual(fields("food101", 8, "class_id", "label_source",
                                    "label_confidence"), (2, "food101", None))
            self.assertEqual(fields("NSFW", 1, "class_id", "label_confidence",
                                    "label_check_class", "label_source"),
                             (3, 0.95, "porn", "pseudo:gantman_inception_v3"))
            self.assertEqual(label_audit_problems(audit_labels(manifold / "shards",
                                                               "class_label")), [])
            info = yaml.safe_load((manifold / "dataset_info.yaml").read_text(encoding="utf-8"))
            self.assertEqual((info["classes"], info["label_json"], info["count"]),
                             (list(NSFW_5), "class_label", len(_SAMPLES) - 2))
            self.assertEqual((manifold / "classes.txt").read_text(encoding="utf-8").split(),
                             list(NSFW_5))
            index = json.loads((manifold / "index.json").read_text(encoding="utf-8"))
            self.assertFalse({_key("NSFW", 2), _key("NSFW", 3)} & {e["name"] for e in index})

    def test_missing_pseudo_label_refuses_the_apply(self) -> None:
        """Without the table the NSFW samples are problems and nothing is written."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            service = self._service(Path(tmp) / "empty")
            plan = service.plan(manifold)
            self.assertEqual(plan.problems, 3)
            self.assertEqual(plan.statuses()[NO_PSEUDO_LABEL], 3)
            before = _labels(manifold)
            with self.assertRaises(SourceLabelError):
                service.apply(plan)
            self.assertEqual(_labels(manifold), before)

    def test_registry_service_for_the_folder(self) -> None:
        """for_class_labels reads the registry entry by folder name."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            table = Path(tmp) / "table"
            _pseudo(manifold, table)
            service = SourceLabelService.for_class_labels(manifold, table)
            self.assertEqual(service.label_json, "class_label")
            self.assertEqual(service.plan(manifold).problems, 0)


if __name__ == "__main__":
    unittest.main()
