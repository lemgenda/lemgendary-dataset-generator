"""Pseudo labels: run image classifiers over the samples a pseudo rule covers.

For every sample whose class rule (:mod:`core.class_labels`) names a
``pseudo_model``, the rule's model (and its ``check_model``) predicts class
probabilities from the image. The results go to a label table outside the
manifold: one JSON Lines file per shard at ``<table root>/<split>/<shard
name>.jsonl``, one line per sample (:class:`PseudoRecord`). A table file is
written to a temporary name and renamed when complete, so it is either whole
or absent; a run skips shards whose table exists (resume). ``repair
class-labels`` reads the table (:func:`load_table`); the shards are not
touched here.

Decoding dominates on a CPU, so images are prepared by a :data:`Decoder`
(:class:`PoolDecoder`: worker processes, the next batch is decoded while the
current one is predicted).
"""

from __future__ import annotations

import json
import os
import tarfile
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from core.class_labels import ClassLabelRules, ClassRule
from tools.repair.image_classifier import ImageClassifier, prepare_images
from utils.shard_audit import members_by_key, split_member_name

__all__ = [
    "TABLE_SUFFIX",
    "Decoder",
    "PoolDecoder",
    "PseudoLabeler",
    "PseudoRecord",
    "ShardPseudoResult",
    "load_table",
    "pseudo_keys",
    "serial_decoder",
    "table_path",
]

TABLE_SUFFIX = ".jsonl"
_TMP_SUFFIX = ".part"
_IMAGE_SUFFIXES = (".webp", ".jpg", ".jpeg", ".png")
_DIGITS = 6

Prepared = list[list[np.ndarray]]
# A decoder starts preparing ``payloads`` at ``sizes`` and returns a callable
# that waits for the result (per size, one array per payload).
Decoder = Callable[[list[bytes], tuple[int, ...]], Callable[[], Prepared]]


def serial_decoder(payloads: list[bytes], sizes: tuple[int, ...]) -> Callable[[], Prepared]:
    """Prepare the images in this process."""
    prepared = prepare_images(payloads, sizes)
    return lambda: prepared


def _chunks(items: list[bytes], count: int) -> list[list[bytes]]:
    step = max(1, -(-len(items) // max(1, count)))
    return [items[i:i + step] for i in range(0, len(items), step)]


class PoolDecoder:
    """Prepares images in worker processes; use as a context manager."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, workers)
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> Self:
        self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __call__(self, payloads: list[bytes], sizes: tuple[int, ...]) -> Callable[[], Prepared]:
        """Submit the chunks now; the returned callable joins them in order."""
        if self._pool is None:
            raise RuntimeError("PoolDecoder is used outside its with block")
        futures: list[Future[Prepared]] = [self._pool.submit(prepare_images, chunk, sizes)
                                           for chunk in _chunks(payloads, self.workers)]

        def _join() -> Prepared:
            joined: Prepared = [[] for _ in sizes]
            for future in futures:
                for i, arrays in enumerate(future.result()):
                    joined[i].extend(arrays)
            return joined
        return _join


@dataclass(frozen=True)
class PseudoRecord:
    """One sample's predicted probabilities (scheme order) by the rule's models."""

    key: str
    model: str
    probs: tuple[float, ...]
    check_model: str | None = None
    check_probs: tuple[float, ...] | None = None

    def to_json(self) -> str:
        """One JSON Lines row."""
        row: dict[str, Any] = {"key": self.key, "model": self.model,
                               "probs": [round(p, _DIGITS) for p in self.probs]}
        if self.check_model and self.check_probs is not None:
            row["check_model"] = self.check_model
            row["check_probs"] = [round(p, _DIGITS) for p in self.check_probs]
        return json.dumps(row, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> PseudoRecord:
        """Parse one row; ``ValueError`` / ``KeyError`` when malformed."""
        row = json.loads(line)
        check = row.get("check_probs")
        return cls(key=str(row["key"]), model=str(row["model"]),
                   probs=tuple(float(p) for p in row["probs"]),
                   check_model=row.get("check_model"),
                   check_probs=tuple(float(p) for p in check) if check is not None else None)


def table_path(table_root: Path, shards_dir: Path, shard: Path) -> Path:
    """Where the table of ``shard`` lives under ``table_root``."""
    return (table_root / shard.relative_to(shards_dir)).with_suffix(TABLE_SUFFIX)


def load_table(table_root: Path) -> dict[str, PseudoRecord]:
    """Every record under ``table_root`` by sample key (empty when there is none).

    Raises ``ValueError`` when a key repeats or a row is malformed.
    """
    records: dict[str, PseudoRecord] = {}
    if not table_root.is_dir():
        return records
    for path in sorted(table_root.rglob(f"*{TABLE_SUFFIX}")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = PseudoRecord.from_json(line)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"{path.name}:{number}: malformed pseudo label") from exc
            if record.key in records:
                raise ValueError(f"{path.name}:{number}: {record.key} is labelled twice")
            records[record.key] = record
    return records


def _pseudo_rule(rules: ClassLabelRules, key: str) -> ClassRule | None:
    rule = rules.rule_for(key)
    return rule if rule is not None and rule.pseudo_model else None


def pseudo_keys(shard: Path, rules: ClassLabelRules) -> int:
    """How many samples of ``shard`` a pseudo rule covers (tar headers only)."""
    with tarfile.open(shard, "r:") as tf:
        return sum(1 for key, _ in members_by_key(tf) if _pseudo_rule(rules, key) is not None)


def _targets(shard: Path, rules: ClassLabelRules) -> Iterator[tuple[str, ClassRule, bytes]]:
    """The pseudo-rule samples of ``shard`` with their image payloads."""
    with tarfile.open(shard, "r:") as tf:
        for key, members in members_by_key(tf):
            rule = _pseudo_rule(rules, key)
            if rule is None:
                continue
            image = next((m for m in members
                          if split_member_name(m.name)[1] in _IMAGE_SUFFIXES), None)
            handle = tf.extractfile(image) if image is not None else None
            if handle is None:
                raise ValueError(f"{shard.name}: {key} has no image member")
            yield key, rule, handle.read()


@dataclass(frozen=True)
class ShardPseudoResult:
    """Samples pseudo-labelled in one shard and the time it took."""

    shard: Path
    samples: int
    seconds: float


Batch = list[tuple[str, ClassRule, bytes]]


@dataclass
class PseudoLabeler:
    """Predicts the pseudo-rule samples of a shard with the rules' models."""

    rules: ClassLabelRules
    classifiers: Mapping[str, ImageClassifier]
    decode: Decoder
    batch_size: int = 64

    def _sizes(self) -> tuple[int, ...]:
        return tuple(sorted({self.classifiers[m].input_size for m in self.rules.models}))

    def _batches(self, shard: Path) -> Iterator[Batch]:
        batch: Batch = []
        for target in _targets(shard, self.rules):
            batch.append(target)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _records(self, batch: Batch, prepared: Prepared) -> list[PseudoRecord]:
        sizes = self._sizes()
        probs = {m: self.classifiers[m].predict(np.stack(prepared[sizes.index(
            self.classifiers[m].input_size)])) for m in self.rules.models}
        records: list[PseudoRecord] = []
        for row, (key, rule, _) in enumerate(batch):
            model = str(rule.pseudo_model)
            check = rule.check_model
            records.append(PseudoRecord(
                key=key, model=model, probs=tuple(float(p) for p in probs[model][row]),
                check_model=check,
                check_probs=tuple(float(p) for p in probs[check][row]) if check else None))
        return records

    def label_shard(self, shard: Path, out: Path) -> ShardPseudoResult:
        """Predict the shard's pseudo-rule samples and write their table to ``out``."""
        started = time.perf_counter()
        sizes = self._sizes()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + _TMP_SUFFIX)
        samples = 0
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                pending: tuple[Batch, Callable[[], Prepared]] | None = None
                for batch in self._batches(shard):
                    started_batch = (batch, self.decode([p for _, _, p in batch], sizes))
                    if pending is not None:
                        samples += self._write(f, pending)
                    pending = started_batch
                if pending is not None:
                    samples += self._write(f, pending)
        except (OSError, ValueError, tarfile.TarError):
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, out)
        return ShardPseudoResult(shard, samples, time.perf_counter() - started)

    def _write(self, f: Any, pending: tuple[Batch, Callable[[], Prepared]]) -> int:
        batch, result = pending
        for record in self._records(batch, result()):
            f.write(record.to_json() + "\n")
        return len(batch)
