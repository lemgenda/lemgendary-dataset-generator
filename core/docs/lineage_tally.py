"""Per-source and per-split sample tally behind the lineage tables.

Every count is either read from the manifold (``index.json`` samples, JSONL
split files, the Forex shard scan) or left as ``None`` ("not recorded"); no
count is estimated. Per-source counts come from the ``source`` tag of each
sample, or, when that tag only names the manifold itself, from the exact
sample key ``<Prefix>_<source>_<idx>``. When neither names the sources, the
registry ``provenance_sources`` are listed without counts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.docs.context_resolvers import read_yaml_count
from core.docs.context_types import SplitCounts
from core.docs.metadata import format_source
from core.docs.scanner import scan_forex_manifold
from utils.naming import sample_key_source

logger = logging.getLogger(__name__)

SourceCounts = dict[str, int | None]
SourceTally = dict[str, SourceCounts]

_LEGACY_SOURCE_TAGS = ("old", "unknown", "none", "legacy")
_FOREX_SOURCE = "MetaTrader 5 Terminal Native Cache (2019-2026)"
_AI_HELPER_SOURCE = "LemGendary AI Documentation Hub"
_NOT_RECORDED = SplitCounts(train=None, val=None)


@dataclass(frozen=True)
class TallyResult:
    """Sources with their counts, the total sample count and the split counts."""

    sources: SourceTally
    total_samples: int
    splits: SplitCounts


def uncounted_sources(names: Iterable[str]) -> SourceTally:
    """Source rows whose per-source counts are not recorded."""
    return {format_source(str(n)): {"train": None, "val": None, "total": None} for n in names}


def _tag_source(item: dict[str, Any]) -> str | None:
    """The recorded ``source`` tag; legacy or empty tags fall back to the sample key."""
    raw_src = str(item.get("source", "") or "")
    if raw_src and raw_src.lower() not in _LEGACY_SOURCE_TAGS:
        return raw_src
    return sample_key_source(str(item.get("name", ""))) or "Unknown"


def _key_source(item: dict[str, Any]) -> str | None:
    """The source named by the sample key alone."""
    return sample_key_source(str(item.get("name", "")))


def _count_by(
    items: list[dict[str, Any]], source_of: Callable[[dict[str, Any]], str | None]
) -> SourceTally | None:
    """Exact train/val/total counts per source; ``None`` when any sample has no source."""
    sources: SourceTally = {}
    for item in items:
        raw_src = source_of(item)
        if raw_src is None:
            return None
        bucket = sources.setdefault(format_source(raw_src), {"train": 0, "val": 0, "total": 0})
        _bump(bucket, "total")
        split = item.get("split", "train")
        if split in ("train", "val"):
            _bump(bucket, split)
    return sources


def _bump(bucket: SourceCounts, key: str) -> None:
    """Add one to an integer count."""
    bucket[key] = (bucket.get(key) or 0) + 1


def _split_counts(items: list[dict[str, Any]]) -> SplitCounts:
    """Real train/val counts over the samples (a missing split counts as train)."""
    splits = [item.get("split", "train") for item in items]
    return SplitCounts(train=splits.count("train"), val=splits.count("val"))


def _names_only_manifold(sources: SourceTally, manifold_name: str) -> bool:
    """True when every source row merely names the manifold itself."""
    return bool(sources) and all(
        "manifold" in k.lower() or k == manifold_name for k in sources
    )


def _tally_items(
    items: list[dict[str, Any]], manifold_name: str, unified_entry: dict[str, Any]
) -> TallyResult:
    """Tally index samples, recovering the sources from the sample keys when needed."""
    splits = _split_counts(items)
    by_tag = _count_by(items, _tag_source) or {}
    if not _names_only_manifold(by_tag, manifold_name):
        return TallyResult(by_tag, len(items), splits)
    by_key = _count_by(items, _key_source)
    if by_key and not _names_only_manifold(by_key, manifold_name):
        return TallyResult(by_key, len(items), splits)
    provenance = unified_entry.get("provenance_sources")
    if provenance:
        return TallyResult(uncounted_sources(provenance), len(items), splits)
    return TallyResult(by_tag, len(items), splits)


def _count_lines_or_zero(path: Path) -> int:
    """Count lines of ``path``; 0 when it is missing or unreadable."""
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _tally_forex(output_root: Path) -> TallyResult:
    """Forex samples from the shard scan (else dataset_info.yaml); splits are not recorded."""
    forex_data = scan_forex_manifold(output_root)
    total_samples = sum(forex_data.get("samples_per_year", {}).values())
    if total_samples == 0:
        total_samples = read_yaml_count(output_root)
    sources: SourceTally = {
        _FOREX_SOURCE: {"train": None, "val": None, "total": total_samples},
    }
    return TallyResult(sources, total_samples, _NOT_RECORDED)


def _ai_helper_files(output_root: Path) -> tuple[Path, Path, Path]:
    """The train/val/test JSONL files (``data/`` first, else the manifold root)."""
    base = output_root / "data"
    if not (base / "ai_helper_train.jsonl").exists():
        base = output_root
    return (
        base / "ai_helper_train.jsonl",
        base / "ai_helper_val.jsonl",
        base / "ai_helper_test.jsonl",
    )


def _tally_ai_helper(output_root: Path, unified_entry: dict[str, Any]) -> TallyResult:
    """Instruction pairs of the AI helper JSONL splits; one source keeps the counts."""
    train_file, val_file, test_file = _ai_helper_files(output_root)
    tr_cnt = _count_lines_or_zero(train_file)
    vl_cnt = _count_lines_or_zero(val_file)
    counted = tr_cnt + vl_cnt + _count_lines_or_zero(test_file)
    total_samples = counted or read_yaml_count(output_root)
    splits = SplitCounts(train=tr_cnt, val=vl_cnt) if counted else _NOT_RECORDED
    names = list(unified_entry.get("provenance_sources") or [_AI_HELPER_SOURCE])
    if len(names) == 1 and counted:
        sources: SourceTally = {
            format_source(str(names[0])): {"train": tr_cnt, "val": vl_cnt, "total": counted}
        }
        return TallyResult(sources, total_samples, splits)
    return TallyResult(uncounted_sources(names), total_samples, splits)


def _load_index_items(output_root: Path) -> list[dict[str, Any]]:
    """Samples recorded in an existing index.json; empty when missing or unreadable."""
    index_file = output_root / "index.json"
    if not index_file.exists():
        return []
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Could not parse index.json for sample counts: %s", exc)
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def tally_sources(
    output_root: Path,
    task: str,
    manifold_name: str,
    unified_entry: dict[str, Any],
    final_index: list[dict[str, Any]] | None,
) -> TallyResult:
    """Return the per-source tally, total sample count and split counts of the manifold."""
    if final_index:
        return _tally_items(final_index, manifold_name, unified_entry)
    if task == "forex":
        return _tally_forex(output_root)
    if task == "ai_helper":
        return _tally_ai_helper(output_root, unified_entry)
    items = _load_index_items(output_root)
    if items:
        return _tally_items(items, manifold_name, unified_entry)
    names = unified_entry.get("provenance_sources") or []
    return TallyResult(uncounted_sources(names), read_yaml_count(output_root), _NOT_RECORDED)
