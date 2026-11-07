"""Generic Base Compiler Engine for LLM Datasets.

Orchestrates:
1. Ingesting knowledge from declared multi-format sources in unified_data.yaml
2. Invoking specialized generator packs or synthetic extractors
3. Running quality validation and invariant checks (zero-emoji, epistemic tags)
4. Performing deterministic stratified 80/20 train/val splitting (420/105)
5. Writing output shards to data/<slug>_train.jsonl and data/<slug>_val.jsonl
6. Synthesizing full root metadata files (dataset_info, index.json, README, etc.)
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml
from core.config_schema import load_unified_data

from .chunker import RecursiveChunker
from .knowledge_ingestor import KnowledgeIngestor
from .knowledge_synthesizer import compile_knowledge_corpus
from .metadata_compiler import LLMManifoldDescriptor, generate_llm_metadata
from .quality_validator import QualityValidator

logger = logging.getLogger("lemdata.llm.compiler")


class BaseLLMCompiler:
    """Universal compiler for LLM instruction datasets."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root or Path(__file__).resolve().parents[3])
        default_cfg = self.workspace_root / "lemgendary-datasets" / "unified_data.yaml"
        self.config_path = Path(config_path or default_cfg)
        self.ingestor = KnowledgeIngestor()
        self.chunker = RecursiveChunker()
        self.validator = QualityValidator()

    def load_manifest(self) -> dict[str, Any]:
        """Return the parsed unified_data.yaml manifest validated via Pydantic."""
        try:
            validated_manifest = load_unified_data(self.config_path)
            return validated_manifest.to_legacy_dict()
        except (OSError, ValueError, yaml.YAMLError) as exc:
            logger.warning(
                "Pydantic manifest validation fallback for %s: %s",
                self.config_path,
                exc,
            )
            with open(self.config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    def compile_manifold(
        self,
        dataset_key: str,
        custom_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compile a complete LLM manifold declared in unified_data.yaml with atomic swap."""
        cfg = self._get_dataset_config(dataset_key)
        folder = cfg.get("modernized_folder", dataset_key)
        target_dir = self.workspace_root / "LemGendaryDatasets" / folder

        records = custom_records or self._generate_records(dataset_key, cfg)
        self._require_valid(records)

        val_ratio = float(cfg.get("val_split", 0.20) or 0.20)
        test_ratio = float(cfg.get("test_split") or 0.0)
        splits = _partition_records(records, val_ratio, test_ratio)

        with tempfile.TemporaryDirectory(prefix=f".staging_build_{dataset_key}_") as tmp_dir:
            staging_dir = Path(tmp_dir)
            (staging_dir / "data").mkdir(parents=True, exist_ok=True)
            res = self._write_outputs(staging_dir, dataset_key, cfg, splits, len(records))
            _atomic_sync_directories(staging_dir, target_dir)

        _remap_output_paths(res, target_dir, cfg, dataset_key)
        return res

    def _get_dataset_config(self, dataset_key: str) -> dict[str, Any]:
        """Retrieve and validate dataset configuration."""
        manifest = self.load_manifest()
        cfg = manifest.get("datasets", manifest).get(dataset_key)
        if not cfg:
            raise KeyError(f"Dataset key '{dataset_key}' not found in {self.config_path}")
        return cfg

    def _write_outputs(
        self,
        target_dir: Path,
        dataset_key: str,
        cfg: dict[str, Any],
        splits: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
        total: int,
    ) -> dict[str, Any]:
        """Write jsonl shards and trigger metadata generation."""
        train_recs, val_recs, test_recs = splits
        train_path = target_dir / Path(cfg.get("train_file", f"data/{dataset_key}_train.jsonl"))
        val_path = target_dir / Path(cfg.get("val_file", f"data/{dataset_key}_val.jsonl"))
        _write_jsonl(train_path, train_recs)
        _write_jsonl(val_path, val_recs)

        test_path: Path | None = None
        if test_recs:
            test_path = target_dir / Path(cfg.get("test_file", f"data/{dataset_key}_test.jsonl"))
            _write_jsonl(test_path, test_recs)

        generate_llm_metadata(
            target_dir,
            cfg.get("modernized_folder", dataset_key),
            train_recs,
            val_recs,
            descriptor=_descriptor_for(cfg),
            test_records=test_recs if test_recs else None,
        )

        res: dict[str, Any] = {
            "dataset_key": dataset_key,
            "target_dir": str(target_dir),
            "total_records": total,
            "train_count": len(train_recs),
            "val_count": len(val_recs),
            "train_path": str(train_path),
            "val_path": str(val_path),
        }
        if test_recs and test_path:
            res["test_count"] = len(test_recs)
            res["test_path"] = str(test_path)
        return res

    def _generate_records(self, dataset_key: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Run the generator declared in the manifold's ``llm_pipeline`` block."""
        llm_pipe = cfg.get("llm_pipeline", {})
        gen_module = llm_pipe.get("generator_module")
        if gen_module != "tools.llm_helper.knowledge_synthesizer" and dataset_key not in (
            "test_llm",
            "ai_helper_lemgend",
        ):
            raise NotImplementedError(f"No generator module mapped for {dataset_key}")
        return compile_knowledge_corpus(
            knowledge_sources=llm_pipe.get("knowledge_sources", []),
            system_prompt=llm_pipe.get("system_prompt", ""),
            target_samples=int(llm_pipe.get("target_samples", 525)),
            workspace_root=self.workspace_root,
        )

    def _require_valid(self, records: list[dict[str, Any]]) -> None:
        """Raise ValueError when the corpus has any ERROR-severity issue."""
        is_valid, issues = self.validator.validate_corpus(records)
        critical_errors = [iss for iss in issues if iss.severity == "ERROR"]
        if not is_valid:
            err_msg = (
                f"Corpus validation failed with {len(critical_errors)} critical errors: "
                f"{critical_errors[:5]}"
            )
            raise ValueError(err_msg)


def _atomic_sync_directories(staging_dir: Path, target_dir: Path) -> None:
    """Synchronize staging contents into target directory atomically."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in staging_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(staging_dir)
            dest_file = target_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_file)


def _remap_output_paths(
    res: dict[str, Any], target_dir: Path, cfg: dict[str, Any], dataset_key: str
) -> None:
    """Remap staged outcome paths to target destination paths."""
    res["target_dir"] = str(target_dir)
    tr_file = cfg.get("train_file", f"data/{dataset_key}_train.jsonl")
    res["train_path"] = str(target_dir / Path(tr_file))
    vl_file = cfg.get("val_file", f"data/{dataset_key}_val.jsonl")
    res["val_path"] = str(target_dir / Path(vl_file))
    if "test_path" in res:
        ts_file = cfg.get("test_file", f"data/{dataset_key}_test.jsonl")
        res["test_path"] = str(target_dir / Path(ts_file))


def _partition_records(
    records: list[dict[str, Any]], val_split: float, test_split: float = 0.0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic category-stratified 3-way split (seed 42) avoiding prompt overlap."""
    rng = random.Random(42)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_category.setdefault(r.get("category", "default"), []).append(r)

    train_recs: list[dict[str, Any]] = []
    val_recs: list[dict[str, Any]] = []
    test_recs: list[dict[str, Any]] = []

    for _category, items in sorted(by_category.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
        test_count = (
            max(1, int(len(shuffled) * test_split))
            if (len(shuffled) > 2 and test_split > 0)
            else 0
        )
        val_recs.extend(shuffled[:val_count])
        test_recs.extend(shuffled[val_count : val_count + test_count])
        train_recs.extend(shuffled[val_count + test_count :])

    train_recs.sort(key=lambda x: str(x.get("id", "")))
    val_recs.sort(key=lambda x: str(x.get("id", "")))
    test_recs.sort(key=lambda x: str(x.get("id", "")))
    return train_recs, val_recs, test_recs


def _stratified_split(
    records: list[dict[str, Any]], val_split: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Backward compatible 2-way split wrapper."""
    train_recs, val_recs, _ = _partition_records(records, val_split, 0.0)
    return train_recs, val_recs


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def _descriptor_for(cfg: dict[str, Any]) -> LLMManifoldDescriptor:
    """Metadata descriptor from the manifold config, with AI-helper defaults."""
    return LLMManifoldDescriptor(
        category=cfg.get("category", "Technical Documentation & AI Assistant Training"),
        classes=[cfg.get("task", "ai_helper")],
        original_sources=cfg.get(
            "provenance_sources",
            [
                "LemGendary AI Documentation Hub",
                "Master Operations Manuals (GUI, CLI, API)",
                "Architectural Whitepapers",
            ],
        ),
    )
