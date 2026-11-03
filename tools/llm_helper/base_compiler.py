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
from pathlib import Path
import random
from typing import Any
import yaml

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
        self.config_path = Path(
            config_path or (self.workspace_root / "lemgendary-datasets" / "unified_data.yaml")
        )
        self.ingestor = KnowledgeIngestor()
        self.chunker = RecursiveChunker()
        self.validator = QualityValidator()

    def load_manifest(self) -> dict[str, Any]:
        """Return the parsed unified_data.yaml manifest."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def compile_manifold(
        self,
        dataset_key: str,
        custom_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compile a complete LLM manifold declared in unified_data.yaml."""
        manifest = self.load_manifest()
        datasets_cfg = manifest.get("datasets", manifest)
        cfg = datasets_cfg.get(dataset_key)
        if not cfg:
            raise KeyError(f"Dataset key '{dataset_key}' not found in {self.config_path}")

        modernized_folder = cfg.get("modernized_folder", dataset_key)
        target_dir = self.workspace_root / "LemGendaryDatasets" / modernized_folder
        (target_dir / "data").mkdir(parents=True, exist_ok=True)

        # 1. Gather records
        if custom_records is not None:
            records = custom_records
        else:
            records = self._generate_records(dataset_key, cfg)

        # 2. Quality Validation
        self._require_valid(records)

        # 3. Stratified Partitioning
        train_records, val_records = _stratified_split(
            records, float(cfg.get("val_split", 0.20) or 0.20)
        )

        # 4. Resolve output filenames
        train_path = target_dir / Path(cfg.get("train_file", f"data/{dataset_key}_train.jsonl"))
        val_path = target_dir / Path(cfg.get("val_file", f"data/{dataset_key}_val.jsonl"))
        _write_jsonl(train_path, train_records)
        _write_jsonl(val_path, val_records)

        # 5. Metadata compilation
        generate_llm_metadata(
            target_dir, modernized_folder, train_records, val_records, _descriptor_for(cfg)
        )

        return {
            "dataset_key": dataset_key,
            "target_dir": str(target_dir),
            "total_records": len(records),
            "train_count": len(train_records),
            "val_count": len(val_records),
            "train_path": str(train_path),
            "val_path": str(val_path),
        }

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
            raise ValueError(
                f"Corpus validation failed with {len(critical_errors)} critical errors:"
                f" {critical_errors[:5]}"
            )


def _stratified_split(
    records: list[dict[str, Any]], val_split: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic per-category train/val split (seed 42), each side sorted by id."""
    rng = random.Random(42)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_category.setdefault(r.get("category", "default"), []).append(r)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for _category, items in sorted(by_category.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
        val_records.extend(shuffled[:val_count])
        train_records.extend(shuffled[val_count:])

    train_records.sort(key=lambda x: str(x.get("id", "")))
    val_records.sort(key=lambda x: str(x.get("id", "")))
    return train_records, val_records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def _descriptor_for(cfg: dict[str, Any]) -> LLMManifoldDescriptor:
    """Metadata descriptor from the manifold config, with AI-helper defaults."""
    return LLMManifoldDescriptor(
        category=cfg.get("category", "Technical Documentation & AI Assistant Training"),
        classes=[cfg.get("task", "ai_helper")],
        original_sources=cfg.get("provenance_sources", [
            "LemGendary AI Documentation Hub",
            "Master Operations Manuals (GUI, CLI, API)",
            "Architectural Whitepapers",
        ]),
    )
