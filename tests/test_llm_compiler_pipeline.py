"""Comprehensive unit tests covering , , and  deliverables."""

import tempfile
import unittest
from pathlib import Path

import yaml
from core.config_schema import UnifiedData
from tools.llm_helper.chunker import KnowledgeChunk, RecursiveChunker
from tools.llm_helper.knowledge_ingestor import KnowledgeIngestor
from tools.llm_helper.metadata_compiler import generate_llm_metadata
from tools.llm_helper.quality_validator import QualityValidator


class TestLLMCompilerPipeline(unittest.TestCase):
    """Verifies all deliverables of , , and ."""

    def test_knowledge_ingestion_and_chunking(self):
        """Ingestion of markdown, json, yaml, and recursive chunking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "test.md").write_text("# Heading 1\n\nContent paragraph.", encoding="utf-8")
            (tmp / "test.json").write_text('{"name": "test", "samples": 100}', encoding="utf-8")
            (tmp / "test.yaml").write_text("dataset:\n  name: Test\n", encoding="utf-8")

            ingestor = KnowledgeIngestor()
            docs = ingestor.ingest_path(tmp)
            self.assertEqual(len(docs), 3)

            chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
            chunks = chunker.chunk_documents(docs)
            self.assertTrue(len(chunks) >= 3)
            for c in chunks:
                self.assertIsInstance(c, KnowledgeChunk)
                self.assertTrue(len(c.text) > 0)

    def test_quality_validator_invariants(self):
        """Quality gate rules (zero-emoji, epistemic tags)."""
        validator = QualityValidator(zero_emojis=True, enforce_epistemic_tags=True)

        valid = {
            "id": "rec_001",
            "category": "troubleshooting_diagnostics",
            "messages": [
                {"role": "system", "content": "You are assistant."},
                {"role": "user", "content": "Help me."},
                {
                    "role": "assistant",
                    "content": (
                        "[CONFIRMED] Stage 1 triage, Stage 2 diagnosis, Stage 3 remediation."
                    ),
                },
            ],
        }
        issues = validator.validate_record(valid)
        self.assertEqual(len(issues), 0)

        # Test emoji detection
        emoji_rec = dict(valid)
        emoji_rec["messages"] = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Emoji here \U0001f600"},
            {"role": "assistant", "content": "Ok"},
        ]
        issues = validator.validate_record(emoji_rec)
        self.assertTrue(any("emoji" in i.issue.lower() for i in issues))

    def test_core_doc_generators_metadata(self):
        """Verifies core doc generators and metadata compiler produce canonical files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            train_recs = [
                {
                    "id": "t1",
                    "category": "api_procedure",
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ],
                }
            ]
            val_recs = [
                {
                    "id": "v1",
                    "category": "api_procedure",
                    "messages": [
                        {"role": "user", "content": "q2"},
                        {"role": "assistant", "content": "a2"},
                    ],
                }
            ]
            generate_llm_metadata(
                manifold_dir=tmp,
                dataset_name="LemGendizedUnitTestLLM",
                train_records=train_recs,
                val_records=val_recs,
                category="Technical Documentation & AI Assistant Training",
                classes=["ai_helper"],
                original_sources=["Source A", "Source B"],
            )

            self.assertTrue((tmp / "dataset_info.yaml").exists())
            self.assertTrue((tmp / "index.json").exists())
            self.assertTrue((tmp / "category.txt").exists())
            self.assertTrue((tmp / "classes.txt").exists())

            with open(tmp / "dataset_info.yaml", encoding="utf-8") as f:
                info = yaml.safe_load(f)
            self.assertEqual(info["count"], 2)
            self.assertEqual(info["train_samples"], 1)
            self.assertEqual(info["val_samples"], 1)
            self.assertEqual(info["canonical_format"], "jsonl")

    def test_pydantic_schema_and_unified_data(self):
        """Schema validation of unified_data.yaml SSOT entries."""
        manifest_path = Path(__file__).resolve().parents[1] / "unified_data.yaml"
        with open(manifest_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        u = UnifiedData.model_validate(data)
        self.assertIn("ai_helper_lemgend", u.datasets)

        ai = u.datasets["ai_helper_lemgend"]
        self.assertEqual(ai.dataset_type, "llm")
        self.assertEqual(ai.split_strategy, "stratified")
        self.assertEqual(ai.val_split, 0.15)
        self.assertEqual(ai.test_split, 0.15)
        self.assertEqual(ai.test_file, "data/ai_helper_test.jsonl")
        pipeline = ai.llm_pipeline
        if pipeline is None:
            self.fail("ai_helper_lemgend declares no llm_pipeline")
        self.assertEqual(pipeline.schema_format, "chatml_turn3")
        self.assertTrue(len(pipeline.knowledge_sources) >= 3)


if __name__ == "__main__":
    unittest.main()
