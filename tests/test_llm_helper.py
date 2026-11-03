"""Unit tests for tools.llm_helper (generic ingestion, chunking, and validation)."""

import json
from pathlib import Path
import tempfile
import unittest

from tools.llm_helper.chunker import RecursiveChunker
from tools.llm_helper.knowledge_ingestor import KnowledgeDocument, KnowledgeIngestor
from tools.llm_helper.quality_validator import QualityValidator


class TestLLMHelper(unittest.TestCase):
    """Ingestion, chunking and validation in tools.llm_helper."""
    def test_knowledge_ingestion_markdown_and_json(self):
        """KnowledgeIngestor reads Markdown and JSON sources."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            md_file = tmp / "sample.md"
            md_file.write_text("# Title\n\nContent paragraph.", encoding="utf-8")

            json_file = tmp / "data.json"
            json_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")

            ingestor = KnowledgeIngestor()
            docs = ingestor.ingest_path(tmp, glob_pattern="*.*")
            self.assertEqual(len(docs), 2)
            types = {d.source_type for d in docs}
            self.assertIn("markdown", types)
            self.assertIn("json", types)

    def test_recursive_chunker(self):
        """RecursiveChunker splits a long document into several chunks."""
        doc = KnowledgeDocument(
            doc_id="test.md",
            content="# Section 1\n\n" + ("Word " * 200) + "\n\n## Section 2\n\n" + ("Other " * 200),
            source_type="markdown",
            source_path="test.md",
        )
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(len(c.text) > 0)
            self.assertTrue(c.chunk_id.startswith("test.md_c"))

    def test_quality_validator_rules(self):
        """QualityValidator accepts a valid record and flags rule violations."""
        validator = QualityValidator(zero_emojis=True, enforce_epistemic_tags=True)

        valid_record = {
            "id": "valid_rec_1",
            "category": "troubleshooting_diagnostics",
            "messages": [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "Help me."},
                {
                    "role": "assistant",
                    "content": (
                        "[CONFIRMED] Stage 1 triage, Stage 2 diagnosis, Stage 3 remediation."
                    ),
                },
            ],
        }
        issues = validator.validate_record(valid_record)
        self.assertEqual(len(issues), 0)

        # Emoji violation
        invalid_emoji = dict(valid_record)
        invalid_emoji["messages"] = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Test \U0001F600"},
            {"role": "assistant", "content": "Ans."},
        ]
        issues = validator.validate_record(invalid_emoji)
        self.assertTrue(any("emoji" in iss.issue.lower() for iss in issues))


if __name__ == "__main__":
    unittest.main()
