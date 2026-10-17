"""Unit tests for AI Helper dataset tools: corpus generation and provenance enrichment."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.ai_helper.corpus_generator import SYSTEM_PROMPT, generate_gui_examples
from tools.ai_helper.enrich_ai_helper_provenance import (
    compute_content_sha256,
    enrich_file,
)


class TestAIHelperTools(unittest.TestCase):
    """Verifies AI Helper corpus generation and provenance enrichment utilities."""

    def test_compute_content_sha256_determinism(self) -> None:
        """Verify content SHA256 is deterministic and valid 64-character hex string."""
        messages = [
            {"role": "user", "content": "How do I optimize learning rate?"},
            {"role": "assistant", "content": "Use cosine annealing with warmup."},
        ]
        hash1 = compute_content_sha256(messages)
        hash2 = compute_content_sha256(messages)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in hash1))

    def test_enrich_file_provenance_metadata(self) -> None:
        """Verify enrich_file adds required provenance metadata fields to JSONL records."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "sample_records.jsonl"
            sample = {
                "id": "test_sample_001",
                "category": "cli_procedure",
                "messages": [
                    {"role": "user", "content": "Run CLI training"},
                    {"role": "assistant", "content": "python -m training.train"},
                ],
            }
            jsonl_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            count = enrich_file(str(jsonl_path), split_name="train")
            self.assertEqual(count, 1)

            with open(jsonl_path, "r", encoding="utf-8") as f:
                enriched = json.loads(f.readline())

            self.assertIn("provenance", enriched)
            prov = enriched["provenance"]
            self.assertEqual(prov["source_doc_id"], "MANUAL_CLI.md")
            self.assertEqual(prov["split_rule"], "domain_stratified_train_v1")
            self.assertEqual(len(prov["content_sha256"]), 64)
            self.assertEqual(prov["curation_method"], "expert_authored_verified")

    def test_corpus_generator_invariants_and_gui_examples(self) -> None:
        """Verify SYSTEM_PROMPT invariants and example generation contract."""
        self.assertFalse(any(ord(c) > 0x1F000 for c in SYSTEM_PROMPT))
        self.assertFalse(any(0x2600 <= ord(c) <= 0x27BF for c in SYSTEM_PROMPT))

        gui_examples = generate_gui_examples()
        self.assertIsInstance(gui_examples, list)
        self.assertGreater(len(gui_examples), 0)

        for ex in gui_examples:
            self.assertIn("id", ex)
            self.assertEqual(ex["category"], "gui_procedure")
            self.assertIn("messages", ex)
            self.assertGreaterEqual(len(ex["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
