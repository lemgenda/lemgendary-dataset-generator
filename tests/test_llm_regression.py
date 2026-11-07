"""Regression and reproducibility verification tests for LLM dataset compiler.

Verifies:
1. Hierarchical chunking and code fence isolation.
2. Taxonomy disambiguation (REST API vs Image Restoration).
3. Authority tier calculation and provenance attribution.
4. Non-overlapping 3-way split partitioning and zero leakage.
5. Deterministic compiler reproducibility (identical hashes across repeat runs).
"""

from __future__ import annotations

import hashlib
import unittest

from tools.llm_helper.base_compiler import _partition_records
from tools.llm_helper.knowledge_synthesizer import (
    _authority_tier,
    _classify_category,
    _iter_sections,
    synthesize_instruction_from_section,
)
from tools.llm_helper.quality_validator import QualityValidator


class TestLLMCompilerRegression(unittest.TestCase):
    """Test regression suites for phases 1-7 deliverables."""

    def test_code_fence_isolation_and_heading_stack(self) -> None:
        """Code fence comments must not create fake document sections."""
        content = (
            "# Main Guide\n\n"
            "Intro paragraph with at least 100 characters so the body is preserved by the "
            "synthesizer cleanly without truncation.\n\n"
            "```python\n"
            "# Inside code block comment\n"
            "def foo():\n"
            "    pass\n"
            "```\n\n"
            "## Subsection Child\n\n"
            "Body paragraph for the subsection child that also has over 100 characters of "
            "length for testing."
        )
        sections = list(_iter_sections(content))
        leaf_names = [leaf for leaf, _, _ in sections]
        self.assertNotIn("# Inside code block comment", leaf_names)
        self.assertIn("Main Guide", leaf_names)
        self.assertIn("Subsection Child", leaf_names)

        # Check full heading stack hierarchy
        child_path = next(path for leaf, path, _ in sections if leaf == "Subsection Child")
        self.assertEqual(child_path, "Main Guide > Subsection Child")

    def test_taxonomy_disambiguation(self) -> None:
        """REST API endpoints must classify as api_procedure not model_pipeline."""
        cat_api = _classify_category("PAPER_API.md", "REST API Endpoints Specification")
        self.assertEqual(cat_api, "api_procedure")

        cat_diag = _classify_category("MANUAL.md", "Diagnostic Intake Protocols and Failure Modes")
        self.assertEqual(cat_diag, "troubleshooting_diagnostic")

    def test_authority_tier_attribution(self) -> None:
        """Authority tiers must match code manifest > manuals > whitepapers > guides."""
        self.assertEqual(_authority_tier("unified_models_v2.yaml"), "tier_1_manifest")
        self.assertEqual(_authority_tier("MANUAL_CLI.md"), "tier_2_operational_manual")
        self.assertEqual(_authority_tier("PAPER_LEMGENDARY_NIMA.md"), "tier_3_whitepaper")
        self.assertEqual(_authority_tier("UNKNOWN_GUIDE.md"), "tier_4_educational")

    def test_provenance_record_payload(self) -> None:
        """Synthesized instruction records must carry structured provenance fields."""
        content = (
            "Detailed operational instructions for configuring REST endpoints in the LemGendary "
            "ecosystem. Ensure all parameters are properly validated before dispatching."
        )
        rec = synthesize_instruction_from_section(
            doc_id="MANUAL_API.md",
            header="REST API > Config",
            content=content,
            system_prompt="System prompt.",
            index=0,
        )
        self.assertIsNotNone(rec)
        if rec:
            self.assertEqual(rec["authority_tier"], "tier_2_operational_manual")
            self.assertEqual(rec["source_doc"], "MANUAL_API.md")
            self.assertEqual(rec["heading_path"], "REST API > Config")
            self.assertEqual(rec["section_title"], "Config")

    def test_zero_leakage_in_3way_partition(self) -> None:
        """Category-stratified partition must yield 0 prompt overlap between splits."""
        records = []
        for cat in ("gui_procedure", "cli_command", "api_procedure"):
            for chunk_idx in range(10):
                records.append({
                    "id": f"rec_{cat}_{chunk_idx}",
                    "source_doc": f"doc_{cat}.md",
                    "category": cat,
                    "messages": [
                        {"role": "system", "content": "Sys."},
                        {"role": "user", "content": f"User question for {cat} #{chunk_idx}?"},
                        {"role": "assistant", "content": f"Assistant answer for {cat}."},
                    ],
                })

        train, val, test = _partition_records(records, val_split=0.20, test_split=0.20)
        self.assertTrue(len(train) > 0)
        self.assertTrue(len(val) > 0)
        self.assertTrue(len(test) > 0)
        self.assertEqual(len(train) + len(val) + len(test), len(records))

        train_ids = {r["id"] for r in train}
        val_ids = {r["id"] for r in val}
        test_ids = {r["id"] for r in test}

        # Assert no ID or record leakage across partition boundaries
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

        # Assert each split contains all categories
        for split_recs in (train, val, test):
            cats = {r["category"] for r in split_recs}
            self.assertEqual(cats, {"gui_procedure", "cli_command", "api_procedure"})

    def test_duplicate_prompt_detection(self) -> None:
        """Validator must flag identical normalized prompts across corpus as a warning."""
        validator = QualityValidator()
        recs = [
            {
                "id": "rec_1",
                "category": "cli_command",
                "messages": [
                    {"role": "system", "content": "Sys."},
                    {"role": "user", "content": "What is the CLI flag?"},
                    {"role": "assistant", "content": "Use flag A."},
                ],
            },
            {
                "id": "rec_2",
                "category": "cli_command",
                "messages": [
                    {"role": "system", "content": "Sys."},
                    {"role": "user", "content": "what   is  the CLI flag? "},
                    {"role": "assistant", "content": "Use flag B."},
                ],
            },
        ]
        is_valid, issues = validator.validate_corpus(recs)
        self.assertTrue(is_valid)  # Warnings do not block validity
        self.assertTrue(any(i.field == "prompt" for i in issues))

    def test_bit_level_dual_build_reproducibility(self) -> None:
        """Dual compiler executions under identical inputs must produce bit-for-bit hashes."""
        records = [
            {
                "id": f"rec_{i}",
                "source_doc": "doc.md",
                "category": "cli_command",
                "messages": [
                    {"role": "system", "content": "Sys."},
                    {"role": "user", "content": f"User question {i}?"},
                    {"role": "assistant", "content": f"Assistant answer {i}."},
                ],
            }
            for i in range(20)
        ]
        train1, val1, test1 = _partition_records(records, val_split=0.15, test_split=0.15)
        train2, val2, test2 = _partition_records(records, val_split=0.15, test_split=0.15)

        h1 = hashlib.sha256(str(train1 + val1 + test1).encode("utf-8")).hexdigest()
        h2 = hashlib.sha256(str(train2 + val2 + test2).encode("utf-8")).hexdigest()
        self.assertEqual(h1, h2, "Dual partition runs must produce bit-level identical hashes")


if __name__ == "__main__":
    unittest.main()
