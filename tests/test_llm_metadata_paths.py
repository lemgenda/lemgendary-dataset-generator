"""LLM manifold metadata: no machine-specific path in dataset_info.yaml; index sources."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.llm_helper.metadata_compiler import (
    LLMManifoldDescriptor,
    generate_llm_metadata,
)


class LlmMetadataPathTests(unittest.TestCase):
    """A build in a temp staging folder records no build folder path."""

    def setUp(self) -> None:
        """Fresh temporary directory per test."""
        self.staging = Path(tempfile.mkdtemp(prefix=".staging_build_ai_helper_"))
        self.addCleanup(shutil.rmtree, self.staging, ignore_errors=True)

    def test_path_and_sources(self) -> None:
        """Path and sources."""
        descriptor = LLMManifoldDescriptor("llm", ["instruction"], ["helper_corpus"])
        generate_llm_metadata(self.staging, "LemGendizedAIHelperLemGend",
                              [{"id": "a", "source": "docs"}, {"id": "b"}], [{"id": "c"}],
                              descriptor)
        info = yaml.safe_load((self.staging / "dataset_info.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("path", info)
        index = json.loads((self.staging / "index.json").read_text(encoding="utf-8"))
        self.assertEqual([e["source"] for e in index],
                         ["docs", "LemGendizedAIHelperLemGend", "LemGendizedAIHelperLemGend"])


if __name__ == "__main__":
    unittest.main()
