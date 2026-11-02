"""
LemGendary Dataset Compiler: ``POST /api/llm/validate`` only opens files
inside the dataset root (review D-01, path part).

The dataset root is redirected to a temporary directory; nothing outside the
temp tree is read or written.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import llm as llm_routes

VALID_RECORD = {
    "id": "rec_1",
    "category": "troubleshooting_diagnostics",
    "messages": [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Help me."},
        {"role": "assistant", "content": "[CONFIRMED] Stage 1 triage, Stage 2 diagnosis, Stage 3 remediation."},
    ],
}


def _write_jsonl(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(VALID_RECORD) + "\n", encoding="utf-8")
    return path


class TestValidatePathContainment(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.root = base / "LemGendaryDatasets"
        self.outside_dir = base / "outside"
        self.inside_file = _write_jsonl(self.root / "ai_helper" / "data" / "ai_helper_train.jsonl")
        self.outside_file = _write_jsonl(self.outside_dir / "secret.jsonl")

        root_patch = patch.object(llm_routes, "_resolve_output_root", return_value=self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

        app = FastAPI()
        app.include_router(llm_routes.router, prefix="/api")
        self.client = TestClient(app)

    def _validate(self, file_path: str):
        return self.client.post("/api/llm/validate", json={"file_path": file_path})

    def test_relative_path_inside_root_is_validated(self) -> None:
        resp = self._validate("ai_helper/data/ai_helper_train.jsonl")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["total_records"], 1)

    def test_absolute_path_inside_root_is_still_accepted(self) -> None:
        resp = self._validate(str(self.inside_file))
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["total_records"], 1)

    def test_dotdot_traversal_is_refused(self) -> None:
        resp = self._validate("../outside/secret.jsonl")
        self.assertEqual(resp.status_code, 403)
        self.assertIn("outside the dataset root", resp.json()["detail"])

    def test_traversal_through_a_subfolder_is_refused(self) -> None:
        resp = self._validate("ai_helper/../../outside/secret.jsonl")
        self.assertEqual(resp.status_code, 403)

    def test_absolute_path_outside_root_is_refused(self) -> None:
        resp = self._validate(str(self.outside_file))
        self.assertEqual(resp.status_code, 403)

    def test_sibling_folder_sharing_the_root_prefix_is_refused(self) -> None:
        sibling = _write_jsonl(self.root.parent / (self.root.name + "Evil") / "x.jsonl")
        resp = self._validate(str(sibling))
        self.assertEqual(resp.status_code, 403)

    def test_missing_file_inside_root_is_404(self) -> None:
        resp = self._validate("ai_helper/data/missing.jsonl")
        self.assertEqual(resp.status_code, 404)

    def test_missing_file_outside_root_is_403_not_404(self) -> None:
        resp = self._validate(str(self.outside_dir / "missing.jsonl"))
        self.assertEqual(resp.status_code, 403)

    def test_directory_inside_root_is_404(self) -> None:
        resp = self._validate("ai_helper")
        self.assertEqual(resp.status_code, 404)

    def test_symlink_escaping_the_root_is_refused(self) -> None:
        link = self.root / "ai_helper" / "escape.jsonl"
        try:
            os.symlink(self.outside_file, link)
        except OSError as exc:  # Windows without Developer Mode / admin rights
            self.skipTest(f"symlink creation is not permitted on this system: {exc}")
        resp = self._validate("ai_helper/escape.jsonl")
        self.assertEqual(resp.status_code, 403)

    def test_inline_records_need_no_path(self) -> None:
        resp = self.client.post("/api/llm/validate", json={"records": [VALID_RECORD]})
        self.assertEqual(resp.status_code, 200, resp.text)


if __name__ == "__main__":
    unittest.main()
