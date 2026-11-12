"""Recoverable errors are logged, never swallowed (audit Phase 2 validator findings)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from forex.mt5_session import _vault_mt5
from forex.mt5_session import logger as mt5_logger
from tools.llm_helper.knowledge_ingestor import _read_jsonl


class TestRecoverableErrorsAreLogged(unittest.TestCase):
    """Each recovery path keeps working and emits a warning with the cause."""

    def test_malformed_jsonl_line_is_logged_and_skipped(self) -> None:
        """A bad JSONL line is skipped with a warning naming its line number."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "records.jsonl"
            path.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
            with self.assertLogs("lemdata.llm.ingestor", level="WARNING") as logs:
                _text, records = _read_jsonl(path)
        self.assertEqual(records, [{"a": 1}, {"b": 2}])
        self.assertIn("line 2", logs.output[0])

    def test_unreadable_mt5_vault_is_logged(self) -> None:
        """A vault read failure falls back to {} and logs the reason."""
        with (
            mock.patch(
                "forex.mt5_session.get_secret_record", side_effect=OSError("vault locked")
            ),
            self.assertLogs(mt5_logger, level="WARNING") as logs,
        ):
            self.assertEqual(_vault_mt5(), {})
        self.assertIn("vault locked", logs.output[0])


if __name__ == "__main__":
    unittest.main()
