"""The VisionAuditor gates are applied to every sample and rejections recorded (audit 2.2.1)."""

import io
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from PIL import Image

from audit.reject_log import RejectLog
from audit.vision_audit import VisionAuditor
from core import sample_audit
from core.registry import initialize_registry
from core.sample_audit import header_ok, target_ok, valid_annotations


def _png(size: tuple[int, int] = (8, 8), mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format="PNG")
    return buf.getvalue()


class _AuditCase(unittest.TestCase):
    """Binds a real auditor and a registry-backed reject log to WORKER."""

    strict = False

    def setUp(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        self.db = Path(tmp) / "manifold_registry.db"
        initialize_registry(self.db).close()
        self.reject_log = RejectLog(self.db)
        audit = sample_audit.WORKER.audit
        auditor = VisionAuditor({"strict_ground_truth": self.strict})
        for name, value in (("auditor", auditor), ("reject_log", self.reject_log)):
            patcher = mock.patch.object(audit, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def rejects(self) -> list[Any]:
        """Flush the log and return its (sample_name, reject_code) rows."""
        self.reject_log.close()
        with sqlite3.connect(self.db) as conn:
            rows = conn.execute("SELECT sample_name, reject_code FROM reject_log").fetchall()
        conn.close()
        return rows


class TestHeaderAndAnnotations(_AuditCase):
    """Header and annotation gates (always on)."""

    def test_zero_byte_and_unknown_headers_rejected(self) -> None:
        """Empty or non-image bytes are rejected and logged; BMP is recognised."""
        self.assertFalse(header_ok("src", "empty", b""))
        self.assertFalse(header_ok("src", "junk", b"hello world!"))
        self.assertTrue(header_ok("src", "png", _png()))
        self.assertTrue(header_ok("src", "bmp", b"BM" + b"\x00" * 10))
        self.assertEqual(
            sorted(self.rejects()),
            [("empty", "ERR_ZERO_BYTES"), ("junk", "ERR_HEADER_INVALID")],
        )

    def test_invalid_annotations_dropped(self) -> None:
        """Degenerate or out-of-bounds boxes and keypoints are dropped, edge boxes kept."""
        anns = [
            {"type": "bbox", "cls": 0, "data": [0, 0, 100.0000001, 50]},
            {"type": "bbox", "cls": 0, "data": [10, 10, 0, 5]},
            {"type": "bbox", "cls": 0, "data": [90, 10, 20, 5]},
            {"type": "pose", "cls": 0, "data": [1, 1, 5, 5, 0.5, 1.4, 2]},
            {"type": "segmentation", "cls": 0, "data": [0.1, 0.2, 0.3, 0.4]},
        ]
        kept = valid_annotations("src", "s", anns, (100, 50))
        self.assertEqual(kept, [anns[0], anns[4]])
        self.assertEqual(len(self.rejects()), 3)

    def test_unknown_dims_leave_annotations(self) -> None:
        """Without decoded pixels nothing can be checked, so nothing is dropped."""
        anns = [{"type": "bbox", "cls": 0, "data": [0, 0, 0, 0]}]
        self.assertEqual(valid_annotations("src", "s", anns, (0, 0)), anns)


class TestTargets(_AuditCase):
    """Pair dimension checks (always) and mode/mask checks (lenient by default)."""

    def test_dimension_mismatch_rejected(self) -> None:
        """A restoration target of another size is rejected; SR pairs are not compared."""
        img = Image.new("RGB", (8, 8))
        self.assertFalse(target_ok("src", "r", "restoration", img, _png((16, 16))))
        self.assertTrue(target_ok("src", "sr", "super-resolution", img, _png((16, 16))))
        self.assertTrue(target_ok("src", "ok", "restoration", img, Image.new("L", (8, 8))))
        self.assertEqual(self.rejects(), [("r", "ERR_PAIR_MISMATCH")])

    def test_target_path_is_checked_and_closed(self) -> None:
        """A target given as a path is opened lazily for its size."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        (tmp / "t.png").write_bytes(_png((4, 4)))
        self.assertFalse(target_ok("src", "p", "segmentation", Image.new("RGB", (8, 8)),
                                   str(tmp / "t.png")))


class TestStrictTargets(_AuditCase):
    """``strict_ground_truth`` adds the colour-mode and single-channel mask checks."""

    strict = True

    def test_mode_and_mask_checks(self) -> None:
        """Mode mismatch rejects restoration; an RGB or one-class mask rejects segmentation."""
        img = Image.new("RGB", (8, 8))
        self.assertFalse(target_ok("src", "mode", "restoration", img, Image.new("L", (8, 8))))
        self.assertFalse(target_ok("src", "rgb", "segmentation", img, Image.new("RGB", (8, 8))))
        mask = Image.new("L", (8, 8))
        mask.putpixel((1, 1), 3)
        self.assertTrue(target_ok("src", "good", "segmentation", img, mask))
        self.assertEqual(
            sorted(code for _, code in self.rejects()),
            ["ERR_MASK_INVALID", "ERR_PAIR_MODE_MISMATCH"],
        )


if __name__ == "__main__":
    unittest.main()
