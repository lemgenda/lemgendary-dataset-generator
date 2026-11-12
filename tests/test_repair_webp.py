"""`repair` converts leftover PNG/JPEG shard members to WebP with the compiler's settings."""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from services.repair_service import RepairService
from tests.tar_fixtures import write_tar
from tools.repair.member_webp import convertible_members, webp_name
from tools.repair.shard_fix import plan_shard_fix
from utils.container_stream import IMAGE_WEBP_QUALITY
from utils.shard_audit import audit_shard


def _encoded(fmt: str, mode: str = "RGB", size: tuple[int, int] = (8, 6)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, 120).save(buf, format=fmt)
    return buf.getvalue()


def _palette_mask() -> bytes:
    img = Image.new("P", (8, 6), 0)
    img.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * 759)
    for x in range(4):
        img.putpixel((x, 1), 2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _members(path: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with tarfile.open(path, "r:") as tf:
        for member in tf.getmembers():
            handle = tf.extractfile(member)
            out[member.name] = handle.read() if handle is not None else b""
    return out


class MemberWebpTests(unittest.TestCase):
    """Repairs re-encode PNG/JPEG members and resync the recorded image format."""

    def setUp(self) -> None:
        """Synthetic manifold: a PNG pair, a raw JPEG, a PNG mask and a WebP sample."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.manifold = self.root / "Demo"
        self.shard = write_tar(self.manifold / "shards" / "train" / "shard-00045.data", [
            ("p0.png", _encoded("PNG")), ("p0.target.png", _encoded("PNG")),
            ("j0.jpg", _encoded("JPEG")), ("j0.json", b"{\"bbox\": []}"),
            ("m0.webp", _encoded("WEBP")), ("m0.mask.png", _palette_mask()),
            ("w0.webp", _encoded("WEBP")), ("w0.target.webp", _encoded("WEBP")),
        ])
        info = {"name": "Demo", "count": 4, "image_format": "png"}
        (self.manifold / "dataset_info.yaml").write_text(yaml.safe_dump(info), encoding="utf-8")

    def test_webp_name_replaces_last_extension(self) -> None:
        """Only the last extension changes; the role suffix stays."""
        self.assertEqual(webp_name("a.target.png"), "a.target.webp")
        self.assertEqual(webp_name("d/a.JPG"), "d/a.webp")

    def test_plan_lists_conversions(self) -> None:
        """The dry run names every PNG/JPEG member and the format change."""
        plan = RepairService().plan(self.manifold)
        self.assertEqual(plan.shard_fixes[0].convert,
                         {"p0.png", "p0.target.png", "j0.jpg", "m0.mask.png"})
        report = "\n".join(plan.report())
        self.assertIn("convert 4 png/jpg members to webp", report)
        self.assertIn("image_format 'png' -> 'webp'", report)

    def test_apply_converts_in_order_and_is_idempotent(self) -> None:
        """Members become WebP in place, masks keep their class values, rerun is a no-op."""
        service = RepairService()
        remaining = service.apply(service.plan(self.manifold))
        self.assertFalse(remaining.changed, remaining.report())
        members = _members(self.shard)
        self.assertEqual(list(members), [
            "p0.webp", "p0.target.webp", "j0.webp", "j0.json",
            "m0.webp", "m0.mask.webp", "w0.webp", "w0.target.webp",
        ])
        for name, data in members.items():
            if name.endswith(".webp"):
                with Image.open(io.BytesIO(data)) as img:
                    self.assertEqual(img.format, "WEBP", name)
        with Image.open(io.BytesIO(members["m0.mask.webp"])) as mask:
            histogram = mask.convert("L").histogram()
        self.assertEqual([value for value, n in enumerate(histogram) if n], [0, 2])
        with Image.open(io.BytesIO(_encoded("PNG"))) as src:
            expected = io.BytesIO()
            src.convert("RGB").save(expected, format="WEBP", quality=IMAGE_WEBP_QUALITY,
                                    method=2)
        self.assertEqual(members["p0.webp"], expected.getvalue())
        info = yaml.safe_load((self.manifold / "dataset_info.yaml").read_text(encoding="utf-8"))
        self.assertEqual(info["image_format"], "webp")
        self.assertEqual(sorted(p.name for p in self.shard.parent.iterdir()), [self.shard.name])
        self.assertIsNone(plan_shard_fix(audit_shard(self.shard)))

    def test_taken_webp_name_is_not_converted(self) -> None:
        """A PNG whose WebP name already exists in the sample is left alone."""
        shard = write_tar(self.root / "dup.data", [
            ("a.webp", _encoded("WEBP")), ("a.png", _encoded("PNG")),
        ])
        audit = audit_shard(shard)
        self.assertEqual(convertible_members(audit), frozenset())
        self.assertIsNone(plan_shard_fix(audit))


if __name__ == "__main__":
    unittest.main()
