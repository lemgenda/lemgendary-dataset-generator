"""Unit and end-to-end tests for WebDataset visual QA, degradation schedule, and shard packing."""

from __future__ import annotations

import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

from core.visual_qa import (
    check_image_integrity,
    check_mask_integrity,
    compute_class_weights,
    validate_label_records,
)
from degrade.schedule import DegradationSchedule
from formats.base import Sample
from formats.webdataset import WebDatasetWriter


def _make_webp(arr: np.ndarray, lossless: bool = False) -> bytes:
    """Encode a numpy array as WebP bytes."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="WEBP", quality=90, lossless=lossless)
    return buf.getvalue()


def test_check_image_integrity_valid_blank_and_corrupt() -> None:
    """Verify image integrity check on valid, blank, and corrupt payloads."""
    rng = np.random.default_rng(0)
    valid_arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    res_ok = check_image_integrity(_make_webp(valid_arr))
    assert res_ok.ok is True
    assert res_ok.width == 32 and res_ok.height == 32

    blank_arr = np.full((32, 32, 3), 255, dtype=np.uint8)
    res_blank = check_image_integrity(_make_webp(blank_arr))
    assert res_blank.ok is False
    assert res_blank.reason == "blank_solid_color"

    res_corrupt = check_image_integrity(b"not-a-webp-image")
    assert res_corrupt.ok is False
    assert res_corrupt.reason is not None and res_corrupt.reason.startswith("decode_error:")


def test_check_mask_integrity_alignment_holes_and_classes() -> None:
    """Verify mask alignment, class ID, and hole ratio checks."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:28, 4:28] = 1
    mask_bytes = _make_webp(mask, lossless=True)

    res_ok = check_mask_integrity(mask_bytes, image_size=(32, 32), expected_classes={0, 1})
    assert res_ok.ok is True
    assert res_ok.hole_ratio == 0.0

    res_mismatch = check_mask_integrity(mask_bytes, image_size=(64, 64))
    assert res_mismatch.ok is False
    assert res_mismatch.reason is not None and res_mismatch.reason.startswith("size_mismatch:")

    # Checkerboard-like holes inside foreground
    hole_mask = mask.copy()
    hole_mask[6:26:2, 6:26:2] = 0
    res_holes = check_mask_integrity(
        _make_webp(hole_mask, lossless=True), image_size=(32, 32), max_hole_ratio=0.05
    )
    assert res_holes.ok is False
    assert res_holes.reason == "excessive_holes"


def test_label_qa_and_class_weights() -> None:
    """Verify class-imbalance weighting and label record schema audit."""
    weights = compute_class_weights({"a": 100, "b": 10000}, strategy="inverse_sqrt")
    assert weights["a"] > weights["b"]
    assert abs((weights["a"] + weights["b"]) / 2.0 - 1.0) < 1e-4

    audit = validate_label_records(
        [{"id": 1, "label": "cat"}, {"id": 2, "label": None}],
        required_keys={"id", "label"},
    )
    assert audit["uniform_schema"] is True
    assert audit["missing_required_count"] == 0
    assert audit["null_field_counts"]["label"] == 1


def test_degradation_schedule_determinism() -> None:
    """Verify deterministic degradation output and metadata across repeated calls."""
    rng = np.random.default_rng(1)
    img_bytes = _make_webp(rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8))
    sched = DegradationSchedule(global_seed=123, mode="cosine", min_severity=0.2, max_severity=0.8)

    out1, rec1 = sched.apply(img_bytes, sample_key="sample_001", progress=0.5)
    out2, rec2 = sched.apply(img_bytes, sample_key="sample_001", progress=0.5)
    assert out1 == out2
    assert rec1 == rec2
    assert 0.2 <= rec1["severity"] <= 0.8


def test_webdataset_end_to_end_with_qa_sidecar_and_visual_report(tmp_path) -> None:
    """Verify end-to-end WebDataset packing with QA sidecar and visual report."""
    rng = np.random.default_rng(42)
    sched = DegradationSchedule(global_seed=7, mode="linear")
    policy = SimpleNamespace(
        wds_shard_size_bytes=10_000_000, wds_max_count=2, shard_extension="data", label_json=None
    )

    writer = WebDatasetWriter()
    writer.open(tmp_path, policy)

    qa_stats = {"total": 0, "passed": 0, "rejected": 0}
    grid_tiles = []
    for i in range(4):
        arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        raw_bytes = _make_webp(arr)
        deg_bytes, deg_rec = sched.apply(raw_bytes, sample_key=f"s_{i:03d}", progress=i / 3.0)
        img_qa = check_image_integrity(deg_bytes)
        qa_stats["total"] += 1
        if img_qa.ok:
            qa_stats["passed"] += 1
            grid_tiles.append(Image.open(io.BytesIO(deg_bytes)).convert("RGB"))
            writer.write(
                Sample(
                    name=f"s_{i:03d}",
                    task="restoration",
                    split="train",
                    image_bytes=deg_bytes,
                    image_format="webp",
                    metadata={
                        "_qa": {"ok": img_qa.ok, "max_stddev": img_qa.max_stddev},
                        "_degradation": deg_rec,
                    },
                )
            )
        else:
            qa_stats["rejected"] += 1

    writer.close()

    shards = sorted((tmp_path / "shards" / "train").glob("shard-*.data"))
    assert len(shards) == 2
    assert (tmp_path / "shards" / "train" / "shard_offsets.json").exists()
    _verify_shard(shards[0])
    _write_visual_qa_report(tmp_path, grid_tiles, qa_stats)
    assert (tmp_path / "visual_qa_grid.png").exists()
    assert qa_stats["passed"] == 4


def _verify_shard(shard_path) -> None:
    """Verify shard tar members and JSON sidecar metadata."""
    with tarfile.open(shard_path, "r") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
        assert "s_000.webp" in names
        assert "s_000.json" in names
        json_member = tf.extractfile("s_000.json")
        assert json_member is not None
        meta = json.loads(json_member.read().decode("utf-8"))
        assert "_qa" in meta and "_degradation" in meta


def _write_visual_qa_report(
    tmp_path, grid_tiles: list[Image.Image], qa_stats: dict[str, int]
) -> None:
    """Write a 2x2 visual QA grid PNG and stats JSON to ``tmp_path``."""
    grid = Image.new("RGB", (64, 64))
    for idx, tile in enumerate(grid_tiles[:4]):
        grid.paste(tile, ((idx % 2) * 32, (idx // 2) * 32))
    grid.save(tmp_path / "visual_qa_grid.png")
    (tmp_path / "visual_qa_stats.json").write_text(json.dumps(qa_stats, indent=2), encoding="utf-8")
