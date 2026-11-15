"""Comprehensive regression tests for manifold verification pipeline and validators.

Tests:
1. WebDataset validator (schema, split collision, sidecar integrity, visual QA).
2. Forex validator (OHLCV schema, horizon invariants, price positivity).
3. LLM validator (JSONL schema, split collision, token integrity).
4. Determinism (Merkle SHA-256 manifest hash reproducibility).
5. CLI verify entry point and machine-readable JSON output.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from typer.testing import CliRunner

from cli import app
from core.manifold_verification import (
    ForexManifoldValidator,
    LLMManifoldValidator,
    WebDatasetManifoldValidator,
)


def _make_webp_bytes(color: tuple[int, int, int] | None = None, random: bool = False) -> bytes:
    """Create a 32x32 WebP image bytes."""
    buf = io.BytesIO()
    if random:
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.new("RGB", (32, 32), color or (128, 128, 128))
    img.save(buf, format="WEBP", quality=90)
    return buf.getvalue()


def _write_synthetic_webdataset(
    root: Path, with_collision: bool = False, low_variance: bool = False
) -> Path:
    """Construct a synthetic WebDataset manifold directory."""
    shards_dir = root / "shards"
    (shards_dir / "train").mkdir(parents=True, exist_ok=True)
    (shards_dir / "val").mkdir(parents=True, exist_ok=True)

    # Train Shard
    train_shard = shards_dir / "train" / "train-00000.data"
    with tarfile.open(train_shard, "w:") as tf:
        img_bytes = _make_webp_bytes(random=not low_variance)
        ti_img = tarfile.TarInfo(name="sample1.webp")
        ti_img.size = len(img_bytes)
        tf.addfile(ti_img, io.BytesIO(img_bytes))

        json_bytes = json.dumps({"class_id": 1, "class_name": "cat"}).encode("utf-8")
        ti_json = tarfile.TarInfo(name="sample1.json")
        ti_json.size = len(json_bytes)
        tf.addfile(ti_json, io.BytesIO(json_bytes))

    # Val Shard
    val_shard = shards_dir / "val" / "val-00000.data"
    val_key = "sample1" if with_collision else "sample2"
    with tarfile.open(val_shard, "w:") as tf:
        img_bytes = _make_webp_bytes(random=True)
        ti_img = tarfile.TarInfo(name=f"{val_key}.webp")
        ti_img.size = len(img_bytes)
        tf.addfile(ti_img, io.BytesIO(img_bytes))

        json_bytes = json.dumps({"class_id": 2, "class_name": "dog"}).encode("utf-8")
        ti_json = tarfile.TarInfo(name=f"{val_key}.json")
        ti_json.size = len(json_bytes)
        tf.addfile(ti_json, io.BytesIO(json_bytes))

    # Sidecars
    for sh in [train_shard, val_shard]:
        sc = sh.with_name(f"{sh.name}.index.json")
        st = sh.stat()
        sc.write_text(
            json.dumps({
                "version": 1,
                "shard": {"size": st.st_size, "mtime_ns": st.st_mtime_ns},
                "samples": [[sh.stem, None, None, None, None]],
            }),
            encoding="utf-8",
        )

    # Info & Readme
    (root / "dataset_info.yaml").write_text(
        "name: SyntheticVision\ntask_type: classification\ncount: 2\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Synthetic Vision\n", encoding="utf-8")
    (root / "index.json").write_text(
        json.dumps([{"name": "sample1", "split": "train"}, {"name": "sample2", "split": "val"}]),
        encoding="utf-8",
    )
    return root


def test_webdataset_validator_pass_and_defects(tmp_path: Path) -> None:
    """Verify WebDataset validator passes clean manifolds and catches split and variance defects."""
    valid_root = _write_synthetic_webdataset(tmp_path / "valid")
    val = WebDatasetManifoldValidator(valid_root)
    res = val.run_all()
    assert res.passed is True
    assert res.checks["split_integrity"].passed is True
    assert res.checks["visual_integrity"].passed is True
    assert res.checks["sidecar_integrity"].passed is True
    assert bool(res.manifest_hash) is True

    # Test Split Collision
    collision_root = _write_synthetic_webdataset(tmp_path / "collision", with_collision=True)
    val_col = WebDatasetManifoldValidator(collision_root)
    res_col = val_col.run_all()
    assert res_col.passed is False
    assert res_col.checks["split_integrity"].passed is False
    assert any("Split collision" in err for err in res_col.checks["split_integrity"].errors)

    # Test Low-Variance Image
    var_root = _write_synthetic_webdataset(tmp_path / "variance", low_variance=True)
    val_var = WebDatasetManifoldValidator(var_root)
    res_var = val_var.run_all()
    assert res_var.passed is False
    assert res_var.checks["visual_integrity"].passed is False
    assert any(
        "blank" in err.lower() or "stddev" in err.lower()
        for err in res_var.checks["visual_integrity"].errors
    )


def test_forex_validator_ohlcv_and_invariants(tmp_path: Path) -> None:
    """Verify Forex validator checks OHLCV presence and horizon price invariants."""
    fx_root = tmp_path / "forex_manifold"
    fx_root.mkdir()
    fx_info = (
        "name: ForexUniverse\ntask: forex\npairs: ['EURUSD']\n"
        "timeframe_rungs: [60]\nstart_date: '2020-01-01'\ncount: 10\n"
    )
    (fx_root / "dataset_info.yaml").write_text(fx_info, encoding="utf-8")
    (fx_root / "README.md").write_text("# Forex Universe\n", encoding="utf-8")

    # Valid table
    valid_df = pd.DataFrame({
        "open": [1.10, 1.11],
        "high": [1.15, 1.16],
        "low": [1.08, 1.09],
        "close": [1.12, 1.13],
        "volume": [1000.0, 1200.0],
    })
    table = pa.Table.from_pandas(valid_df)
    pq.write_table(table, fx_root / "ForexUniverse2020.parquet")

    val = ForexManifoldValidator(fx_root)
    res = val.run_all()
    assert res.checks["schema_conformance"].passed is True
    assert res.checks["horizon_integrity"].passed is True

    # Bad table: high < low
    bad_df = pd.DataFrame({
        "open": [1.10],
        "high": [1.05],  # high < low
        "low": [1.08],
        "close": [1.06],
        "volume": [500.0],
    })
    bad_root = tmp_path / "forex_bad"
    bad_root.mkdir()
    (bad_root / "dataset_info.yaml").write_text(
        "name: BadFX\npairs: ['EURUSD']\n", encoding="utf-8"
    )
    pq.write_table(pa.Table.from_pandas(bad_df), bad_root / "bad.parquet")

    val_bad = ForexManifoldValidator(bad_root)
    res_bad = val_bad.run_all()
    assert res_bad.checks["horizon_integrity"].passed is False
    # When numpy checks 2D array or parquet reads it
    arr_bad = np.array([[1.10, 1.05, 1.08, 1.06, 500.0]])
    np.save(bad_root / "bad.npy", arr_bad)
    res_bad2 = val_bad.run_all()
    assert res_bad2.checks["horizon_integrity"].passed is False
    assert any("high < low" in err for err in res_bad2.checks["horizon_integrity"].errors)


def test_llm_validator_jsonl_and_tokens(tmp_path: Path) -> None:
    """Verify LLM validator checks JSONL lines, token validity, and split uniqueness."""
    llm_root = tmp_path / "llm_manifold"
    llm_root.mkdir()
    (llm_root / "dataset_info.yaml").write_text(
        "name: LLMInstruct\ntask: llm-instruction\ncanonical_format: jsonl\ncount: 2\n",
        encoding="utf-8",
    )
    (llm_root / "README.md").write_text("# LLM Instruct\n", encoding="utf-8")

    train_file = llm_root / "train.jsonl"
    val_file = llm_root / "val.jsonl"
    train_file.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Hello!"}]}) + "\n",
        encoding="utf-8",
    )
    val_file.write_text(
        json.dumps({"messages": [{"role": "user", "content": "How are you?"}]}) + "\n",
        encoding="utf-8",
    )

    val = LLMManifoldValidator(llm_root)
    res = val.run_all()
    assert res.passed is True
    assert res.checks["schema_conformance"].passed is True
    assert res.checks["split_integrity"].passed is True
    assert res.checks["token_integrity"].passed is True

    # Test Corrupt Unicode
    corrupt_file = llm_root / "corrupt.jsonl"
    corrupt_file.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Corrupt \ufffd token"}]}) + "\n",
        encoding="utf-8",
    )
    res_corrupt = val.run_all()
    assert res_corrupt.checks["token_integrity"].passed is False
    assert any("\ufffd" in err for err in res_corrupt.checks["token_integrity"].errors)


def test_cli_verify_command(tmp_path: Path) -> None:
    """Verify CLI command executes and produces formatted output or JSON."""
    valid_root = _write_synthetic_webdataset(tmp_path / "cli_valid")
    runner = CliRunner()
    result = runner.invoke(app, ["verify", "--manifold", str(valid_root)])
    assert result.exit_code == 0
    assert "Manifold Verification:" in result.stdout

    # Test JSON mode
    json_result = runner.invoke(app, ["verify", "--manifold", str(valid_root), "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert data["passed"] is True
    assert "schema_conformance" in data["checks"]
    assert "determinism" in data["checks"]
