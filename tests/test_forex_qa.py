"""Unit and end-to-end tests for Forex QA, leakage audit, and purged walk-forward splits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from forex.qa import (
    audit_horizon_leakage,
    build_split_audit_summary,
    classify_volatility_regime,
    flag_anomalous_bars,
    purged_walk_forward_splits,
    resample_ohlcv_utc,
)


def _sample_ohlcv(n: int = 120) -> pd.DataFrame:
    """Create a synthetic 15-minute OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0005, size=n))
    high = close + rng.uniform(0.0001, 0.0008, size=n)
    low = close - rng.uniform(0.0001, 0.0008, size=n)
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    vol = rng.integers(100, 1000, size=n)
    return pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_resample_ohlcv_utc_deterministic() -> None:
    """Verify UTC conversion and left-closed deterministic resampling."""
    df = _sample_ohlcv(40)
    res1 = resample_ohlcv_utc(df, rule="1h", source_tz="UTC")
    res2 = resample_ohlcv_utc(df, rule="1h", source_tz="UTC")
    assert res1.equals(res2)
    assert str(res1.index.tz) == "UTC"
    assert len(res1) == 10


def test_flag_anomalous_bars_and_regime() -> None:
    """Verify anomalous bar flagging and causal volatility regime labeling."""
    df = _sample_ohlcv(60)
    df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 0.001
    flags = flag_anomalous_bars(df)
    assert bool(flags.iloc[10]) is True

    regimes = classify_volatility_regime(df["close"], window=10)
    assert set(regimes.unique()).issubset({"low_vol", "normal_vol", "high_vol"})


def test_purged_walk_forward_splits_zero_leakage(tmp_path) -> None:
    """Verify purged walk-forward splits have zero horizon overlap and emit audit report."""
    n = 100
    pred_times = pd.Series(pd.date_range("2025-01-01", periods=n, freq="1h"))
    eval_times = pred_times + pd.Timedelta(hours=4)

    splits = purged_walk_forward_splits(n, pred_times, eval_times, n_splits=4, embargo_pct=0.02)
    summary = build_split_audit_summary(pred_times, eval_times, splits)
    assert len(summary) == 4
    assert all(row["ok"] and row["overlap_count"] == 0 for row in summary)

    # Confirm unpurged overlap is caught by audit_horizon_leakage
    bad_report = audit_horizon_leakage(
        pred_times, eval_times, np.arange(0, 50), np.arange(48, 80)
    )
    assert bad_report.ok is False
    assert bad_report.overlap_count > 0

    report_path = tmp_path / "forex_leakage_audit.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert report_path.exists()
