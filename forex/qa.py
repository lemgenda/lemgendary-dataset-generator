"""Forex time-series dataset QA, leakage prevention, and purged walk-forward splitting.

Implements:
- Deterministic UTC resampling with left-closed/left-labeled bars.
- Reproducible anomalous bar flagging (OHLC invariants, non-positive prices, MAD spikes).
- Causal volatility regime labeling (low / normal / high volatility).
- Purged, embargoed walk-forward splits (Lopez de Prado).
- Strict horizon-overlap leakage audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeakageAuditReport:
    """Summary of horizon-overlap and ordering leakage checks."""

    ok: bool
    overlap_count: int
    monotonic_ok: bool
    train_count: int
    test_count: int


def resample_ohlcv_utc(
    df: pd.DataFrame,
    rule: str = "1h",
    source_tz: str = "UTC",
) -> pd.DataFrame:
    """Normalize timestamps to UTC and deterministically resample OHLCV bars."""
    if df.empty:
        return df.copy()
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is None:
        idx = idx.tz_localize(source_tz)
    out.index = idx.tz_convert("UTC")
    out = out[~out.index.duplicated(keep="first")].sort_index()

    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in out.columns:
        agg["volume"] = "sum"
    resampled = out.resample(rule, label="left", closed="left").agg(agg)
    return resampled.dropna(subset=["open", "high", "low", "close"])


def flag_anomalous_bars(
    df: pd.DataFrame,
    mad_z_thresh: float = 6.0,
) -> pd.Series:
    """Flag anomalous OHLCV bars using reproducible price, invariant, and MAD rules."""
    o = np.asarray(df["open"], dtype=np.float64)
    h = np.asarray(df["high"], dtype=np.float64)
    lo = np.asarray(df["low"], dtype=np.float64)
    c = np.asarray(df["close"], dtype=np.float64)

    bad_price = (o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)
    bad_hl = (h < lo) | (h < np.maximum(o, c)) | (lo > np.minimum(o, c))

    log_ret = np.zeros_like(c)
    if len(c) > 1:
        valid = (c[:-1] > 0) & (c[1:] > 0)
        ratio = np.where(valid, c[1:] / np.maximum(c[:-1], 1e-12), 1.0)
        log_ret[1:] = np.log(ratio)
    med = float(np.median(log_ret))
    mad = float(np.median(np.abs(log_ret - med)))
    if mad > 1e-12:
        spike = (np.abs(log_ret - med) / (1.4826 * mad)) > mad_z_thresh
    else:
        spike = np.zeros_like(c, dtype=bool)

    out = pd.Series(bad_price | bad_hl | spike)
    out.index = df.index
    out.name = "is_anomalous"
    return out


def classify_volatility_regime(
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Classify each bar into ``low_vol``, ``normal_vol``, or ``high_vol`` causally."""
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    vol = log_ret.rolling(window=max(2, window), min_periods=2).std().fillna(0.0)
    q33 = vol.expanding(min_periods=2).quantile(0.33).bfill()
    q67 = vol.expanding(min_periods=2).quantile(0.67).bfill()

    regimes = np.where(
        vol <= q33,
        "low_vol",
        np.where(vol >= q67, "high_vol", "normal_vol"),
    )
    out = pd.Series(regimes)
    out.index = close.index
    out.name = "vol_regime"
    return out


def audit_horizon_leakage(
    pred_times: pd.Series,
    eval_times: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> LeakageAuditReport:
    """Verify no training label interval overlaps with any test interval."""
    monotonic_ok = pred_times.is_monotonic_increasing
    if len(train_idx) == 0 or len(test_idx) == 0:
        return LeakageAuditReport(
            ok=monotonic_ok,
            overlap_count=0,
            monotonic_ok=monotonic_ok,
            train_count=len(train_idx),
            test_count=len(test_idx),
        )

    t_start = pred_times.iloc[test_idx].min()
    t_end = eval_times.iloc[test_idx].max()
    tr_pred = pred_times.iloc[train_idx]
    tr_eval = eval_times.iloc[train_idx]
    overlaps = int(((tr_pred <= t_end) & (tr_eval >= t_start)).sum())
    return LeakageAuditReport(
        ok=(overlaps == 0) and monotonic_ok,
        overlap_count=overlaps,
        monotonic_ok=monotonic_ok,
        train_count=len(train_idx),
        test_count=len(test_idx),
    )


def _purged_train_indices(
    pred_arr: np.ndarray,
    eval_arr: np.ndarray,
    test_start: int,
    test_end: int,
    embargo: int,
) -> np.ndarray:
    """Return training indices with overlapping intervals and post-test embargo removed."""
    n_samples = len(pred_arr)
    t_min = pred_arr[test_start:test_end].min()
    t_max = eval_arr[test_start:test_end].max()
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[test_start : min(n_samples, test_end + embargo)] = False
    train_mask &= ~((pred_arr <= t_max) & (eval_arr >= t_min))
    return np.flatnonzero(train_mask)


def purged_walk_forward_splits(
    n_samples: int,
    pred_times: pd.Series,
    eval_times: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate purged, embargoed walk-forward ``(train_idx, test_idx)`` splits."""
    if n_splits < 1 or n_samples < n_splits + 1:
        raise ValueError(f"Invalid n_splits={n_splits} for n_samples={n_samples}")

    pred_arr = pred_times.to_numpy()
    eval_arr = eval_times.to_numpy()
    fold_size = n_samples // (n_splits + 1)
    embargo = max(1, int(n_samples * embargo_pct))
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = min(n_samples, (k + 1) * fold_size) if k < n_splits else n_samples
        train_idx = _purged_train_indices(
            pred_arr, eval_arr, test_start, test_end, embargo
        )
        splits.append((train_idx, np.arange(test_start, test_end)))

    return splits


def build_split_audit_summary(
    pred_times: pd.Series,
    eval_times: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    """Run ``audit_horizon_leakage`` on every fold and return serializable dicts."""
    summary: list[dict[str, Any]] = []
    for fold_id, (tr, te) in enumerate(splits):
        rep = audit_horizon_leakage(pred_times, eval_times, tr, te)
        summary.append(
            {
                "fold": fold_id,
                "ok": rep.ok,
                "overlap_count": rep.overlap_count,
                "monotonic_ok": rep.monotonic_ok,
                "train_count": rep.train_count,
                "test_count": rep.test_count,
            }
        )
    return summary
