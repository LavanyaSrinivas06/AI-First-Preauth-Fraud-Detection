from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class PSIRules:
    stable: float = 0.10
    moderate: float = 0.25


def _safe_pct(arr: np.ndarray) -> np.ndarray:
    s = arr.sum()
    if s <= 0:
        return np.ones_like(arr, dtype=float) / max(1, len(arr))
    return arr / s


def _psi_from_hist(expected_counts: np.ndarray, actual_counts: np.ndarray, eps: float = 1e-6) -> float:
    e = _safe_pct(expected_counts.astype(float))
    a = _safe_pct(actual_counts.astype(float))
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(
    expected: pd.Series,
    actual: pd.Series,
    n_bins: int = 10,
    clip_quantiles: Tuple[float, float] = (0.001, 0.999),
) -> float:
    exp = pd.to_numeric(expected, errors="coerce").dropna().to_numpy()
    act = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()

    if len(exp) < 50 or len(act) < 50:
        return float("nan")

    lo, hi = np.quantile(exp, clip_quantiles[0]), np.quantile(exp, clip_quantiles[1])
    exp = np.clip(exp, lo, hi)
    act = np.clip(act, lo, hi)

    # quantile bins on baseline
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(exp, qs))
    if len(edges) < 3:  # near-constant feature
        # treat as 1 bin
        return 0.0

    exp_hist, _ = np.histogram(exp, bins=edges)
    act_hist, _ = np.histogram(act, bins=edges)
    return _psi_from_hist(exp_hist, act_hist)


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    e = expected.astype(str).fillna("NA")
    a = actual.astype(str).fillna("NA")
    cats = sorted(set(e.unique()).union(set(a.unique())))
    if len(cats) <= 1:
        return 0.0

    e_counts = e.value_counts().reindex(cats, fill_value=0).to_numpy()
    a_counts = a.value_counts().reindex(cats, fill_value=0).to_numpy()
    return _psi_from_hist(e_counts, a_counts)


def psi_feature(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    # if looks numeric -> numeric psi, else categorical
    exp_num = pd.to_numeric(expected, errors="coerce")
    act_num = pd.to_numeric(actual, errors="coerce")
    num_ratio = float(exp_num.notna().mean())
    if num_ratio > 0.98:
        return psi_numeric(exp_num, act_num, n_bins=n_bins)
    return psi_categorical(expected, actual)


def psi_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    n_bins: int = 10,
    feature_allowlist: Optional[list[str]] = None,
) -> pd.DataFrame:
    # Align columns
    base_cols = set(baseline_df.columns)
    curr_cols = set(current_df.columns)
    common = sorted(list(base_cols.intersection(curr_cols)))

    if feature_allowlist is not None:
        allow = set(feature_allowlist)
        common = [c for c in common if c in allow]

    rows = []
    for col in common:
        val = psi_feature(baseline_df[col], current_df[col], n_bins=n_bins)
        rows.append((col, val))

    out = pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)
    return out
