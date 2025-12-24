#!/usr/bin/env python3
"""
FPN-10 | Explainability (SHAP – XGBoost only)

Purpose (Thesis-aligned)
- Global interpretability (feature importance)
- Local explanations for high-risk / review-like cases
- Offline analysis only (NOT in API path)

Outputs
- artifacts/explainability/shap/
    - shap_summary.png
    - shap_bar.png
    - shap_local_0.png
    - shap_local_1.png
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class SHAPConfig:
    model_path: Path = Path("artifacts/xgb_model.pkl")
    test_csv_path: Path = Path("data/processed/test.csv")
    target_col: str = "Class"

    shap_dir: Path = Path("artifacts/explainability/shap")

    shap_sample_size: int = 2000
    n_local_explanations: int = 2


# -----------------------------
# Helpers
# -----------------------------
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs(cfg: SHAPConfig) -> None:
    cfg.shap_dir.mkdir(parents=True, exist_ok=True)


def _load_test_data(path: Path, target_col: str):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from {path}")
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])
    return X, y


# -----------------------------
# SHAP logic
# -----------------------------
def run_shap(
    model,
    X_sample: np.ndarray,
    feature_names: np.ndarray,
    out_dir: Path,
    local_indices: List[int],
) -> Dict[str, str]:
    """
    Generates global + local SHAP plots using model-agnostic Explainer.
    """
    results: Dict[str, str] = {}

    # ---- model-agnostic prediction wrapper ----
    def model_fn(X):
        return model.predict_proba(X)[:, 1]

    explainer = shap.Explainer(
        model_fn,
        X_sample,
        feature_names=feature_names,
    )

    shap_values = explainer(X_sample)

    # -----------------------------
    # Global summary plot
    # -----------------------------
    summary_path = out_dir / "shap_summary.png"
    plt.figure()
    shap.summary_plot(
        shap_values.values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=30,
    )
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    results["summary_plot"] = str(summary_path)

    # -----------------------------
    # Global bar plot
    # -----------------------------
    bar_path = out_dir / "shap_bar.png"
    plt.figure()
    shap.summary_plot(
        shap_values.values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=30,
    )
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    results["bar_plot"] = str(bar_path)

    # -----------------------------
    # Local explanations (waterfall)
    # -----------------------------
    local_paths: List[str] = []
    for i, idx in enumerate(local_indices):
        path = out_dir / f"shap_local_{i}.png"
        plt.figure()
        shap.plots.waterfall(
            shap_values[idx],
            max_display=15,
            show=False,
        )
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        local_paths.append(str(path))

    results["local_plots"] = local_paths
    return results


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = SHAPConfig()
    _ensure_dirs(cfg)

    print("🔄 Loading model...")
    model = joblib.load(cfg.model_path)

    print("🔄 Loading test data...")
    X_df, y = _load_test_data(cfg.test_csv_path, cfg.target_col)
    X = X_df.values
    feature_names = X_df.columns.astype(str).to_numpy()

    # Subsample for performance
    rng = np.random.default_rng(42)
    n = X.shape[0]
    sample_size = min(cfg.shap_sample_size, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    X_sample = X[sample_idx]

    # Choose local examples:
    fraud_idx = np.where(y == 1)[0]
    if fraud_idx.size > 0:
        local_indices = fraud_idx[: cfg.n_local_explanations].tolist()
    else:
        proba = model.predict_proba(X)[:, 1]
        local_indices = np.argsort(proba)[::-1][: cfg.n_local_explanations].tolist()

    print("✅ Running SHAP...")
    results = run_shap(
        model=model,
        X_sample=X_sample,
        feature_names=feature_names,
        out_dir=cfg.shap_dir,
        local_indices=local_indices,
    )

    # Save metadata
    meta = {
        "generated_at": _now_utc(),
        "model": str(cfg.model_path),
        "test_data": str(cfg.test_csv_path),
        "outputs": results,
    }

    meta_path = cfg.shap_dir / "shap_index.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ SHAP explainability complete")
    print(f"📁 Outputs in: {cfg.shap_dir}")


if __name__ == "__main__":
    main()
