from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def load_features(feature_path: Path) -> dict:
    return json.loads(feature_path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-id", required=True, help="e.g. rev_6dbcd9bcb15131cc")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--static-dir", default="dashboard/static")
    ap.add_argument("--xgb-model", default="xgb_model.pkl")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    static_dir = Path(args.static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)

    feature_path = artifacts_dir / "feature_snapshots" / f"{args.review_id}.json"
    if not feature_path.exists():
        raise SystemExit(f"Feature snapshot not found: {feature_path}")

    xgb_path = artifacts_dir / args.xgb_model
    if not xgb_path.exists():
        raise SystemExit(f"XGB model not found: {xgb_path}")

    feats = load_features(feature_path)

    # Build a single-row dataframe in the saved feature order
    X = pd.DataFrame([feats])

    model = joblib.load(xgb_path)

    # TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)

    # For binary classification, SHAP may return (n, m, 2). Use class=1 if needed.
    values = sv.values
    base_values = sv.base_values

    if isinstance(values, np.ndarray) and values.ndim == 3:
        # pick positive class
        values = values[:, :, 1]
        if isinstance(base_values, np.ndarray) and base_values.ndim == 2:
            base_values = base_values[:, 1]

    # Waterfall plot (best for thesis “why this ticket”)
    shap.plots.waterfall(
        shap.Explanation(
            values=values[0],
            base_values=float(base_values[0]) if np.ndim(base_values) else float(base_values),
            data=X.iloc[0].values,
            feature_names=list(X.columns),
        ),
        show=False,
        max_display=15,
    )

    out_path = static_dir / f"shap_{args.review_id}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
