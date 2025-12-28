from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_feature_path(sqlite_path: Path, review_id: str) -> Optional[str]:
    if not sqlite_path.exists():
        return None
    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT feature_path FROM reviews WHERE id = ?", (review_id,))
        row = cur.fetchone()
        if not row:
            return None
        return row[0]
    finally:
        con.close()


def _save_waterfall(sv, X: pd.DataFrame, out_path: Path, max_display: int = 15) -> None:
    import shap  # lazy import

    values = sv.values
    base_values = getattr(sv, "base_values", None)

    if isinstance(values, np.ndarray) and values.ndim == 3:
        values = values[:, :, 1]
        if isinstance(base_values, np.ndarray) and base_values.ndim == 2:
            base_values = base_values[:, 1]

    bv = float(base_values[0]) if isinstance(base_values, (list, np.ndarray)) else float(base_values) if base_values is not None else 0.0

    shap.plots.waterfall(
        shap.Explanation(
            values=values[0],
            base_values=bv,
            data=X.iloc[0].values,
            feature_names=list(X.columns),
        ),
        show=False,
        max_display=int(max_display),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _generate_shap(model, X: pd.DataFrame, out_path: Path, max_display: int) -> Tuple[bool, str]:
    try:
        import shap  # noqa

        # Try TreeExplainer first
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer(X)
            _save_waterfall(sv, X, out_path, max_display=max_display)
            return True, f"Saved: {out_path}"
        except Exception:
            pass

        # Robust fallback (avoids base_score parsing issues)
        background = X.copy()
        masker = shap.maskers.Independent(background)

        def f(xx: np.ndarray):
            df = pd.DataFrame(xx, columns=X.columns)
            return model.predict_proba(df)[:, 1]

        explainer2 = shap.Explainer(f, masker)
        sv2 = explainer2(X.values)
        _save_waterfall(sv2, X, out_path, max_display=max_display)
        return True, f"Saved (fallback explainer): {out_path}"

    except Exception as e:
        return False, f"SHAP failed: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-id", required=True)
    ap.add_argument("--out-dir", default="dashboard/static")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--sqlite", default="artifacts/stores/inference_store.sqlite")
    ap.add_argument("--xgb-model", default="models/xgb_model.pkl")
    ap.add_argument("--max-display", type=int, default=15)
    ap.add_argument("--feature-path", default=None, help="Optional override; otherwise read from sqlite")
    args = ap.parse_args()

    review_id = str(args.review_id)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shap_{review_id}.png"

    fp = args.feature_path
    if not fp:
        fp = _read_feature_path(Path(args.sqlite), review_id)

    if not fp:
        raise SystemExit("Review has no feature_path in sqlite (snapshot not stored).")

    feature_path = Path(fp)
    if not feature_path.exists():
        raise SystemExit(f"Feature snapshot not found: {feature_path}")

    feats: Dict[str, Any] = json.loads(feature_path.read_text(encoding="utf-8"))
    X = pd.DataFrame([feats])

    model_path = Path(args.artifacts_dir) / args.xgb_model
    if not model_path.exists():
        raise SystemExit(f"XGB model not found: {model_path}")

    model = joblib.load(model_path)

    ok, msg = _generate_shap(model, X, out_path, max_display=int(args.max_display))
    if not ok:
        raise SystemExit(msg)
    print(msg)


if __name__ == "__main__":
    main()
