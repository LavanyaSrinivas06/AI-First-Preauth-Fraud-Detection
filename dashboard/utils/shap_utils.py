from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# shap is optional at runtime; we import inside functions to avoid dashboard crash if missing


def shap_png_path_for_review(static_dir: str, review_id: str) -> str:
    return str((Path(static_dir) / f"shap_{review_id}.png").resolve())


def terminal_command_for_shap(review_id: str, out_dir: str = "dashboard/static") -> str:
    return f"python scripts/generate_shap_for_review.py --review-id {review_id} --out-dir {out_dir}"


def _read_feature_path_from_sqlite(sqlite_path: Path, review_id: str) -> Optional[str]:
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


def generate_shap_png(
    *,
    review: Dict[str, Any],
    static_dir: str,
    artifacts_dir: str = "artifacts",
    sqlite_path: str = "artifacts/stores/inference_store.sqlite",
    xgb_model_relpath: str = "models/xgb_model.pkl",
    max_display: int = 15,
) -> Tuple[bool, str]:
    """
    Returns (ok, message). If ok=True, image is written to dashboard/static/shap_{review_id}.png
    Robust fallback if TreeExplainer fails (your '[5E-1]' base_score issue).
    """

    review_id = str(review.get("id") or "")
    if not review_id:
        return False, "Missing review.id"

    out_dir = Path(static_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shap_{review_id}.png"

    feature_path = review.get("feature_path")
    if not feature_path:
        # try sqlite as fallback (handles stale API responses)
        feature_path = _read_feature_path_from_sqlite(Path(sqlite_path), review_id)

    if not feature_path:
        return False, "Review has no feature_path. (Snapshot not stored)"

    fp = Path(str(feature_path))
    if not fp.exists():
        return False, f"Feature snapshot not found: {fp}"

    feats = json.loads(fp.read_text(encoding="utf-8"))
    X = pd.DataFrame([feats])

    model_path = (Path(artifacts_dir) / xgb_model_relpath)
    if not model_path.exists():
        return False, f"XGB model not found: {model_path}"

    model = joblib.load(model_path)

    # ---------
    # Try SHAP TreeExplainer first
    # ---------
    try:
        import shap  # noqa

        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer(X)
            _save_waterfall(sv, X, out_path, max_display=max_display)
            return True, f"Saved: {out_path}"
        except Exception:
            # Some SHAP+xgboost versions break on base_score parsing; fallback below
            pass

        # ---------
        # Robust fallback: explain predict_proba via model-agnostic explainer for 1 row
        # (slower but stable for thesis demo)
        # ---------
        background = X.copy()
        masker = shap.maskers.Independent(background)

        def f(xx: np.ndarray):
            df = pd.DataFrame(xx, columns=X.columns)
            proba = model.predict_proba(df)[:, 1]
            return proba

        explainer2 = shap.Explainer(f, masker)
        sv2 = explainer2(X.values)
        _save_waterfall(sv2, X, out_path, max_display=max_display)
        return True, f"Saved (fallback explainer): {out_path}"

    except Exception as e:
        return False, f"SHAP generation failed: {type(e).__name__}: {e}"


def _save_waterfall(sv, X: pd.DataFrame, out_path: Path, max_display: int = 15) -> None:
    import shap  # noqa

    values = sv.values
    base_values = getattr(sv, "base_values", None)

    # Handle (n, m, 2) -> pick class 1
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
