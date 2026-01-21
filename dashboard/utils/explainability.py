from pathlib import Path
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import logging
import importlib

def shap_path(review_id: str):
    return Path("dashboard/static/shap") / f"shap_{review_id}.png"


def _make_callable_model(m):
    """Return a callable wrapper for common model types so shap.Explainer can use it.

    Prefer sklearn-style predict_proba for binary probability output, then predict.
    If an xgboost Booster or sklearn wrapper is provided, use Booster.predict with DMatrix.
    """
    if callable(m):
        return m
    # sklearn-style API
    if hasattr(m, "predict_proba"):
        return lambda X: m.predict_proba(X)[:, 1]
    if hasattr(m, "predict"):
        return lambda X: m.predict(X)
    # xgboost Booster or XGBClassifier wrapper
    try:
        xgb = importlib.import_module("xgboost")
        if hasattr(m, "get_booster"):
            booster = m.get_booster()
            return lambda X: booster.predict(xgb.DMatrix(X))
        if isinstance(m, getattr(xgb, "Booster", object)):
            return lambda X: m.predict(xgb.DMatrix(X))
    except Exception:
        pass
    raise TypeError("The passed model is not callable and cannot be wrapped for SHAP explanation")

def generate_shap(review: dict):
    model = joblib.load("artifacts/models/xgb_model.pkl")

    with open("artifacts/features.json") as f:
        feature_names = json.load(f)

    # Prefer full processed feature vector if available (stored as JSON string in DB)
    processed_json = None
    try:
        processed_json = review.get("processed_features_json") if isinstance(review, dict) else None
        if isinstance(processed_json, str):
            processed_json = json.loads(processed_json)
    except Exception:
        logging.getLogger("fpn_api").warning("Failed to parse processed_features_json for SHAP generation")

    if processed_json and all((f in processed_json) for f in feature_names):
        features_full = processed_json
    else:
        # fallback to payload_min (may be a reduced set); fill missing features with 0.0
        features_full = review.get("payload_min", {}) if isinstance(review, dict) else {}

    # Build input vector in the correct feature order, using 0.0 for missing features
    try:
        x = np.array([[float(features_full.get(f, 0.0)) for f in feature_names]])
    except Exception as e:
        logging.getLogger("fpn_api").exception("Failed building SHAP input vector: %s", e)
        raise

    # Try TreeExplainer first (fast for tree models). If that fails due to model
    # metadata parsing issues, fall back to using shap.Explainer which is more
    # tolerant (though sometimes slower).
    explainer = None
    shap_values = None
    try:
        explainer = shap.TreeExplainer(model)
        # Some SHAP versions use different APIs; prefer the newer .shap_values when available
        try:
            shap_values = explainer.shap_values(x)
        except Exception:
            # newer Explainers may return an Explanation object when called
            vals = explainer(x)
            shap_values = getattr(vals, "values", vals)
    except Exception:
        logging.getLogger("fpn_api").warning("TreeExplainer failed, trying booster then generic Explainer")
        try:
            booster = getattr(model, "get_booster", lambda: model)()
            explainer = shap.TreeExplainer(booster)
            try:
                shap_values = explainer.shap_values(x)
            except Exception:
                vals = explainer(x)
                shap_values = getattr(vals, "values", vals)
        except Exception:
            # Final fallback: use a callable wrapper and generic shap.Explainer which
            # will call the model's predict/predict_proba under the hood.
            try:
                callable_model = _make_callable_model(model)
                explainer = shap.Explainer(callable_model, masker=shap.maskers.Independent(x))
                vals = explainer(x)
                shap_values = getattr(vals, "values", vals)
            except Exception as e:
                logging.getLogger("fpn_api").exception("Failed to create any SHAP explainer: %s", e)
                raise

    out = shap_path(review["id"])
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        x,
        feature_names=feature_names,
        max_display=10,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    return out


def generate_shap_png(review_id: str, model, features: dict):
    """Compatibility wrapper used by the API router.

    Signature expected by `api/routers/review.py` is (review_id, model, features).
    This builds the full input vector from `features` (a payload_min dict), fills
    missing features with 0.0, computes SHAP values using the provided model,
    saves the plot to `dashboard/static/shap/shap_{review_id}.png` and returns the
    path as a string.
    """
    with open("artifacts/features.json") as f:
        feature_names = json.load(f)

    # Build a full feature mapping from the provided `features` dict.
    features_full = features or {}

    try:
        x = np.array([[float(features_full.get(f, 0.0)) for f in feature_names]])
    except Exception as e:
        logging.getLogger("fpn_api").exception("Failed building SHAP input vector: %s", e)
        raise

    # Try TreeExplainer, then booster TreeExplainer, then generic shap.Explainer
    shap_values = None
    try:
        explainer = shap.TreeExplainer(model)
        try:
            shap_values = explainer.shap_values(x)
        except Exception:
            vals = explainer(x)
            shap_values = getattr(vals, "values", vals)
    except Exception:
        try:
            booster = getattr(model, "get_booster", lambda: model)()
            explainer = shap.TreeExplainer(booster)
            try:
                shap_values = explainer.shap_values(x)
            except Exception:
                vals = explainer(x)
                shap_values = getattr(vals, "values", vals)
        except Exception:
            try:
                callable_model = _make_callable_model(model)
                explainer = shap.Explainer(callable_model, masker=shap.maskers.Independent(x))
                vals = explainer(x)
                shap_values = getattr(vals, "values", vals)
            except Exception as e:
                logging.getLogger("fpn_api").exception("Failed to create any SHAP explainer: %s", e)
                raise

    out = shap_path(review_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Debug: persist the input vector and feature mapping alongside the image so
    # we can inspect what SHAP was given for this review during troubleshooting.
    try:
        dbg = out.with_suffix(".input.json")
        # compute compact top-5 positive/negative feature contributions for the single sample
        try:
            sv = shap_values
            # normalize shap_values into a numpy array for indexing
            arr = np.asarray(sv)
            # try to squeeze to shape (n_features,)
            arr_s = np.squeeze(arr)
            # if it's 2D with shape (n_samples, n_features), take first row
            if arr_s.ndim == 2 and arr_s.shape[0] > 1:
                sample_vals = arr_s[0]
            elif arr_s.ndim == 1:
                sample_vals = arr_s
            else:
                # fallback: flatten and take last dimension
                sample_vals = arr_s.flatten()
        except Exception:
            sample_vals = None

        top_pos = []
        top_neg = []
        if sample_vals is not None:
            try:
                # get indices sorted by contribution
                idx_desc = np.argsort(-sample_vals)
                idx_asc = np.argsort(sample_vals)
                # top positive
                for i in idx_desc[:5]:
                    v = float(sample_vals[i])
                    if v <= 0:
                        break
                    top_pos.append({"feature": feature_names[int(i)], "value": v})
                # top negative
                for i in idx_asc[:5]:
                    v = float(sample_vals[i])
                    if v >= 0:
                        break
                    top_neg.append({"feature": feature_names[int(i)], "value": v})
            except Exception:
                top_pos = []
                top_neg = []

        dump = {
            "feature_names": feature_names,
            "features_full": features_full,
            "x": x.tolist(),
            "top_positive": top_pos,
            "top_negative": top_neg,
        }
        with open(dbg, "w", encoding="utf-8") as _f:
            json.dump(dump, _f, ensure_ascii=False, indent=2)
    except Exception:
        logging.getLogger("fpn_api").exception("Failed to write SHAP debug input file")

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        x,
        feature_names=feature_names,
        max_display=10,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    # return a string path for API usage
    return str(out)
