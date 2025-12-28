# api/services/model_service.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from api.core.config import Settings
from api.core.errors import ApiError


# -------------------------
# Loaded artifacts cache
# -------------------------
@dataclass
class LoadedArtifacts:
    xgb: Any
    ae: Any
    model_features: List[str]              # exact 102 processed features (ORDER MATTERS)
    ae_review: float
    ae_block: float
    ae_legit_sorted_errors: np.ndarray     # baseline distribution for AE percentile


_ART: Optional[LoadedArtifacts] = None


# -------------------------
# Helpers
# -------------------------
def _sha256_dict(d: Dict[str, Any]) -> str:
    raw = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ensure_numpy_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):  # sparse
        X = X.toarray()
    return np.asarray(X, dtype="float32")


def _load_ae_thresholds(art_dir: Path, settings: Settings) -> Tuple[float, float]:
    """
    Expects JSON like:
    { "review": 0.6915, "block": 4.8955, ... }
    """
    p = art_dir / settings.ae_thresholds_path
    if not p.exists():
        raise ApiError(500, "artifact_missing", f"Missing AE thresholds file: {p}")

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return float(obj["review"]), float(obj["block"])
    except Exception as e:
        raise ApiError(500, "artifact_invalid", f"Invalid AE thresholds JSON: {type(e).__name__}: {e}")


def _load_model_schema_from_processed_train(settings: Settings) -> List[str]:
    """
    Most reliable schema source = processed train.csv (because it matches what XGB saw).
    """
    train_path = Path(settings.data_dir) / "train.csv"
    if not train_path.exists():
        raise ApiError(500, "artifact_missing", f"Missing processed train.csv for schema: {train_path}")

    df = pd.read_csv(train_path, nrows=1)
    if "Class" not in df.columns:
        raise ApiError(500, "artifact_invalid", f"{train_path} must contain 'Class' column.")

    feats = [c for c in df.columns if c != "Class"]
    if len(feats) != 102:
        raise ApiError(
            500,
            "artifact_invalid",
            f"Expected 102 processed features, found {len(feats)} in {train_path}.",
        )
    return feats


def _compute_legit_error_baseline_from_val(settings: Settings, ae_model) -> np.ndarray:
    """
    Compute reconstruction errors distribution for legit (Class==0) from processed val.csv.
    """
    val_path = Path(settings.data_dir) / "val.csv"
    if not val_path.exists():
        raise ApiError(500, "artifact_missing", f"Missing validation file for AE baseline: {val_path}")

    try:
        df = pd.read_csv(val_path)
    except Exception as e:
        raise ApiError(500, "artifact_load_failed", f"Failed reading {val_path}: {type(e).__name__}: {e}")

    if "Class" not in df.columns:
        raise ApiError(500, "artifact_invalid", f"{val_path} must contain 'Class' column.")

    legit = df[df["Class"].astype(int) == 0]
    if legit.empty:
        raise ApiError(500, "artifact_invalid", f"No legit rows (Class==0) found in {val_path}")

    X_legit = _ensure_numpy_dense(legit.drop(columns=["Class"]).values)

    batch = 2048
    errs: List[float] = []
    for i in range(0, X_legit.shape[0], batch):
        xb = X_legit[i : i + batch]
        xb_rec = ae_model.predict(xb, verbose=0)
        eb = np.mean((xb - xb_rec) ** 2, axis=1)
        errs.extend([float(x) for x in eb])

    base = np.asarray(errs, dtype="float32")
    base.sort()
    return base


def _load_or_build_ae_baseline(art_dir: Path, settings: Settings, ae_model) -> np.ndarray:
    """
    Preferred: load baseline from artifacts/ae_errors/ae_baseline_legit_errors.npy
    Fallback: compute from val.csv and save it to that path.
    """
    baseline_path = art_dir / settings.ae_baseline_path
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    if baseline_path.exists():
        try:
            base = np.load(baseline_path)
            base = np.asarray(base, dtype="float32")
            base.sort()
            return base
        except Exception as e:
            raise ApiError(500, "artifact_invalid", f"Invalid AE baseline npy: {baseline_path} ({type(e).__name__}: {e})")

    # build once and persist
    base = _compute_legit_error_baseline_from_val(settings, ae_model)
    try:
        np.save(baseline_path, base)
    except Exception:
        # do not fail the API just because save failed
        pass
    return base


# -------------------------
# Public: load once
# -------------------------
def ensure_loaded(settings: Settings) -> LoadedArtifacts:
    global _ART
    if _ART is not None:
        return _ART

    art_dir = settings.artifacts_path()

    try:
        xgb = joblib.load(art_dir / settings.xgb_model_path)
        ae = load_model(art_dir / settings.ae_model_path, compile=False)

        model_features = _load_model_schema_from_processed_train(settings)
        ae_review, ae_block = _load_ae_thresholds(art_dir, settings)
        ae_legit_sorted_errors = _load_or_build_ae_baseline(art_dir, settings, ae)

    except ApiError:
        raise
    except FileNotFoundError as e:
        raise ApiError(500, "artifact_missing", f"Missing artifact file: {e}")
    except Exception as e:
        raise ApiError(500, "artifact_load_failed", f"Failed loading artifacts: {type(e).__name__}: {e}")

    _ART = LoadedArtifacts(
        xgb=xgb,
        ae=ae,
        model_features=model_features,
        ae_review=ae_review,
        ae_block=ae_block,
        ae_legit_sorted_errors=ae_legit_sorted_errors,
    )
    return _ART


# -------------------------
# Scoring utils
# -------------------------
def _validate_processed_payload(payload: Dict[str, Any], model_features: List[str]) -> None:
    missing = [k for k in model_features if k not in payload]
    if missing:
        raise ApiError(
            400,
            "missing_required_field",
            f"Missing required processed feature fields: {missing[:10]}{'...' if len(missing) > 10 else ''}",
            param="data",
        )


def score_xgb(art: LoadedArtifacts, X_df: pd.DataFrame) -> float:
    return float(art.xgb.predict_proba(X_df)[:, 1][0])


def ae_reconstruction_error(art: LoadedArtifacts, X_dense: np.ndarray) -> float:
    X_rec = art.ae.predict(X_dense, verbose=0)
    return float(np.mean((X_dense - X_rec) ** 2, axis=1)[0])


def ae_percentile_vs_legit(art: LoadedArtifacts, ae_err: float) -> Optional[float]:
    base = art.ae_legit_sorted_errors
    if base is None or len(base) == 0:
        return None
    rank = int(np.searchsorted(base, ae_err, side="right"))
    return float(100.0 * rank / float(len(base)))


def ae_bucket(art: LoadedArtifacts, ae_err: float) -> str:
    if ae_err >= art.ae_block:
        return "extreme"
    if ae_err >= art.ae_review:
        return "elevated"
    return "normal"


# -------------------------
# MAIN entry for Option A
# -------------------------
def predict_from_processed_102(
    settings: Settings,
    payload: Dict[str, Any],
) -> Tuple[float, Optional[float], str, Optional[float], str]:
    """
    Option A (thesis-clean):
    Payload MUST contain the exact 102 processed features (same as train/val/test).
    Returns:
      (p_xgb, ae_err_or_none, payload_hash, ae_pct_or_none, ae_bucket_or_na)
    """
    art = ensure_loaded(settings)
    payload_hash = _sha256_dict(payload)

    _validate_processed_payload(payload, art.model_features)

    # Build row in correct order
    row = {k: payload[k] for k in art.model_features}
    X_df = pd.DataFrame([row], columns=art.model_features)

    # XGB always
    p_xgb = score_xgb(art, X_df)

    # AE only in gray zone
    if not (settings.xgb_t_low <= p_xgb < settings.xgb_t_high):
        return p_xgb, None, payload_hash, None, "n/a"

    X_dense = _ensure_numpy_dense(X_df.values)
    ae_err = ae_reconstruction_error(art, X_dense)
    ae_pct = ae_percentile_vs_legit(art, ae_err)
    ae_bkt = ae_bucket(art, ae_err)

    return p_xgb, ae_err, payload_hash, ae_pct, ae_bkt
