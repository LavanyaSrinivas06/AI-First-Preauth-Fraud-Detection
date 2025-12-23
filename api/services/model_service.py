from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from api.core.config import Settings
from api.core.errors import ApiError


@dataclass
class LoadedArtifacts:
    preprocess: Any
    xgb: Any
    ae: Any
    features: Dict[str, Any]
    ae_review: float
    ae_block: float


_ART: Optional[LoadedArtifacts] = None


def _sha256_dict(d: Dict[str, Any]) -> str:
    raw = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ensure_loaded(settings: Settings) -> LoadedArtifacts:
    global _ART
    if _ART is not None:
        return _ART

    art = settings.artifacts_path()
    try:
        preprocess = joblib.load(art / settings.preprocess_path)
        xgb = joblib.load(art / settings.xgb_model_path)
        ae = load_model(art / settings.ae_model_path, compile=False)

        features = json.loads((art / settings.features_path).read_text(encoding="utf-8"))

        thresholds = json.loads((art / settings.ae_thresholds_path).read_text(encoding="utf-8"))
        ae_review = float(thresholds["review"])
        ae_block = float(thresholds["block"])
    except FileNotFoundError as e:
        raise ApiError(500, "artifact_missing", f"Missing artifact file: {e}")
    except Exception as e:
        raise ApiError(500, "artifact_load_failed", f"Failed loading artifacts: {type(e).__name__}: {e}")

    _ART = LoadedArtifacts(
        preprocess=preprocess,
        xgb=xgb,
        ae=ae,
        features=features,
        ae_review=ae_review,
        ae_block=ae_block,
    )
    return _ART


def build_dataframe(payload: Dict[str, Any], art: LoadedArtifacts, strict: bool = True) -> pd.DataFrame:
    # Use the exact expected raw feature order (before preprocessing)
    cat = art.features.get("categorical_features", [])
    num = art.features.get("numerical_features", [])
    raw_feature_cols = list(num) + list(cat)

    if strict:
        missing = [c for c in raw_feature_cols if c not in payload]
        if missing:
            raise ApiError(
                400,
                "missing_required_field",
                f"Missing required fields: {missing[:10]}{'...' if len(missing) > 10 else ''}",
                param="data",
            )

    # fill non-provided columns with None (then pipeline handles)
    row = {c: payload.get(c, None) for c in raw_feature_cols}
    return pd.DataFrame([row], columns=raw_feature_cols)


def score_xgb(art: LoadedArtifacts, X: np.ndarray) -> float:
    p = float(art.xgb.predict_proba(X)[:, 1][0])
    return p


def ae_reconstruction_error(art: LoadedArtifacts, X: np.ndarray) -> float:
    # X expected numeric dense array
    X = np.asarray(X).astype("float32")
    X_rec = art.ae.predict(X, verbose=0)
    err = float(np.mean((X - X_rec) ** 2, axis=1)[0])
    return err


def predict_scores(
    settings: Settings,
    payload: Dict[str, Any],
) -> Tuple[float, Optional[float], str]:
    """
    Returns: (p_xgb, ae_error or None, payload_hash)
    """
    art = ensure_loaded(settings)
    payload_hash = _sha256_dict(payload)

    df = build_dataframe(payload, art, strict=settings.strict_feature_check)

    # preprocess -> model space
    try:
        X = art.preprocess.transform(df)
    except Exception as e:
        raise ApiError(400, "preprocess_failed", f"Preprocessing failed: {type(e).__name__}: {e}", param="data")

    # xgb proba
    p_xgb = score_xgb(art, X)

    # AE error only needed in gray-zone by caller; keep simple: compute always (fast enough)
    ae_err = ae_reconstruction_error(art, X)

    return p_xgb, ae_err, payload_hash
