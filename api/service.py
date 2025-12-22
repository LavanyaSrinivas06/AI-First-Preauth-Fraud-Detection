# api/service.py
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from api.config import (
    AE_PATH,
    AE_THRESHOLD_PATH,
    ALPHA,
    FEATURES_PATH,
    PREPROCESS_PATH,
    T_HIGH,
    T_LOW,
    XGB_PATH,
)
from api.utils import load_required_raw_features

logger = logging.getLogger(__name__)

class Preprocessor(Protocol):
    def transform(self, X: pd.DataFrame) -> Any: ...


@dataclass(frozen=True)
class Artifacts:
    preprocess: Preprocessor
    required_raw_features: List[str]
    xgb: Any
    ae: Any
    ae_threshold: float


_lock = threading.Lock()
_ARTIFACTS: Optional[Artifacts] = None

# If your training raw column names differ from API payload keys, map here.
FEATURE_ALIASES = {
    "amount": "Amount",  # training expects Amount
}


def _apply_aliases(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    for api_key, train_key in FEATURE_ALIASES.items():
        if api_key in out and train_key not in out:
            out[train_key] = out[api_key]
    return out


def _read_float(path: Union[str, Path]) -> float:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing AE threshold file: {p.as_posix()}")
    return float(p.read_text().strip())


def load_artifacts(
    preprocess_path: Union[str, Path] = PREPROCESS_PATH,
    features_path: Union[str, Path] = FEATURES_PATH,
    xgb_path: Union[str, Path] = XGB_PATH,
    ae_path: Union[str, Path] = AE_PATH,
    ae_threshold_path: Union[str, Path] = AE_THRESHOLD_PATH,
) -> Artifacts:
    global _ARTIFACTS
    with _lock:
        if _ARTIFACTS is not None:
            return _ARTIFACTS

        # --- preprocess ---
        p_pre = Path(preprocess_path)
        if not p_pre.exists():
            raise FileNotFoundError(f"Missing preprocessing artifact: {p_pre.as_posix()}")
        preprocess = joblib.load(p_pre)
        if not hasattr(preprocess, "transform"):
            raise TypeError("Loaded preprocess artifact does not implement `transform`")

        # --- features/raw schema ---
        required_raw_features = load_required_raw_features(features_path)

        # --- xgb ---
        p_xgb = Path(xgb_path)
        if not p_xgb.exists():
            raise FileNotFoundError(f"Missing XGB model artifact: {p_xgb.as_posix()}")
        xgb = joblib.load(p_xgb)

        # --- autoencoder ---
        p_ae = Path(ae_path)
        if not p_ae.exists():
            raise FileNotFoundError(f"Missing AE model artifact: {p_ae.as_posix()}")

        # IMPORTANT: fixes your current 'keras.metrics.mse' deserialization error
        ae = load_model(p_ae.as_posix(), compile=False)

        # --- AE threshold ---
        ae_threshold = _read_float(ae_threshold_path)

        _ARTIFACTS = Artifacts(
            preprocess=preprocess,
            required_raw_features=required_raw_features,
            xgb=xgb,
            ae=ae,
            ae_threshold=ae_threshold,
        )
        logger.info(
            "Artifacts loaded: preprocess=%s, raw_features=%d, xgb=%s, ae=%s, ae_T=%.10f",
            p_pre.as_posix(),
            len(required_raw_features),
            p_xgb.as_posix(),
            p_ae.as_posix(),
            ae_threshold,
        )
        return _ARTIFACTS


def unload_artifacts() -> None:
    global _ARTIFACTS
    with _lock:
        _ARTIFACTS = None


def is_ready() -> bool:
    return _ARTIFACTS is not None


def build_dataframe(payload: Dict[str, Any], required_raw_features: List[str]) -> pd.DataFrame:
    payload = _apply_aliases(payload)
    row = {f: payload.get(f, None) for f in required_raw_features}
    return pd.DataFrame([row], columns=required_raw_features)


def _hash_payload(payload: Dict[str, Any]) -> str:
    """
    Log-friendly hash: no raw values stored.
    """
    stable = repr(sorted(payload.keys())).encode("utf-8")
    return hashlib.sha256(stable).hexdigest()[:12]


def _score_xgb(xgb: Any, X: np.ndarray) -> float:
    p = xgb.predict_proba(X)[:, 1][0]
    return float(p)


def _score_ae(ae: Any, X: np.ndarray, ae_threshold: float) -> float:
    """
    Start simple per ticket: thresholded anomaly score (0 or 1).
    You can smooth later if you want.
    """
    X_rec = ae.predict(X, verbose=0)
    err = float(np.mean((X - X_rec) ** 2))
    return 1.0 if err >= ae_threshold else 0.0


def _final_decision(p_final: float) -> str:
    if p_final >= T_HIGH:
        return "block"
    if p_final <= T_LOW:
        return "approve"
    return "review"


def run_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = load_artifacts()
    df = build_dataframe(payload, artifacts.required_raw_features)

    # Transform using trained preprocess
    X_t = artifacts.preprocess.transform(df)

    # Ensure numpy array
    if hasattr(X_t, "toarray"):  # sparse
        X = X_t.toarray()
    else:
        X = np.asarray(X_t)

    p_xgb = _score_xgb(artifacts.xgb, X)
    p_ae = _score_ae(artifacts.ae, X, artifacts.ae_threshold)

    p_final = float(ALPHA * p_xgb + (1.0 - ALPHA) * p_ae)
    label = _final_decision(p_final)

    logger.info(
        "inference payload_hash=%s p_xgb=%.6f p_ae=%.6f p_final=%.6f label=%s",
        _hash_payload(payload),
        p_xgb,
        p_ae,
        p_final,
        label,
    )

    return {
        "label": label,
        "score_xgb": p_xgb,
        "score_ae": p_ae,
        "ensemble_score": p_final,
        "reason_codes": [],
    }
