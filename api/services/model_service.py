from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from api.core.config import Settings
from api.core.errors import ApiError
from api.services.feature_adapter import adapt_payload_to_processed_102, validate_checkout_payload


@dataclass
class LoadedArtifacts:
    preprocess: Any
    xgb: Any
    ae: Any
    model_features: List[str]


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

        # IMPORTANT: use pipeline feature names as the contract for "102 processed features"
        try:
            model_features = list(preprocess.get_feature_names_out())
        except Exception:
            # fallback: if preprocess doesn't expose names, we cannot safely adapt
            raise ApiError(500, "artifact_invalid", "Preprocess pipeline does not expose feature names.")

    except FileNotFoundError as e:
        raise ApiError(500, "artifact_missing", f"Missing artifact file: {e}")
    except ApiError:
        raise
    except Exception as e:
        raise ApiError(500, "artifact_load_failed", f"Failed loading artifacts: {type(e).__name__}: {e}")

    _ART = LoadedArtifacts(preprocess=preprocess, xgb=xgb, ae=ae, model_features=model_features)
    return _ART


def score_xgb(art: LoadedArtifacts, X: np.ndarray) -> float:
    return float(art.xgb.predict_proba(X)[:, 1][0])


def ae_reconstruction_error(art: LoadedArtifacts, X: np.ndarray) -> float:
    X = np.asarray(X).astype("float32")
    X_rec = art.ae.predict(X, verbose=0)
    return float(np.mean((X - X_rec) ** 2, axis=1)[0])


def build_reason_codes(payload: Dict[str, Any], settings: Settings) -> List[str]:
    reasons: List[str] = []

    country = str(payload.get("country"))
    ip_country = str(payload.get("ip_country"))
    currency = str(payload.get("currency"))
    card_currency = str(payload.get("card_currency"))
    hour = int(payload.get("hour", 12))
    v1h = int(payload.get("velocity_1h", 0))
    v24h = int(payload.get("velocity_24h", 0))
    is_vpn = bool(payload.get("is_proxy_vpn", False))
    amt = float(payload.get("amount", 0.0))

    if country and ip_country and country != ip_country:
        reasons.append("geo_mismatch")

    if currency and card_currency and currency != card_currency:
        reasons.append("currency_mismatch")

    if hour in {0, 1, 2, 3, 4, 5}:
        reasons.append("night_txn")

    if v1h >= settings.rule_velocity_1h_block:
        reasons.append("high_velocity_1h")
    elif v1h >= settings.rule_velocity_1h_review:
        reasons.append("med_velocity_1h")

    if v24h >= settings.rule_velocity_24h_block:
        reasons.append("high_velocity_24h")
    elif v24h >= settings.rule_velocity_24h_review:
        reasons.append("med_velocity_24h")

    if amt >= settings.rule_amount_block:
        reasons.append("high_amount")

    if is_vpn:
        reasons.append("proxy_vpn")

    return reasons


def predict_scores(
    settings: Settings,
    payload: Dict[str, Any],
) -> Tuple[float, float, str, List[str]]:
    """
    Returns: (p_xgb, ae_err, payload_hash, reason_codes)
    """
    art = ensure_loaded(settings)
    payload_hash = _sha256_dict(payload)

    # validate checkout payload
    try:
        validate_checkout_payload(payload)
    except ValueError as e:
        raise ApiError(400, "missing_required_field", str(e), param="data")

    # adapt checkout -> 102 processed features
    df_102 = adapt_payload_to_processed_102(payload, art.model_features)

    # Some repos store preprocess as identity; some as real transformer.
    # We try transform, else fallback to raw 102 matrix.
    try:
        X = art.preprocess.transform(df_102)
    except Exception:
        X = df_102.values

    p_xgb = score_xgb(art, X)
    ae_err = ae_reconstruction_error(art, X)

    reason_codes = build_reason_codes(payload, settings)
    return p_xgb, ae_err, payload_hash, reason_codes
