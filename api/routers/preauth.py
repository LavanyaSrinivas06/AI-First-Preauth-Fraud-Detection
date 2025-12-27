# api/routers/preauth.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from api.core.config import Settings, get_settings
from api.services.model_service import predict_from_processed_102
from api.services.store import save_review_if_needed, log_decision, save_feature_snapshot


router = APIRouter(tags=["preauth"])

REASON_TEXT = {
    "xgb_high_risk": "XGBoost predicts high fraud risk (above block threshold).",
    "xgb_gray_zone": "XGBoost is uncertain (between approve and block thresholds).",
    "ae_elevated": "Autoencoder anomaly is elevated (unusual vs. normal legitimate behavior).",
    "ae_extreme": "Autoencoder anomaly is extreme (strong deviation from legitimate baseline).",
}

def reason_details(codes: List[str]) -> List[Dict[str, str]]:
    return [{"code": c, "message": REASON_TEXT.get(c, c)} for c in codes]


def _persist_feature_snapshot(settings: Settings, payload_hash: str, features: Dict[str, Any]) -> str:
    """
    Save full processed_102 features so we can generate real SHAP later.
    Returns absolute path as string.
    """
    out_dir = settings.artifacts_path() / "feature_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / f"{payload_hash}.json"
    p.write_text(json.dumps(features, ensure_ascii=False), encoding="utf-8")
    return str(p)


#def preauth_decision(payload: Dict[str, Any], settings: Settings = Depends(get_settings)):
    #p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = predict_from_processed_102(settings, payload)
    
@router.post("/preauth/decision")
def preauth_decision(body: Dict[str, Any], settings: Settings = Depends(get_settings)):
    meta = body.get("meta", {}) or {}
    features = body.get("features", {}) or {}


    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = predict_from_processed_102(settings, features)
    feature_path = None
    review_id = None

    # review_id format must match store.py (rev_{payload_hash[:16]})
    candidate_review_id = f"rev_{payload_hash[:16]}"

# Only save feature snapshot when we actually create/keep a REVIEW item
# (this is what you'll use for per-ticket SHAP)

    if p_xgb < settings.xgb_t_low:
        decision = "APPROVE"
        reason_codes: List[str] = []
        ae_err, ae_pct, ae_bkt = None, None, "n/a"

    elif p_xgb >= settings.xgb_t_high:
        decision = "BLOCK"
        reason_codes = ["xgb_high_risk"]
        ae_err, ae_pct, ae_bkt = None, None, "n/a"

    else:
        reason_codes = ["xgb_gray_zone"]
        if ae_bkt == "extreme":
            decision = "BLOCK"
            reason_codes.append("ae_extreme")
        else:
            decision = "REVIEW"
            if ae_bkt == "elevated":
                reason_codes.append("ae_elevated")
    if decision == "REVIEW":
        review_id = candidate_review_id
        feature_path = save_feature_snapshot(
            settings.artifacts_path(),   # artifacts/
            review_id=review_id,
            features=features,           # exact processed-102 used by model
        )
    review_id = save_review_if_needed(
        sqlite_path=settings.abs_sqlite_path(),
        decision=decision,
        payload=features,     # still used for payload_min
        meta=meta,            # still used for logging
        p_xgb=p_xgb,
        ae_err=ae_err,
        payload_hash=payload_hash,
        reason_codes=reason_codes,
        ae_percentile=ae_pct,
        ae_bucket=ae_bkt,
        feature_path=feature_path,
    )


    log_decision(
        sqlite_path=settings.abs_sqlite_path(),
        decision=decision,
        payload=features,
        meta=meta,
        p_xgb=p_xgb,
        ae_err=ae_err,
        payload_hash=payload_hash,
        reason_codes=reason_codes,
        ae_percentile=ae_pct,
        ae_bucket=ae_bkt,
        feature_path=feature_path,
    )

    resp = {
        "decision": decision,
        "request": {"payload_hash": payload_hash, "review_id": review_id},
        "scores": {
            "xgb_probability": float(p_xgb),
            "ae_error": float(ae_err) if ae_err is not None else None,
            "ae_bucket": ae_bkt,
            "ae_percentile_vs_legit": float(ae_pct) if ae_pct is not None else None,
        },
    }

    if decision != "APPROVE":
        resp["reason_codes"] = reason_codes
        resp["reason_details"] = reason_details(reason_codes)

    return resp
