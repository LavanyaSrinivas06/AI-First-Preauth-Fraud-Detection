from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from api.core.config import Settings, get_settings
from api.services.model_service import predict_scores
from api.services.store import save_review_if_needed

router = APIRouter(tags=["preauth"])


@router.post("/preauth/decision")
def preauth_decision(payload: Dict[str, Any], settings: Settings = Depends(get_settings)):
    """
    Returns: APPROVE | REVIEW | BLOCK
    """

    p_xgb, ae_err, payload_hash, reason_codes = predict_scores(settings, payload)

    # --- Rule overrides (stable demos) ---
    hard_block = False
    hard_review = False

    if "high_velocity_1h" in reason_codes or "high_velocity_24h" in reason_codes:
        hard_block = True

    if "proxy_vpn" in reason_codes and "geo_mismatch" in reason_codes:
        hard_block = True

    if "high_amount" in reason_codes and ("geo_mismatch" in reason_codes or "currency_mismatch" in reason_codes):
        hard_block = True

    if (not hard_block) and (
        "geo_mismatch" in reason_codes
        or "currency_mismatch" in reason_codes
        or "night_txn" in reason_codes
        or "med_velocity_1h" in reason_codes
        or "med_velocity_24h" in reason_codes
        or "proxy_vpn" in reason_codes
    ):
        hard_review = True

    # --- Model-based decision ---
    if hard_block:
        decision = "BLOCK"
    elif hard_review:
        decision = "REVIEW"
    else:
        # purely model-driven
        if p_xgb < settings.xgb_t_low:
            decision = "APPROVE"
        elif p_xgb >= settings.xgb_t_high:
            decision = "BLOCK"
        else:
            # gray zone -> AE helps
            decision = "BLOCK" if ae_err >= settings.ae_block else "REVIEW"

    # Persist only REVIEW for dashboard queue
    save_review_if_needed(
        decision=decision,
        payload=payload,
        p_xgb=p_xgb,
        ae_err=ae_err,
        payload_hash=payload_hash,
        reason_codes=reason_codes,
    )

    # For thesis: only show reason_codes when not APPROVE
    resp = {
        "decision": decision,
        "scores": {"xgb_probability": round(p_xgb, 6), "ae_error": round(ae_err, 6)},
    }
    if decision != "APPROVE":
        resp["reason_codes"] = reason_codes

    return resp
