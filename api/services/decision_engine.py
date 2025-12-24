# api/services/decision_engine.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def is_model_feature_complete(payload: Dict[str, Any]) -> bool:
    """
    Your model schema has strong dependence on V1..V28 (and engineered numeric).
    If those are missing, ML score tends to become meaningless (defaults -> 0s).
    """
    for k in [f"V{i}" for i in range(1, 29)]:
        if k in payload:
            return True
    # if you ever allow engineered fields explicitly, you can expand this check
    return False


def rule_reason_codes(payload: Dict[str, Any], amount_high: float = 900.0) -> List[str]:
    rc: List[str] = []

    country = payload.get("country")
    ip_country = payload.get("ip_country")
    if country and ip_country and str(country) != str(ip_country):
        rc.append("geo_mismatch")

    cur = payload.get("currency")
    card_cur = payload.get("card_currency")
    if cur and card_cur and str(cur) != str(card_cur):
        rc.append("currency_mismatch")

    hour = payload.get("hour")
    try:
        hr = int(hour) if hour is not None else None
    except Exception:
        hr = None
    if hr is not None and hr in {0, 1, 2, 3, 4, 5}:
        rc.append("night_txn")

    amt = payload.get("amount")
    try:
        a = float(amt) if amt is not None else None
    except Exception:
        a = None
    if a is not None and a >= amount_high:
        rc.append("high_amount")

    v1h = payload.get("velocity_1h")
    v24 = payload.get("velocity_24h")
    try:
        v1h_i = int(v1h) if v1h is not None else 0
    except Exception:
        v1h_i = 0
    try:
        v24_i = int(v24) if v24 is not None else 0
    except Exception:
        v24_i = 0
    if v1h_i >= 6:
        rc.append("high_velocity_1h")
    if v24_i >= 20:
        rc.append("high_velocity_24h")

    if payload.get("is_proxy_vpn") is True:
        rc.append("proxy_vpn")

    if payload.get("is_new_device") is True:
        rc.append("new_device")

    risky = {"NG", "GH", "PK", "BD", "RU", "UA"}
    if ip_country and str(ip_country) in risky:
        rc.append("risky_ip_country")

    return rc


def decide_checkout(
    payload: Dict[str, Any],
    p_xgb: float | None,
    ae_err: float | None,
    xgb_t_low: float,
    xgb_t_high: float,
    ae_review: float,
    ae_block: float,
) -> Tuple[str, List[str], str]:
    """
    Decision policy (thesis-friendly):
    - If model features are complete: use ML thresholds (+ AE in grey-zone)
    - If not: use simple rules to produce REVIEW/BLOCK sometimes (not always APPROVE)
    """
    if is_model_feature_complete(payload) and p_xgb is not None:
        # ML mode
        if p_xgb < xgb_t_low:
            return "APPROVE", [], "ml_xgb_low"
        if p_xgb >= xgb_t_high:
            return "BLOCK", ["xgb_high_risk"], "ml_xgb_high"
        # grey zone
        if ae_err is not None and ae_err >= ae_block:
            return "BLOCK", ["ae_high_recon_error"], "ml_ae_block"
        return "REVIEW", ["xgb_grey_zone"], "ml_grey_zone"

    # Rule fallback mode
    reason_codes = rule_reason_codes(payload)

    # Very simple scoring
    score = 0
    weights = {
        "geo_mismatch": 2,
        "currency_mismatch": 1,
        "night_txn": 1,
        "high_amount": 1,
        "high_velocity_1h": 2,
        "high_velocity_24h": 2,
        "proxy_vpn": 2,
        "new_device": 1,
        "risky_ip_country": 2,
    }
    for r in reason_codes:
        score += weights.get(r, 1)

    # thresholds (simple)
    if score >= 6:
        return "BLOCK", reason_codes, "rules_block"
    if score >= 3:
        return "REVIEW", reason_codes, "rules_review"
    return "APPROVE", reason_codes, "rules_approve"
