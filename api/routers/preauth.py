# api/routers/preauth.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends

from api.core.config import Settings, get_settings
from api.services.model_service import predict_from_processed_102, ensure_loaded
from api.services.store import save_review_if_needed, log_decision

router = APIRouter(tags=["preauth"])


def _save_feature_snapshot(settings: Settings, review_id: str, features: Dict[str, Any]) -> str:
    out_dir = settings.abs_feature_snapshots_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{review_id}.json"
    p.write_text(json.dumps(features, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _save_review_payload_snapshot(repo_root: Path, review_id: str, payload: Dict[str, Any]) -> Optional[str]:
    try:
        out_dir = repo_root / "payloads" / "review"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{review_id}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(p)
    except Exception:
        return None


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def _build_reason_details(
    codes: List[str],
    *,
    p_xgb: float,
    t_low: float,
    t_high: float,
    ae_err: Optional[float],
    ae_pct: Optional[float],
    ae_bkt: str,
    ae_review: Optional[float],
    ae_block: Optional[float],
) -> List[Dict[str, str]]:
    details: List[Dict[str, str]] = []

    for c in codes:
        if c == "xgb_low_risk":
            details.append(
                {
                    "code": c,
                    "message": f"XGBoost low risk: p_xgb={_fmt(p_xgb)} < t_low={_fmt(t_low)}.",
                }
            )
        elif c == "xgb_gray_zone":
            details.append(
                {
                    "code": c,
                    "message": f"XGBoost uncertainty: p_xgb={_fmt(p_xgb)} between t_low={_fmt(t_low)} and t_high={_fmt(t_high)}.",
                }
            )
        elif c == "xgb_high_risk":
            details.append(
                {
                    "code": c,
                    "message": f"XGBoost high risk: p_xgb={_fmt(p_xgb)} >= t_high={_fmt(t_high)}.",
                }
            )
        elif c == "ae_review_gate":
            details.append(
                {
                    "code": c,
                    "message": (
                        f"AE elevated anomaly (review gate): "
                        f"ae_error={_fmt(ae_err)}, ae_review={_fmt(ae_review)}, ae_block={_fmt(ae_block)}, "
                        f"percentile_vs_legit={_fmt(ae_pct)}, bucket={ae_bkt}."
                    ),
                }
            )
        elif c == "ae_block_gate":
            details.append(
                {
                    "code": c,
                    "message": (
                        f"AE extreme anomaly (block gate): "
                        f"ae_error={_fmt(ae_err)} >= ae_block={_fmt(ae_block)}, "
                        f"percentile_vs_legit={_fmt(ae_pct)}, bucket={ae_bkt}."
                    ),
                }
            )
        else:
            details.append({"code": c, "message": c})

    return details


@router.post("/preauth/decision")
def preauth_decision(body: Dict[str, Any], settings: Settings = Depends(get_settings)):
    meta = body.get("meta", {}) or {}
    features = body.get("features", {}) or {}

    # Model scoring (Option A: processed_102 input)
    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = predict_from_processed_102(settings, features)

    # Load AE thresholds for messaging (optional, but makes reasons correct)
    art = ensure_loaded(settings)
    ae_review_th = getattr(art, "ae_review", None)
    ae_block_th = getattr(art, "ae_block", None)

    # Decision + reason_codes
    reason_codes: List[str] = []
    decision: str

    if p_xgb < settings.xgb_t_low:
        decision = "APPROVE"
        # optional: if you want reasons even for approve
        # reason_codes = ["xgb_low_risk"]

    elif p_xgb >= settings.xgb_t_high:
        decision = "BLOCK"
        reason_codes = ["xgb_high_risk"]

    else:
        # gray zone -> consult AE bucket
        reason_codes = ["xgb_gray_zone"]

        if ae_bkt == "extreme":
            decision = "BLOCK"
            reason_codes.append("ae_block_gate")
        else:
            decision = "REVIEW"
            if ae_bkt == "elevated":
                reason_codes.append("ae_review_gate")

    # Review ID is deterministic (used in DB + snapshots)
    candidate_review_id = f"rev_{payload_hash[:16]}"
    feature_path: Optional[str] = None

    if decision == "REVIEW":
        feature_path = _save_feature_snapshot(settings, candidate_review_id, features)
        _save_review_payload_snapshot(
            settings.root_path(),
            candidate_review_id,
            {"meta": meta, "features": features, "reason_codes": reason_codes},
        )

    review_id = save_review_if_needed(
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

    resp: Dict[str, Any] = {
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
        resp["reason_details"] = _build_reason_details(
            reason_codes,
            p_xgb=p_xgb,
            t_low=settings.xgb_t_low,
            t_high=settings.xgb_t_high,
            ae_err=ae_err,
            ae_pct=ae_pct,
            ae_bkt=ae_bkt,
            ae_review=ae_review_th,
            ae_block=ae_block_th,
        )

    return resp
