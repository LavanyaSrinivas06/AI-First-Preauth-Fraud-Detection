# api/routers/preauth.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from api.core.logging import json_log

from fastapi import APIRouter, Depends

from api.core.config import Settings, get_settings
from api.services.model_service import predict_from_processed_102, ensure_loaded
from api.services.store import save_review_if_needed, log_decision
from api.services.reason_builder import build_reason_details_v2


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


@router.post("/preauth/decision")
def preauth_decision(body: Dict[str, Any], settings: Settings = Depends(get_settings)):
    meta = body.get("meta", {}) or {}
    features = body.get("features", {}) or {}

    # Model scoring (Option A: processed_102 input)
    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = predict_from_processed_102(settings, features)

    # Load AE thresholds for messaging (optional, but makes reasons correct)
    art = ensure_loaded(settings)
    model_version = getattr(art, "model_version", "v1")
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
        model_version=model_version,   
        processed_features_json=json.dumps(features)
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
        model_version=model_version,   # NEW
    )

    # Structured JSON audit log (supplementary to DB record) for tracing and offline analysis
    try:
        logger = logging.getLogger("fpn_api")
        audit = {
            "decision": decision,
            "review_id": review_id,
            "payload_hash": payload_hash,
            "p_xgb": float(p_xgb) if p_xgb is not None else None,
            "ae_error": float(ae_err) if ae_err is not None else None,
            "ae_bucket": ae_bkt,
            "reason_codes": reason_codes,
            "model_version": model_version,
            # include any client-provided request id if present in meta
            "request_id": meta.get("request_id") if isinstance(meta, dict) else None,
        }
        json_log(logger, audit)
    except Exception:
        # Logging should not break request handling
        pass

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
        resp["reason_details"] = build_reason_details_v2(
            decision=decision,
            p_xgb=p_xgb,
            t_low=settings.xgb_t_low,
            t_high=settings.xgb_t_high,
            features=features,
            ae_percentile=ae_pct,
            ae_bucket=ae_bkt,
        )


    return resp
