from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Request
from api.core.config import get_settings
from api.core.errors import ApiError
from api.core.logging import json_log, setup_logging

from api.schemas.risk_assessment import (
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    ThresholdsOut,
)
from api.services.model_service import ensure_loaded, predict_scores
from api.services.decision_engine import decide
from api.services.store import init_db, insert_risk_assessment, create_review

from api.schemas.review import ReviewItem

router = APIRouter()
logger = None


@router.post("/risk_assessments", response_model=RiskAssessmentResponse)
def create_risk_assessment(req: RiskAssessmentRequest, request: Request):
    settings = get_settings()
    global logger
    if logger is None:
        logger = setup_logging(settings)

    init_db(settings.abs_sqlite_path())
    art = ensure_loaded(settings)

    payload = req.data or {}
    start = time.perf_counter()

    p_xgb, ae_err, payload_hash = predict_scores(settings, payload)

    # decision
    res = decide(
        p_xgb=p_xgb,
        xgb_t_low=settings.xgb_t_low,
        xgb_t_high=settings.xgb_t_high,
        ae_error_val=ae_err,
        ae_t_review=art.ae_review,
        ae_t_block=art.ae_block,
    )

    latency_ms = (time.perf_counter() - start) * 1000.0
    now = int(time.time())
    rid = f"ra_{uuid.uuid4().hex[:24]}"

    # persist risk assessment
    insert_risk_assessment(
        settings.abs_sqlite_path(),
        {
            "id": rid,
            "created": now,
            "payload_hash": payload_hash,
            "label": res.label,
            "decided_by": res.decided_by,
            "score_xgb": float(p_xgb),
            "ae_error": float(res.ae_error) if res.ae_error is not None else None,
            "ae_bucket": res.ae_bucket,
            "latency_ms": float(latency_ms),
        },
    )

    # if review -> create review object
    if res.label == "review":
        review_id = f"rev_{uuid.uuid4().hex[:24]}"
        create_review(
            settings.abs_sqlite_path(),
            {
                "id": review_id,
                "created": now,
                "updated": now,
                "risk_assessment_id": rid,
                "status": "open",
                "analyst_decision": None,
                "notes": None,
            },
        )

    # log (no PII, only hash + scores)
    json_log(
        logger,
        {
            "ts": now,
            "request_id": getattr(request.state, "request_id", None),
            "object": "risk_assessment",
            "id": rid,
            "payload_hash": payload_hash,
            "score_xgb": float(p_xgb),
            "ae_error": float(res.ae_error) if res.ae_error is not None else None,
            "ae_bucket": res.ae_bucket,
            "label": res.label,
            "decided_by": res.decided_by,
            "latency_ms": float(latency_ms),
        },
    )

    return RiskAssessmentResponse(
        id=rid,
        label=res.label,
        decided_by=res.decided_by,
        score_xgb=float(p_xgb),
        ae_error=float(res.ae_error) if res.ae_error is not None else None,
        ae_bucket=res.ae_bucket,
        thresholds=ThresholdsOut(
            xgb_t_low=settings.xgb_t_low,
            xgb_t_high=settings.xgb_t_high,
            ae_review=art.ae_review,
            ae_block=art.ae_block,
        ),
        reason_codes=[],
        latency_ms=float(latency_ms),
        created=now,
    )


@router.get("/risk_assessments/{risk_assessment_id}", response_model=RiskAssessmentResponse)
def get_risk_assessment(risk_assessment_id: str):
    settings = get_settings()
    init_db(settings.abs_sqlite_path())
    art = ensure_loaded(settings)

    from api.services.store import get_risk_assessment as _get
    rec = _get(settings.abs_sqlite_path(), risk_assessment_id)
    if not rec:
        raise ApiError(404, "resource_missing", "Risk assessment not found.", param="risk_assessment_id")

    return RiskAssessmentResponse(
        id=rec["id"],
        label=rec["label"],
        decided_by=rec["decided_by"],
        score_xgb=float(rec["score_xgb"]),
        ae_error=float(rec["ae_error"]) if rec["ae_error"] is not None else None,
        ae_bucket=rec["ae_bucket"],
        thresholds=ThresholdsOut(
            xgb_t_low=settings.xgb_t_low,
            xgb_t_high=settings.xgb_t_high,
            ae_review=art.ae_review,
            ae_block=art.ae_block,
        ),
        reason_codes=[],
        latency_ms=float(rec["latency_ms"]),
        created=int(rec["created"]),
    )
