# api/routers/review.py
from __future__ import annotations

from typing import Optional
import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.core.config import Settings, get_settings
from api.core.errors import ApiError
from api.schemas.review import ReviewCloseIn
from dashboard.utils.explainability import generate_shap_png as generate_shap_plot
from api.services.model_service import ensure_loaded
from api.services.store import assign_review_to_analyst
from api.services.store import (
    insert_feedback_event,
    load_review_queue,
    get_review_by_id,
    update_review,
    set_review_shap_path,
)

router = APIRouter(tags=["review"])

@router.get("/review/queue")
def review_queue(status: str = "open", settings: Settings = Depends(get_settings)):
    """Return review queue. Accepts optional ?status=open|closed|all (default open)."""
    # sanitize
    status = (status or "open").lower()
    if status not in {"open", "closed", "all"}:
        status = "open"
    items = load_review_queue(settings.abs_sqlite_path(), limit=200, status=status)
    return {"items": items}


@router.get("/review/{review_id}")
def review_get(review_id: str, settings: Settings = Depends(get_settings)):
    item = get_review_by_id(settings.abs_sqlite_path(), review_id)
    if not item:
        raise ApiError(404, "resource_missing", "Review not found.", param="review_id")
    return item


@router.post("/review/{review_id}/close")
def review_close(review_id: str, body: ReviewCloseIn, settings: Settings = Depends(get_settings)):
    analyst_decision = str(body.analyst_decision).upper().strip()
    analyst = str(body.analyst).strip()
    notes = body.notes

    if analyst_decision not in {"APPROVE", "BLOCK"}:
        raise ApiError(
            400,
            "invalid_request_error",
            "analyst_decision must be APPROVE or BLOCK",
            param="analyst_decision",
        )
    if not analyst:
        raise ApiError(400, "invalid_request_error", "analyst is required", param="analyst")

    ok = update_review(
        settings.abs_sqlite_path(),
        review_id=review_id,
        analyst_decision=analyst_decision,
        analyst=analyst,
        notes=str(notes) if notes is not None else None,
    )
    if not ok:
        raise ApiError(404, "resource_missing", "Review not found.", param="review_id")

    # feedback event written automatically
    fb_id = insert_feedback_event(
        settings.abs_sqlite_path(),
        review_id=review_id,
        outcome=analyst_decision,
        notes=notes,
    )

    return {
    "ok": True,
    "status": "ok",
    "review_id": review_id,
    "closed_as": body.analyst_decision,
    "feedback_id": fb_id,
}




@router.post("/review/{review_id}/explain")
def explain_review(review_id: str, settings: Settings = Depends(get_settings)):
    review = get_review_by_id(settings.abs_sqlite_path(), review_id)
    if not review:
        raise ApiError(404, "resource_missing", "Review not found")

    # Prefer full processed features if stored (this gives more accurate SHAP).
    features = None
    processed_json = review.get("processed_features_json")
    try:
        if isinstance(processed_json, str):
            processed_json = json.loads(processed_json)
    except Exception:
        processed_json = None

    if isinstance(processed_json, dict) and any(k.startswith("num__") or k.startswith("cat__") for k in processed_json.keys()):
        features = processed_json
    else:
        features = review.get("payload_min")

    if not features:
        raise ApiError(400, "invalid_request_error", "No features stored")

    artifacts = ensure_loaded(settings)
    model = artifacts.xgb

    path = generate_shap_plot(
        review_id=review_id,
        model=model,
        features=features,
    )

    # OPTIONAL: store path in DB for audit
    try:
        set_review_shap_path(settings.abs_sqlite_path(), review_id=review_id, shap_path=path)
    except Exception:
        # don't fail the request if DB write fails
        pass

    return {
        "review_id": review_id,
        "shap_path": path,
        "status": "ok",
    }


class AssignIn(BaseModel):
    analyst: str

@router.post("/review/{review_id}/assign")
def assign_review(review_id: str, body: AssignIn, settings: Settings = Depends(get_settings)):
    analyst = str(body.analyst).strip()
    if not analyst:
        raise ApiError(400, "invalid_request_error", "analyst is required", param="analyst")

    # local import to avoid modifying top-level imports

    ok = assign_review_to_analyst(settings.abs_sqlite_path(), review_id=review_id, analyst=analyst)
    if not ok:
        raise ApiError(404, "resource_missing", "Review not found.", param="review_id")
    return {"status": "ok", "review_id": review_id, "analyst": analyst}

