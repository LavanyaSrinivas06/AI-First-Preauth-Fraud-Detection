# api/routers/review.py
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.core.config import Settings, get_settings
from api.core.errors import ApiError
from api.schemas.review import ReviewCloseIn
from api.services.store import (
    insert_feedback_event,
    load_review_queue,
    get_review_by_id,
    update_review,
)

router = APIRouter(tags=["review"])

@router.get("/review/queue")
def review_queue(settings: Settings = Depends(get_settings)):
    items = load_review_queue(settings.abs_sqlite_path(), limit=200)
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

    return {"status": "ok", "review_id": review_id, "closed_as": analyst_decision, "feedback_id": fb_id}
