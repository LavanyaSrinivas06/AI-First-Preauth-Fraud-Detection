# api/routers/feedback.py
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from api.core.config import Settings, get_settings
from api.core.errors import ApiError
from api.services.store import (
    get_review_by_id,
    insert_feedback_event,
    list_feedback_events,
    feedback_summary,
    export_feedback_samples,
)

router = APIRouter(tags=["feedback"])


class FeedbackIn(BaseModel):
    review_id: str
    outcome: str  # "APPROVE" | "BLOCK"
    notes: Optional[str] = None


@router.post("/feedback/label")
def feedback_label(body: FeedbackIn, settings: Settings = Depends(get_settings)):
    if body.outcome not in {"APPROVE", "BLOCK"}:
        raise ApiError(400, "invalid_request_error", "outcome must be APPROVE or BLOCK", param="outcome")

    review = get_review_by_id(settings.abs_sqlite_path(), body.review_id)
    if not review:
        raise ApiError(404, "resource_missing", "Review not found.", param="review_id")

    fb_id = insert_feedback_event(
        settings.abs_sqlite_path(),
        review_id=body.review_id,
        outcome=body.outcome,
        notes=body.notes,
    )
    return {"status": "ok", "feedback_id": fb_id, "review_id": body.review_id}


@router.get("/feedback/events")
def feedback_events(settings: Settings = Depends(get_settings), limit: int = 200):
    items = list_feedback_events(settings.abs_sqlite_path(), limit=int(limit))
    return {"items": items}


@router.get("/feedback/summary")
def feedback_metrics(settings: Settings = Depends(get_settings)):
    return feedback_summary(settings.abs_sqlite_path())


@router.get("/feedback/export")
def feedback_export(settings: Settings = Depends(get_settings), limit: int = 1000):
    items = export_feedback_samples(settings.abs_sqlite_path(), limit=int(limit))
    return {"items": items, "count": len(items)}
