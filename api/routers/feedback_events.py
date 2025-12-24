# api/routers/feedback_events.py
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends

from api.core.config import Settings, get_settings
from api.core.errors import ApiError
from api.services.store import get_review_by_id, insert_feedback_event

router = APIRouter(tags=["feedback"])


class FeedbackIn(BaseModel):
    review_id: str
    outcome: str                  # "APPROVE" | "BLOCK"
    notes: Optional[str] = None


@router.post("/feedback/label")
def feedback_label(body: FeedbackIn, settings: Settings = Depends(get_settings)):
    review = get_review_by_id(settings.abs_sqlite_path(), body.review_id)
    if not review:
        raise ApiError(404, "resource_missing", "Review not found.", param="review_id")

    ev = insert_feedback_event(
        settings.abs_sqlite_path(),
        review_id=body.review_id,
        outcome=body.outcome,
        notes=body.notes,
    )
    return {"event": ev}
