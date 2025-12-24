# api/routers/feedback.py
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.core.config import Settings, get_settings
from api.core.errors import ApiError
from api.services.store import get_review_by_id, insert_feedback_event

router = APIRouter(tags=["feedback"])


@router.post("/feedback/label")
def feedback_label(payload: dict, settings: Settings = Depends(get_settings)):
    """
    payload example:
    { "review_id": "rev_....", "outcome": "APPROVE", "notes": "false positive" }
    """
    review_id = payload.get("review_id")
    outcome = payload.get("outcome")
    notes = payload.get("notes")

    if not review_id:
        raise ApiError(400, "missing_required_field", "Missing review_id", param="review_id")
    if outcome not in {"APPROVE", "BLOCK"}:
        raise ApiError(400, "invalid_request_error", "outcome must be APPROVE or BLOCK", param="outcome")

    rev = get_review_by_id(settings.abs_sqlite_path(), review_id)
    if not rev:
        raise ApiError(404, "resource_missing", "Review not found.", param="review_id")

    fb_id = insert_feedback_event(settings.abs_sqlite_path(), review_id, outcome, notes)
    return {"status": "ok", "feedback_id": fb_id, "review_id": review_id}
