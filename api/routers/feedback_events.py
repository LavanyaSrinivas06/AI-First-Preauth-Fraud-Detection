from __future__ import annotations

import time
import uuid

from fastapi import APIRouter
from api.core.config import get_settings
from api.core.errors import ApiError
from api.services.store import init_db, insert_feedback_event, get_risk_assessment
from api.schemas.feedback import FeedbackEventIn, FeedbackEventOut

router = APIRouter()


@router.post("/feedback_events", response_model=FeedbackEventOut)
def create_feedback_event(payload: FeedbackEventIn):
    settings = get_settings()
    init_db(settings.abs_sqlite_path())

    ra = get_risk_assessment(settings.abs_sqlite_path(), payload.risk_assessment_id)
    if not ra:
        raise ApiError(404, "resource_missing", "Risk assessment not found.", param="risk_assessment_id")

    now = int(time.time())
    eid = f"fb_{uuid.uuid4().hex[:24]}"

    insert_feedback_event(
        settings.abs_sqlite_path(),
        {
            "id": eid,
            "created": now,
            "risk_assessment_id": payload.risk_assessment_id,
            "outcome": payload.outcome,
            "notes": payload.notes,
        },
    )

    return FeedbackEventOut(
        id=eid,
        risk_assessment_id=payload.risk_assessment_id,
        outcome=payload.outcome,
        notes=payload.notes,
        created=now,
    )
