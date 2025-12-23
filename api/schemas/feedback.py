from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


class FeedbackEventIn(BaseModel):
    risk_assessment_id: str
    outcome: Literal["fraud", "legit"]
    notes: Optional[str] = None


class FeedbackEventOut(BaseModel):
    id: str
    object: Literal["feedback_event"] = "feedback_event"
    risk_assessment_id: str
    outcome: Literal["fraud", "legit"]
    notes: Optional[str] = None
    created: int
