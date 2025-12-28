# api/schemas/feedback.py
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


class FeedbackLabelIn(BaseModel):
    """
    Input schema for POST /feedback/label
    """
    review_id: str
    outcome: Literal["APPROVE", "BLOCK"]
    notes: Optional[str] = None


class FeedbackEvent(BaseModel):
    """
    Stored feedback event (from feedback_events table)
    """
    id: str
    review_id: str
    outcome: Literal["APPROVE", "BLOCK"]
    notes: Optional[str] = None
    created: int
