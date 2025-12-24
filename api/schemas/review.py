#api/schemas/review.py
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ReviewItem(BaseModel):
    id: str
    object: Literal["review"] = "review"

    risk_assessment_id: str
    status: Literal["open", "resolved"] = "open"
    analyst_decision: Optional[Literal["fraud", "legit"]] = None
    notes: Optional[str] = None

    created: int
    updated: int


class ReviewUpdateRequest(BaseModel):
    status: Optional[Literal["open", "resolved"]] = None
    analyst_decision: Optional[Literal["fraud", "legit"]] = None
    notes: Optional[str] = None
