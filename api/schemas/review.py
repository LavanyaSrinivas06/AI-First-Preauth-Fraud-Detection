# api/schemas/review.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel


class ReviewCloseIn(BaseModel):
    analyst_decision: Literal["APPROVE", "BLOCK"]
    analyst: str
    notes: Optional[str] = None


class ReviewItem(BaseModel):
    id: str
    object: Literal["review"] = "review"

    status: Literal["open", "closed"] = "open"
    decision: Literal["REVIEW"] = "REVIEW"

    # model signals (what drove the review)
    score_xgb: Optional[float] = None
    ae_error: Optional[float] = None
    ae_percentile_vs_legit: Optional[float] = None
    ae_bucket: Optional[str] = None
    reason_codes: List[str] = []

    # safe snapshot for UI
    payload_min: Dict[str, Any] = {}

    # analyst closure fields
    analyst_decision: Optional[Literal["APPROVE", "BLOCK"]] = None
    analyst: Optional[str] = None
    notes: Optional[str] = None

    created: int
    updated: int
