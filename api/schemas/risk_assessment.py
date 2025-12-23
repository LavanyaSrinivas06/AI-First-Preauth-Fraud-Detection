from __future__ import annotations

from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class RiskAssessmentRequest(BaseModel):
    # optional wrapper like Stripe: { "data": {...} }
    data: Optional[Dict[str, Any]] = None

    # allow raw body payload too (features directly at top-level)
    @model_validator(mode="before")
    @classmethod
    def accept_raw_or_wrapped(cls, v):
        if isinstance(v, dict) and "data" in v and isinstance(v["data"], dict):
            return v
        if isinstance(v, dict):
            return {"data": v}
        return v


class ThresholdsOut(BaseModel):
    xgb_t_low: float
    xgb_t_high: float
    ae_review: float
    ae_block: float


class RiskAssessmentResponse(BaseModel):
    id: str
    object: Literal["risk_assessment"] = "risk_assessment"

    label: Literal["approve", "review", "block"]
    decided_by: str

    score_xgb: float
    ae_error: Optional[float] = None
    ae_bucket: Optional[Literal["approve", "review", "block"]] = None

    thresholds: ThresholdsOut
    reason_codes: list[str] = Field(default_factory=list)
    latency_ms: float
    created: int  # unix seconds
