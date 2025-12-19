from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, constr

ShortStr = constr(min_length=1, max_length=50, strip_whitespace=True)
CountryCode = constr(min_length=2, max_length=2, pattern=r"^[A-Za-z]{2}$")
ReasonCode = constr(min_length=1, max_length=50, strip_whitespace=True)


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., gt=0)
    ip_reputation: float = Field(..., ge=0, le=1)
    velocity_1h: float = Field(..., ge=0)
    velocity_24h: float = Field(..., ge=0)

    device_os: ShortStr
    browser: ShortStr
    ip_country: CountryCode
    billing_country: CountryCode
    shipping_country: CountryCode

    hour_of_day: Optional[int] = Field(None, ge=0, le=23)

    @field_validator("device_os", "browser", mode="before")
    def _strip_string(cls, v: str) -> str:
        return v.strip()

    @field_validator("ip_country", "billing_country", "shipping_country", mode="before")
    def _normalize_country(cls, v: str) -> str:
        return v.strip().upper()


class PredictionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["approve", "review", "block"]
    score_xgb: Optional[float] = Field(None, ge=0, le=1)
    score_ae: Optional[float] = Field(None, ge=0, le=1)
    ensemble_score: Optional[float] = Field(None, ge=0, le=1)
    reason_codes: List[ReasonCode] = Field(default_factory=list)

    @field_validator("reason_codes", mode="before")
    def _normalize_reason_codes(cls, v):
        if v is None:
            return []
        return v


__all__ = ["TransactionIn", "PredictionOut"]
