# api/schemas.py
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic.types import StrictBool, StrictFloat, StrictStr, constr

ShortStr = constr(min_length=1, max_length=50, strip_whitespace=True)
CountryCode = constr(min_length=2, max_length=2, pattern=r"^[A-Za-z]{2}$")
DeviceId = constr(min_length=1, max_length=64, strip_whitespace=True)
ReasonCode = constr(min_length=1, max_length=50, strip_whitespace=True)


class TransactionIn(BaseModel):
    """
    Full raw-input schema matching artifacts/features.json:
    categorical_features + numerical_features.
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        anystr_strip_whitespace=True,
        populate_by_name=True,   # allows alias input
    )

    # ---------- categorical_features ----------
    device_id: DeviceId
    device_os: ShortStr
    browser: ShortStr
    is_new_device: StrictBool
    ip_country: CountryCode
    is_proxy_vpn: StrictBool
    billing_country: CountryCode
    shipping_country: CountryCode
    night_txn: StrictBool
    weekend_txn: StrictBool

    # ---------- numerical_features ----------
    V1: StrictFloat
    V2: StrictFloat
    V3: StrictFloat
    V4: StrictFloat
    V5: StrictFloat
    V6: StrictFloat
    V7: StrictFloat
    V8: StrictFloat
    V9: StrictFloat
    V10: StrictFloat
    V11: StrictFloat
    V12: StrictFloat
    V13: StrictFloat
    V14: StrictFloat
    V15: StrictFloat
    V16: StrictFloat
    V17: StrictFloat
    V18: StrictFloat
    V19: StrictFloat
    V20: StrictFloat
    V21: StrictFloat
    V22: StrictFloat
    V23: StrictFloat
    V24: StrictFloat
    V25: StrictFloat
    V26: StrictFloat
    V27: StrictFloat
    V28: StrictFloat

    # Pipeline expects "Amount" (capital A). Accept client sending "amount".
    Amount: StrictFloat = Field(..., gt=0, alias="amount")
    ip_reputation: StrictFloat = Field(..., ge=0, le=1)

    txn_count_5m: StrictFloat = Field(..., ge=0)
    txn_count_30m: StrictFloat = Field(..., ge=0)
    txn_count_60m: StrictFloat = Field(..., ge=0)

    avg_amount_7d: StrictFloat = Field(..., ge=0)
    account_age_days: StrictFloat = Field(..., ge=0)
    token_age_days: StrictFloat = Field(..., ge=0)
    avg_spend_user_30d: StrictFloat = Field(..., ge=0)
    geo_distance_km: StrictFloat = Field(..., ge=0)
    amount_zscore: StrictFloat

    # ---------- normalization ----------
    @field_validator("ip_country", "billing_country", "shipping_country", mode="before")
    def _normalize_country(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("device_os", "browser", "device_id", mode="before")
    def _strip_strings(cls, v: str) -> str:
        return v.strip()


class PredictionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["approve", "review", "block"]
    score_xgb: float | None = Field(None, ge=0, le=1)
    score_ae: float | None = Field(None, ge=0, le=1)
    ensemble_score: float | None = Field(None, ge=0, le=1)
    reason_codes: List[ReasonCode] = Field(default_factory=list)


__all__ = ["TransactionIn", "PredictionOut"]
