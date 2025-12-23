from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


DecisionLabel = Literal["approve", "review", "block"]
AEBucket = Literal["approve", "review", "block"]


@dataclass
class DecisionResult:
    label: DecisionLabel
    decided_by: str
    ae_error: Optional[float] = None
    ae_bucket: Optional[AEBucket] = None


def ae_bucket(error: float, t_review: float, t_block: float) -> AEBucket:
    if error >= t_block:
        return "block"
    if error >= t_review:
        return "review"
    return "approve"


def decide(
    p_xgb: float,
    xgb_t_low: float,
    xgb_t_high: float,
    ae_error_val: Optional[float],
    ae_t_review: float,
    ae_t_block: float,
) -> DecisionResult:
    # 1) XGB dominates when confident
    if p_xgb >= xgb_t_high:
        return DecisionResult(label="block", decided_by="xgb_high")

    if p_xgb <= xgb_t_low:
        return DecisionResult(label="approve", decided_by="xgb_low")

    # 2) Gray-zone -> AE triage
    if ae_error_val is None:
        return DecisionResult(label="review", decided_by="gray_no_ae")

    bucket = ae_bucket(ae_error_val, ae_t_review, ae_t_block)

    # Conservative: unknown patterns → manual review (even if AE says block)
    if bucket in ("review", "block"):
        return DecisionResult(
            label="review",
            decided_by="ae_gray",
            ae_error=ae_error_val,
            ae_bucket=bucket,
        )

    return DecisionResult(
        label="approve",
        decided_by="ae_gray",
        ae_error=ae_error_val,
        ae_bucket=bucket,
    )
