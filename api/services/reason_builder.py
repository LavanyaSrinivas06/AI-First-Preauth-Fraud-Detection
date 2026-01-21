from typing import Any, Dict, List, Optional


def build_reason_details_v2(
    *,
    decision: str,
    p_xgb: float,
    t_low: float,
    t_high: float,
    features: Dict[str, Any],
    ae_percentile: Optional[float],
    ae_bucket: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Build human-readable reason details explaining WHY a transaction
    was BLOCKED or sent to REVIEW, using interpretable risk signals
    from both XGBoost and Autoencoder.
    """

    reasons: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Decision-level explanation (policy)
    # ------------------------------------------------------------------
    if decision == "BLOCK":
        reasons.append(
            {
                "code": "HIGH_RISK_SCORE",
                "message": (
                    "The predicted fraud probability exceeded the high-risk threshold, "
                    "indicating strong confidence of fraudulent behavior."
                ),
            }
        )
    elif decision == "REVIEW":
        reasons.append(
            {
                "code": "UNCERTAIN_RISK_SCORE",
                "message": (
                    "The predicted fraud probability falls within the uncertainty range "
                    "and cannot be confidently approved or blocked."
                ),
            }
        )

    # ------------------------------------------------------------------
    # 2. XGBoost contributing signals (interpretable business signals)
    # ------------------------------------------------------------------
    xgb_signals: List[str] = []

    if features.get("cat__country_mismatch_True", 0) == 1:
        xgb_signals.append("Country mismatch between billing and IP location")

    if features.get("cat__is_new_device_True", 0) == 1:
        xgb_signals.append("New or previously unseen device")

    if features.get("cat__is_proxy_vpn_True", 0) == 1:
        xgb_signals.append("VPN or proxy usage detected")

    if features.get("num__geo_distance_km") is not None and features.get("num__geo_distance_km", 0) > 1.0:
        xgb_signals.append("Unusually large geographic distance from typical activity")

    if (
        features.get("num__txn_count_60m", 0) > 1.0
        or features.get("num__txn_count_30m", 0) > 1.0
    ):
        xgb_signals.append("Elevated transaction velocity within a short time window")

    if (
        features.get("num__amount_zscore") is not None
        and abs(features.get("num__amount_zscore", 0)) > 1.5
    ):
        xgb_signals.append("Transaction amount deviates from the user’s historical spending pattern")

    # Latent embedding deviation (V-features)
    v_deviation = any(
        abs(features.get(f"num__V{i}", 0)) > 2.5 for i in (7, 10, 14)
    )
    if v_deviation:
        xgb_signals.append(
            "Behavioral embedding features indicate unusual transaction patterns"
        )

    if xgb_signals:
        reasons.append(
            {
                "code": "XGB_CONTRIBUTING_SIGNALS",
                "message": (
                    "The supervised risk model identified the following contributing risk signals:"
                ),
                "signals": xgb_signals,
            }
        )

    # ------------------------------------------------------------------
    # 3. Autoencoder contributing signals (anomaly perspective)
    # ------------------------------------------------------------------
    ae_signals: List[str] = []

    if ae_percentile is not None and ae_percentile > 90:
        ae_signals.append(
            "Transaction behavior is unusual compared to the majority of legitimate transactions"
        )

    if ae_bucket == "extreme":
        ae_signals.append(
            "Behavior is strongly inconsistent with normal legitimate transaction patterns"
        )

    if ae_signals:
        reasons.append(
            {
                "code": "AE_CONTRIBUTING_SIGNALS",
                "message": (
                    "The anomaly detection model observed deviations from typical legitimate behavior:"
                ),
                "signals": ae_signals,
            }
        )

    # ------------------------------------------------------------------
    # 4. Human-in-the-loop explanation (REVIEW only)
    # ------------------------------------------------------------------
    if decision == "REVIEW":
        reasons.append(
            {
                "code": "HUMAN_REVIEW_RECOMMENDED",
                "message": (
                    "The detected risk signals are inconclusive. Manual review is recommended "
                    "to confirm transaction legitimacy and reduce false positives."
                ),
            }
        )

    return reasons
