from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


# These are the checkout-like fields we require (small + realistic).
REQUIRED_CHECKOUT_FIELDS = [
    "txn_id",
    "timestamp",
    "amount",
    "country",
    "ip_country",
    "currency",
    "card_currency",
    "hour",
    "velocity_1h",
    "velocity_24h",
    "is_new_device",
    "is_proxy_vpn",
]


def validate_checkout_payload(payload: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_CHECKOUT_FIELDS if k not in payload]
    if missing:
        # keep message short like Stripe
        raise ValueError(f"Missing required checkout fields: {missing}")


def _set_onehot(row: Dict[str, Any], feature: str, value: bool) -> None:
    # expects features like cat__is_proxy_vpn_True / cat__is_proxy_vpn_False
    row[feature] = bool(value)


def adapt_payload_to_processed_102(
    payload: Dict[str, Any],
    model_features: List[str],
) -> pd.DataFrame:
    """
    Builds a DataFrame with EXACT columns = model_features (102),
    defaulting missing numeric to 0.0 and one-hot bool/cat to a safe baseline.
    """

    validate_checkout_payload(payload)

    row: Dict[str, Any] = {}

    # --- First: defaults for all 102 ---
    for f in model_features:
        if f.startswith("num__"):
            row[f] = 0.0
        elif f.endswith("_True"):
            row[f] = False
        elif f.endswith("_False"):
            row[f] = True
        else:
            # safe fallback
            row[f] = 0.0

    # --- Map numeric fields we DO have ---
    def set_num(name: str, val: Any) -> None:
        if name in model_features:
            try:
                row[name] = float(val)
            except Exception:
                row[name] = 0.0

    set_num("num__Amount", payload.get("amount"))
    set_num("num__txn_count_60m", payload.get("velocity_1h"))
    set_num("num__txn_count_30m", payload.get("velocity_1h"))
    set_num("num__txn_count_5m", max(0, int(payload.get("velocity_1h", 0)) // 3))
    set_num("num__txn_count_60m", payload.get("velocity_1h"))
    set_num("num__txn_count_30m", payload.get("velocity_1h"))
    set_num("num__txn_count_5m", max(0, int(payload.get("velocity_1h", 0)) // 3))

    # If your processed schema includes these (it does):
    set_num("num__ip_reputation", 0.0)  # unknown at checkout → neutral
    set_num("num__geo_distance_km", 0.0)  # unknown → neutral
    set_num("num__account_age_days", 0.0)
    set_num("num__token_age_days", 0.0)

    # --- One-hot device / VPN ---
    if "cat__is_new_device_True" in model_features:
        _set_onehot(row, "cat__is_new_device_True", bool(payload.get("is_new_device")))
    if "cat__is_new_device_False" in model_features:
        _set_onehot(row, "cat__is_new_device_False", not bool(payload.get("is_new_device")))

    if "cat__is_proxy_vpn_True" in model_features:
        _set_onehot(row, "cat__is_proxy_vpn_True", bool(payload.get("is_proxy_vpn")))
    if "cat__is_proxy_vpn_False" in model_features:
        _set_onehot(row, "cat__is_proxy_vpn_False", not bool(payload.get("is_proxy_vpn")))

    # --- One-hot countries (ip + billing + shipping) ---
    ip = str(payload.get("ip_country"))
    bill = str(payload.get("country"))
    ship = str(payload.get("country"))  # at checkout you may not have shipping → use country as proxy

    ip_key = f"cat__ip_country_{ip}"
    if ip_key in model_features:
        row[ip_key] = True

    bill_key = f"cat__billing_country_{bill}"
    if bill_key in model_features:
        row[bill_key] = True

    ship_key = f"cat__shipping_country_{ship}"
    if ship_key in model_features:
        row[ship_key] = True

    # --- Derived booleans present in your 102 schema ---
    country_mismatch = (ip != bill)
    if "cat__country_mismatch_True" in model_features:
        row["cat__country_mismatch_True"] = country_mismatch
    if "cat__country_mismatch_False" in model_features:
        row["cat__country_mismatch_False"] = not country_mismatch

    hour = int(payload.get("hour", 12))
    night = hour in {0, 1, 2, 3, 4, 5}
    if "cat__night_txn_True" in model_features:
        row["cat__night_txn_True"] = night
    if "cat__night_txn_False" in model_features:
        row["cat__night_txn_False"] = not night

    # weekend unknown from payload timestamp unless parsed; keep safe default False (already)

    return pd.DataFrame([row], columns=model_features)
