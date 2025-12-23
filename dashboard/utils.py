# dashboard/utils.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def safe_bool(x: Any) -> Optional[bool]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n"}:
            return False
    return None


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_review_log(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(
            columns=[
                "txn_id",
                "timestamp",
                "ensemble_score",
                "analyst_decision",
                "analyst",
                "notes",
                "decision_time",
            ]
        )
    return pd.read_parquet(p)


def append_review_log(path: str, record: Dict[str, Any]) -> None:
    """
    Idempotent append:
    - prevents duplicates by (txn_id, timestamp)
    """
    ensure_parent_dir(path)
    df = load_review_log(path)

    key_txn = record.get("txn_id")
    key_ts = record.get("timestamp")
    if key_txn is None or key_ts is None:
        raise ValueError("append_review_log requires txn_id and timestamp")

    # normalize timestamp to string for consistent keying
    record = dict(record)
    record["timestamp"] = str(record["timestamp"])

    if not df.empty:
        df["timestamp"] = df["timestamp"].astype(str)
        dup = (df["txn_id"] == key_txn) & (df["timestamp"] == record["timestamp"])
        if dup.any():
            # already exists => do nothing (idempotent)
            return

    df2 = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df2.to_parquet(path, index=False)


def _col(cfg: Dict[str, Any], group: str, name: str, default: str) -> str:
    return cfg.get(group, {}).get(name, default)


def pick_first_existing(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c and c in df.columns:
            return c
    return None


def compute_amount_threshold(df: pd.DataFrame, amount_col: str, q: float) -> Optional[float]:
    if amount_col not in df.columns:
        return None
    s = pd.to_numeric(df[amount_col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.quantile(q))


def generate_reasons_for_row(
    row: pd.Series,
    cfg: Dict[str, Any],
    amount_threshold: Optional[float] = None,
) -> List[str]:
    """
    Rule-based "why flagged" reasons.
    Only uses columns if present; otherwise skips.
    """
    cols = cfg.get("columns", {})
    expl = cfg.get("explainability", {})
    reasons: List[str] = []

    # geo mismatch signals
    billing_country = cols.get("billing_country", "billing_country")
    ip_country = cols.get("ip_country", "ip_country")
    merchant_country = cols.get("merchant_country", "merchant_country")
    country = cols.get("country", "country")

    bc = row.get(billing_country) if billing_country in row.index else None
    ic = row.get(ip_country) if ip_country in row.index else None
    mc = row.get(merchant_country) if merchant_country in row.index else None
    cc = row.get(country) if country in row.index else None

    if bc is not None and ic is not None and str(bc) and str(ic) and str(bc) != str(ic):
        reasons.append(f"Geo mismatch: billing_country={bc} vs ip_country={ic}")
    if bc is not None and mc is not None and str(bc) and str(mc) and str(bc) != str(mc):
        reasons.append(f"Geo mismatch: billing_country={bc} vs merchant_country={mc}")
    if cc is not None and ic is not None and str(cc) and str(ic) and str(cc) != str(ic):
        reasons.append(f"Geo mismatch: country={cc} vs ip_country={ic}")

    # currency mismatch
    currency = cols.get("currency", "currency")
    card_currency = cols.get("card_currency", "card_currency")
    cur = row.get(currency) if currency in row.index else None
    ccur = row.get(card_currency) if card_currency in row.index else None
    if cur is not None and ccur is not None and str(cur) and str(ccur) and str(cur) != str(ccur):
        reasons.append(f"Currency mismatch: currency={cur} vs card_currency={ccur}")

    # night transaction
    hour_col = cols.get("hour", "hour")
    is_night_col = cols.get("is_night", "is_night")
    night_hours = set(expl.get("night_hours", [0, 1, 2, 3, 4, 5]))

    is_night = None
    if is_night_col in row.index:
        is_night = safe_bool(row.get(is_night_col))

    hr = None
    if hour_col in row.index:
        try:
            hr = int(float(row.get(hour_col)))
        except Exception:
            hr = None

    if is_night is True or (hr is not None and hr in night_hours):
        reasons.append("Unusual time: night-time transaction")

    # high amount
    amount_col = cols.get("amount", "amount")
    if amount_threshold is not None and amount_col in row.index:
        amt = pd.to_numeric(pd.Series([row.get(amount_col)]), errors="coerce").iloc[0]
        if pd.notna(amt) and float(amt) >= float(amount_threshold):
            reasons.append(f"High amount: {float(amt):.2f} >= p{int(100*expl.get('high_amount_quantile', 0.95))} threshold")

    # velocity
    v1h_col = cols.get("velocity_1h", "velocity_1h")
    v24h_col = cols.get("velocity_24h", "velocity_24h")
    v1h_thr = expl.get("velocity_1h_threshold", 6)
    v24h_thr = expl.get("velocity_24h_threshold", 20)

    if v1h_col in row.index:
        v = pd.to_numeric(pd.Series([row.get(v1h_col)]), errors="coerce").iloc[0]
        if pd.notna(v) and float(v) >= float(v1h_thr):
            reasons.append(f"High velocity: {int(v)} txns in last 1h")

    if v24h_col in row.index:
        v = pd.to_numeric(pd.Series([row.get(v24h_col)]), errors="coerce").iloc[0]
        if pd.notna(v) and float(v) >= float(v24h_thr):
            reasons.append(f"High velocity: {int(v)} txns in last 24h")

    # new device / new email
    nd_col = cols.get("is_new_device", "is_new_device")
    ne_col = cols.get("is_new_email", "is_new_email")
    if nd_col in row.index and safe_bool(row.get(nd_col)) is True:
        reasons.append("Behavior: new device")
    if ne_col in row.index and safe_bool(row.get(ne_col)) is True:
        reasons.append("Behavior: new email")

    # risky country (any available country signal)
    risk = set(map(str, expl.get("risk_countries", [])))
    for cval in [bc, ic, mc, cc]:
        if cval is not None and str(cval) in risk:
            reasons.append(f"Context: risky country signal ({cval})")
            break

    # cap reasons
    max_r = int(expl.get("max_reasons", 6))
    return reasons[:max_r]


def find_shap_png(static_dir: str, txn_id: str) -> Optional[str]:
    p = Path(static_dir) / f"shap_{txn_id}.png"
    return str(p) if p.exists() else None


def find_lime_html(static_dir: str, txn_id: str) -> Optional[str]:
    p = Path(static_dir) / f"lime_{txn_id}.html"
    return str(p) if p.exists() else None


def ensure_sample_explainability_assets(static_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    If no txn-specific assets exist, try to show generic examples:
      dashboard/static/shap_example.png
      dashboard/static/lime_example.html
    """
    sdir = Path(static_dir)
    shp = sdir / "shap_example.png"
    lime = sdir / "lime_example.html"
    return (str(shp) if shp.exists() else None, str(lime) if lime.exists() else None)
