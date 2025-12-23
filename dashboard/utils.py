# dashboard/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
    ensure_parent_dir(path)
    df = load_review_log(path)

    key_txn = record.get("txn_id")
    key_ts = record.get("timestamp")
    if key_txn is None or key_ts is None:
        raise ValueError("append_review_log requires txn_id and timestamp")

    record = dict(record)
    record["timestamp"] = str(record["timestamp"])

    if not df.empty:
        df["timestamp"] = df["timestamp"].astype(str)
        dup = (df["txn_id"] == key_txn) & (df["timestamp"] == record["timestamp"])
        if dup.any():
            return

    pd.concat([df, pd.DataFrame([record])], ignore_index=True).to_parquet(path, index=False)


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
    cols = cfg.get("columns", {})
    expl = cfg.get("explainability", {})
    reasons: List[str] = []

    bc = row.get(cols.get("billing_country", "billing_country"))
    ic = row.get(cols.get("ip_country", "ip_country"))
    if bc and ic and str(bc) != str(ic):
        reasons.append(f"Geo mismatch: billing_country={bc} vs ip_country={ic}")

    cur = row.get(cols.get("currency", "currency"))
    ccur = row.get(cols.get("card_currency", "card_currency"))
    if cur and ccur and str(cur) != str(ccur):
        reasons.append(f"Currency mismatch: currency={cur} vs card_currency={ccur}")

    hr = row.get(cols.get("hour", "hour"))
    night_hours = set(expl.get("night_hours", []))
    if hr in night_hours:
        reasons.append("Unusual time: night-time transaction")

    if amount_threshold is not None:
        amt = row.get(cols.get("amount", "amount"))
        if pd.notna(amt) and float(amt) >= amount_threshold:
            reasons.append("High amount relative to recent transactions")

    if safe_bool(row.get(cols.get("is_new_device", "is_new_device"))):
        reasons.append("Behavior: new device")

    return reasons[: expl.get("max_reasons", 6)]


def find_shap_png(static_dir: str, txn_id: str) -> Optional[str]:
    p = Path(static_dir) / f"shap_{txn_id}.png"
    return str(p) if p.exists() else None


def find_fallback_shap(static_dir: str) -> Optional[str]:
    p = Path(static_dir) / "shap_example.png"
    return str(p) if p.exists() else None
