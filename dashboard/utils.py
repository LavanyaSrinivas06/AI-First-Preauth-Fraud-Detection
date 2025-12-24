# dashboard/utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Review log (analyst actions)
# ----------------------------
def load_review_log(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(
            columns=[
                "review_id",
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
    Idempotent append by (review_id) if present, else by (txn_id, timestamp).
    """
    ensure_parent_dir(path)
    df = load_review_log(path)

    record = dict(record)
    record["timestamp"] = str(record.get("timestamp", ""))

    rid = record.get("review_id")
    if rid:
        if not df.empty and "review_id" in df.columns and (df["review_id"] == rid).any():
            return
    else:
        key_txn = record.get("txn_id")
        key_ts = record.get("timestamp")
        if key_txn is None or key_ts is None:
            raise ValueError("append_review_log requires review_id OR (txn_id and timestamp)")
        if not df.empty:
            df["timestamp"] = df["timestamp"].astype(str)
            dup = (df["txn_id"] == key_txn) & (df["timestamp"] == key_ts)
            if dup.any():
                return

    df2 = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df2.to_parquet(path, index=False)


# ----------------------------
# Queue loader (from API jsonl)
# ----------------------------
def load_review_queue_jsonl(path: str, limit: int = 200) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    items = items[::-1]  # newest first
    return items[:limit]


# ----------------------------
# SHAP assets
# ----------------------------
def find_shap_png(static_dir: str, review_id: str) -> Optional[str]:
    p = Path(static_dir) / f"shap_{review_id}.png"
    return str(p) if p.exists() else None


def ensure_sample_shap(static_dir: str) -> Optional[str]:
    p = Path(static_dir) / "shap_example.png"
    return str(p) if p.exists() else None
