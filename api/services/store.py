# api/services/store.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Simple file-based stores (thesis-friendly)
REVIEW_QUEUE_PATH = Path("artifacts/review_queue.jsonl")
FEEDBACK_LOG_PATH = Path("artifacts/feedback_log.jsonl")


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    # newest first
    items = items[::-1]
    return items[:limit]


def save_review_if_needed(
    decision: str,
    payload: Dict[str, Any],
    p_xgb: Optional[float],
    ae_err: Optional[float],
    payload_hash: str,
    reason_codes: List[str],
) -> None:
    """
    Persist only REVIEW events for the dashboard queue.
    """
    if decision != "REVIEW":
        return

    now = int(time.time())

    item = {
        "id": f"rev_{payload_hash[:16]}",
        "created": now,
        "txn_id": payload.get("txn_id"),
        "timestamp": payload.get("timestamp"),
        "decision": decision,
        "score_xgb": p_xgb,
        "ae_error": ae_err,
        "payload_hash": payload_hash,
        "reason_codes": reason_codes,
        # keep payload minimal (avoid PII); include only what dashboard needs
        "amount": payload.get("amount"),
        "country": payload.get("country"),
        "ip_country": payload.get("ip_country"),
        "currency": payload.get("currency"),
        "card_currency": payload.get("card_currency"),
        "hour": payload.get("hour"),
        "velocity_1h": payload.get("velocity_1h"),
        "velocity_24h": payload.get("velocity_24h"),
        "is_new_device": payload.get("is_new_device"),
        "is_proxy_vpn": payload.get("is_proxy_vpn"),
    }

    append_jsonl(REVIEW_QUEUE_PATH, item)


def load_review_queue(limit: int = 200) -> List[Dict[str, Any]]:
    return load_jsonl(REVIEW_QUEUE_PATH, limit=limit)


def append_feedback_event(event: Dict[str, Any]) -> None:
    append_jsonl(FEEDBACK_LOG_PATH, event)
