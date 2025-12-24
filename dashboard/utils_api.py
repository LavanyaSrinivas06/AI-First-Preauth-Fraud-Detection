# dashboard/utils_api.py
from __future__ import annotations
from typing import Any, Dict,List, Optional
import requests

API_BASE = "http://127.0.0.1:8000"


def api_get_review_queue() -> list[Dict[str, Any]]:
    r = requests.get(f"{API_BASE}/review/queue", timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("items",[])


def api_get_review(review_id: str)-> Dict[str, Any]:
    r = requests.get(f"{API_BASE}/review/{review_id}", timeout=5)
    r.raise_for_status()
    return r.json()


def api_close_review(review_id: str, analyst: str, decision: str, notes: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "analyst_decision": decision,
        "analyst": analyst,
        "notes": notes,
    }
    r = requests.post(f"{API_BASE}/review/{review_id}/close", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()
