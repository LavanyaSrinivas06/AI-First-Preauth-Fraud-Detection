# dashboard/utils_api.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests

API_BASE = "http://127.0.0.1:8000"


def api_get_review_queue() -> list[Dict[str, Any]]:
    r = requests.get(f"{API_BASE}/review/queue", timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("items", [])


def api_get_review(review_id: str) -> Dict[str, Any]:
    r = requests.get(f"{API_BASE}/review/{review_id}", timeout=5)
    r.raise_for_status()
    return r.json()


def api_close_review(review_id: str, analyst: str, decision: str, notes: Optional[str] = None) -> Dict[str, Any]:
    url = f"{API_BASE}/review/{review_id}/close"
    payload = {
        "analyst_decision": decision,
        "analyst": analyst,
        "notes": notes,
    }
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


# -------------------------
# Feedback loop endpoints
# -------------------------

def api_feedback_label(review_id: str, outcome: str, notes: Optional[str] = None) -> Dict[str, Any]:
    """
    POST /feedback/label
    Body: { "review_id": "...", "outcome": "APPROVE"|"BLOCK", "notes": "..." }
    """
    url = f"{API_BASE}/feedback/label"
    payload = {"review_id": review_id, "outcome": outcome, "notes": notes}
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def api_get_feedback_events(limit: int = 200) -> List[Dict[str, Any]]:
    """
    GET /feedback/events?limit=200
    """
    r = requests.get(f"{API_BASE}/feedback/events", params={"limit": int(limit)}, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("items", [])


def api_get_feedback_summary() -> Dict[str, Any]:
    """
    GET /feedback/summary
    """
    r = requests.get(f"{API_BASE}/feedback/summary", timeout=10)
    r.raise_for_status()
    return r.json()


def api_export_feedback(limit: int = 1000) -> Dict[str, Any]:
    """
    GET /feedback/export?limit=1000
    Returns: { items: [...], count: N }
    """
    r = requests.get(f"{API_BASE}/feedback/export", params={"limit": int(limit)}, timeout=20)
    r.raise_for_status()
    return r.json()

def api_get_closed_reviews(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Uses feedback export as the source of truth for labeled/closed reviews.
    Returns items where analyst_decision exists (closed reviews).
    """
    data = api_export_feedback(limit=limit)
    items = data.get("items", [])
    # extra safety filter
    out = []
    for r in items:
        if r.get("analyst_decision") in ("APPROVE", "BLOCK"):
            out.append(r)
    return out


# -------------------------
# Backwards-compatible aliases
# -------------------------

def api_get_feedback_export(limit: int = 1000) -> Dict[str, Any]:
    return api_export_feedback(limit=limit)


def api_get_feedback_events_list(limit: int = 200) -> List[Dict[str, Any]]:
    return api_get_feedback_events(limit=limit)


def api_get_feedback_summary_obj() -> Dict[str, Any]:
    return api_get_feedback_summary()
