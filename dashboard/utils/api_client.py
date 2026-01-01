from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


def api_get_review_queue(api_base: str) -> List[Dict[str, Any]]:
    r = requests.get(f"{api_base}/review/queue", timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("items", []) if isinstance(data, dict) else []


def api_get_review(api_base: str, review_id: str) -> Dict[str, Any]:
    r = requests.get(f"{api_base}/review/{review_id}", timeout=15)
    r.raise_for_status()
    return r.json()


def api_close_review(
    api_base: str,
    review_id: str,
    analyst: str,
    decision: str,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{api_base}/review/{review_id}/close"
    payload = {"analyst_decision": decision, "analyst": analyst, "notes": notes}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def api_assign_review(api_base: str, review_id: str, analyst: str) -> Dict[str, Any]:
    """
    Requires API endpoint: POST /review/{review_id}/assign
    If you don't have it yet, you'll see a clear error in the UI.
    """
    url = f"{api_base}/review/{review_id}/assign"
    payload = {"analyst": analyst}
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def api_get_feedback_export(api_base: str, limit: int = 1000) -> Dict[str, Any]:
    r = requests.get(f"{api_base}/feedback/export", params={"limit": int(limit)}, timeout=25)
    r.raise_for_status()
    return r.json()


def api_get_feedback_summary(api_base: str) -> Dict[str, Any]:
    """
    Requires API endpoint: GET /feedback/summary
    If you don't have it yet, UI will just skip this card.
    """
    r = requests.get(f"{api_base}/feedback/summary", timeout=15)
    r.raise_for_status()
    return r.json()


def api_get_health_model(api_base: str) -> Dict[str, Any]:
    r = requests.get(f"{api_base}/health/model", timeout=10)
    r.raise_for_status()
    return r.json()
