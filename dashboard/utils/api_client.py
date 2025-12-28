from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests


def api_get_review_queue(api_base: str) -> List[Dict[str, Any]]:
    r = requests.get(f"{api_base}/review/queue", timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("items", [])


def api_get_review(api_base: str, review_id: str) -> Dict[str, Any]:
    r = requests.get(f"{api_base}/review/{review_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_close_review(api_base: str, review_id: str, analyst: str, decision: str, notes: Optional[str] = None) -> Dict[str, Any]:
    url = f"{api_base}/review/{review_id}/close"
    payload = {"analyst_decision": decision, "analyst": analyst, "notes": notes}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def api_get_feedback_export(api_base: str, limit: int = 1000) -> Dict[str, Any]:
    r = requests.get(f"{api_base}/feedback/export", params={"limit": int(limit)}, timeout=20)
    r.raise_for_status()
    return r.json()
