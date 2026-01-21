import requests

TIMEOUT = 30

def api_get_review_queue(api_base: str, status: str = "all"):
    """Get review queue from API.

    By default the dashboard requests `status=all` so the client can filter Open/Approved/Blocked locally.
    """
    params = {"status": status}
    r = requests.get(f"{api_base}/review/queue", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["items"]


def api_get_review(api_base: str, review_id: str):
    r = requests.get(f"{api_base}/review/{review_id}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def api_close_review(api_base: str, review_id: str, analyst: str, decision: str, notes: str):
    payload = {
        "analyst": analyst,
        "analyst_decision": decision,
        "notes": notes,
    }
    r = requests.post(
        f"{api_base}/review/{review_id}/close",
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()
