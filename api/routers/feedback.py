from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter

from api.services.store import append_feedback_event

router = APIRouter(tags=["feedback"])


@router.post("/feedback/label")
def feedback_label(payload: Dict[str, Any]):
    """
    Example:
    {
      "review_id": "rev_xxx",
      "label": "fraud" | "legit",
      "notes": "optional"
    }
    """
    event = dict(payload)
    event["created"] = int(time.time())
    append_feedback_event(event)
    return {"status": "ok"}
