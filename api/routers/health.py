# api/routers/health.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends

from api.core.config import Settings, get_settings
from api.services.model_service import ensure_loaded

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@router.get("/health/model")
def health_model(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """
    Dashboard-friendly info about the active XGB model + promotion metadata.
    """
    art = ensure_loaded(settings)
    reg_path = settings.artifacts_path() / "models" / "active_xgb.json"

    reg: Dict[str, Any] = {}
    if reg_path.exists():
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
        except Exception:
            reg = {"error": "invalid active_xgb.json"}

    return {
        "active_model_version": getattr(art, "model_version", "v1"),
        "registry_path": str(reg_path),
        "registry": reg,
    }
