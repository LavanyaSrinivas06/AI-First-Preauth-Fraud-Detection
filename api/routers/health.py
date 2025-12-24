# api/routers/health.py
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.core.config import Settings, get_settings
from api.services.store import init_db

router = APIRouter(tags=["health"])


@router.get("/health/api")
def health_api(settings: Settings = Depends(get_settings)):
    init_db(settings.abs_sqlite_path())
    return {"status": "ok"}


@router.get("/health/model")
def health_model():
    # keep simple; if model loads at first request, no need to force-load here
    return {"status": "ok"}
