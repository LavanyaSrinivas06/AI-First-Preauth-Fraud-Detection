from __future__ import annotations

from fastapi import APIRouter
from api.core.config import get_settings
from api.services.model_service import ensure_loaded
from api.services.store import init_db

router = APIRouter()


@router.get("/health")
def health():
    settings = get_settings()
    init_db(settings.abs_sqlite_path())
    _ = ensure_loaded(settings)
    return {"status": "ok", "models_loaded": True}
