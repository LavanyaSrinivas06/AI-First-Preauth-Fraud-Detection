from fastapi import APIRouter

from api.core.config import get_settings
from api.services.model_service import ensure_loaded

router = APIRouter(tags=["health"])


@router.get("/health/api")
def health_api():
    return {"status": "ok"}


@router.get("/health/model")
def health_model():
    settings = get_settings()
    _ = ensure_loaded(settings)
    return {"status": "ok", "models_loaded": True}
