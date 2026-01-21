# api/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import get_settings
from api.core.logging import setup_logging
from api.core.errors import register_exception_handlers

from api.routers.health import router as health_router
from api.routers.preauth import router as preauth_router
from api.routers.review import router as review_router
from api.routers.feedback import router as feedback_router


def create_app() -> FastAPI:
    settings = get_settings()

    setup_logging(settings)

    app = FastAPI(
        title="AI-First Preauth Fraud Detection API",
        version="1.0.0",
    )

    # Allow cross-origin requests from local dashboard/dev hosts. In dev it's
    # convenient to allow the Streamlit dashboard (localhost:8501) and the
    # local API swagger UI. For production pin this to the real dashboard URL.
    allowed_origins = [
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(preauth_router)
    app.include_router(review_router)
    app.include_router(feedback_router)

    return app


app = create_app()
