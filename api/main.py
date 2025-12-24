from fastapi import FastAPI

from api.core.config import get_settings
from api.core.errors import register_exception_handlers
from api.core.logging import RequestIdMiddleware, setup_logging

from api.routers.health import router as health_router
from api.routers.preauth import router as preauth_router
from api.routers.review import router as review_router
from api.routers.feedback import router as feedback_router

settings = get_settings()
_ = setup_logging(settings)

app = FastAPI(
    title="AI-First Preauth Fraud Detection API",
    version="1.0.0",
)

app.add_middleware(RequestIdMiddleware)
register_exception_handlers(app)

app.include_router(health_router)
app.include_router(preauth_router)
app.include_router(review_router)
app.include_router(feedback_router)
