from fastapi import FastAPI
from api.core.config import get_settings
from api.core.logging import setup_logging, RequestIdMiddleware
from api.core.errors import register_exception_handlers

from api.routers.health import router as health_router
from api.routers.risk_assessments import router as risk_router
from api.routers.reviews import router as reviews_router
from api.routers.feedback_events import router as feedback_router

settings = get_settings()
logger = setup_logging(settings)

app = FastAPI(
    title="AI-First Preauth Fraud API",
    version="1.0.0",
)

# middleware
app.add_middleware(RequestIdMiddleware)

# exception handlers (Stripe-like errors)
register_exception_handlers(app)

# routers
app.include_router(health_router, tags=["health"])
app.include_router(risk_router, prefix="/v1", tags=["risk_assessments"])
app.include_router(reviews_router, prefix="/v1", tags=["reviews"])
app.include_router(feedback_router, prefix="/v1", tags=["feedback_events"])
