#api/core/logging.py
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from api.core.config import Settings


def setup_logging(settings: Settings) -> logging.Logger:
    settings.logs_path().mkdir(parents=True, exist_ok=True)
    Path(settings.abs_log_path()).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("fpn_api")
    logger.setLevel(logging.INFO)

    # avoid duplicate handlers on reload
    if not logger.handlers:
        fh = logging.FileHandler(settings.abs_log_path(), encoding="utf-8")
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    return logger


def json_log(logger: logging.Logger, record: dict):
    logger.info(json.dumps(record, ensure_ascii=False))


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Latency-Ms"] = f"{latency_ms:.3f}"

        # attach for downstream usage if needed
        request.state.request_id = request_id
        request.state.latency_ms = latency_ms

        return response
