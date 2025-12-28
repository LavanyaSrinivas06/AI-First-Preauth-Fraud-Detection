# api/core/errors.py
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, param: str | None = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.param = param


def stripe_error(code: str, message: str, param: str | None = None, type_: str = "invalid_request_error"):
    body = {
        "error": {
            "type": type_,
            "code": code,
            "message": message,
        }
    }
    if param:
        body["error"]["param"] = param
    return body


def register_exception_handlers(app: FastAPI):
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status_code,
            content=stripe_error(exc.code, exc.message, exc.param),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        # keep it safe; do not leak internals in API response
        return JSONResponse(
            status_code=500,
            content=stripe_error("internal_error", f"{type(exc).__name__}: {exc}" ),
        )
