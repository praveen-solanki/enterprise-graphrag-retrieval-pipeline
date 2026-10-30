"""Exception → HTTP mapping. Callers always get `{code, message, request_id}`;
internals never leak into responses."""
from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_graph_retrieval.core.exceptions import GraphQueryError

logger = structlog.get_logger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GraphQueryError)
    async def handle_domain_error(request: Request, exc: GraphQueryError) -> JSONResponse:
        logger.warning("request_failed", code=exc.code, error_message=exc.safe_message)
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "code": exc.code,
                "message": exc.safe_message,
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error")
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "Internal error",
                "request_id": _request_id(request),
            },
        )
