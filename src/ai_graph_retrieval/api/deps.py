"""FastAPI dependencies — container access and internal auth."""
from __future__ import annotations

import secrets

from fastapi import Depends, Header, Request

from ai_graph_retrieval.container import Container
from ai_graph_retrieval.core.exceptions import InvalidRequest


def get_container(request: Request) -> Container:
    return request.app.state.container


async def require_internal_auth(
    container: Container = Depends(get_container),
    x_internal_auth: str | None = Header(default=None),
) -> None:
    """When a shared secret is configured, every call must present it."""
    expected = container.settings.internal_auth_token
    if expected is None:
        return
    if not x_internal_auth or not secrets.compare_digest(x_internal_auth, expected):
        raise InvalidRequest("Missing or invalid internal auth token")


async def identity_headers(
    x_tenant_id: str | None = Header(
        default=None,
        description="Tenant id (defaults to AGR_DEFAULT_TENANT_ID).",
    ),
    x_user_id: str | None = Header(
        default=None,
        description="User id (conversation ownership).",
    ),
) -> tuple[str | None, str | None]:
    """Declare the optional identity headers so Swagger renders inputs for them.
    Resolution still reads them from the request in `_access_context` (via
    `settings.tenant_header` / `settings.user_header`)."""
    return x_tenant_id, x_user_id
