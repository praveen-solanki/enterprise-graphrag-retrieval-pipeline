"""Request-scoped ACL audit → ONE summary log per request.

The repository records each fetched graph node's ACL check into a ContextVar during
a request; at the request boundary the orchestrator emits a single ``🔒 acl_summary``
line with the resolved **user token** and every **node** it searched (its
``allow_access_control`` + ``isACLMatch``).

Logger name ``"acl"`` — the app filters by level (not name), so this emits at INFO.
Never raises: logging must not break a request.
"""
from __future__ import annotations

import contextvars
from typing import Any

import structlog

logger = structlog.get_logger("acl")

# Per-request node list. None = no audit active (record() is then a no-op).
_NODES: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "acl_audit_nodes", default=None
)


def begin() -> None:
    """Start collecting for this request. Call once at the request boundary."""
    _NODES.set([])


def record(*, node: str | None, graph: str, acl: list, is_match: bool | None) -> None:
    """Record one fetched node's ACL check. No-op when no audit is active."""
    nodes = _NODES.get()
    if nodes is not None:
        nodes.append({
            "node": node, "graph": graph,
            "isACLMatch": is_match, "allow_access_control": acl,
        })


def log_summary(ctx: Any) -> None:
    """Emit the per-request ACL summary and end collection. Never raises."""
    try:
        nodes = _NODES.get() or []
        # The caller's ACL principal identifiers (site:/caud:/user: group labels —
        # NOT credentials; the same values are stamped on every audited node's
        # allow_access_control below). Named `principals`, not "token": the values
        # are group identifiers, and a token-named field reads as a credential to
        # humans and secret scanners alike.
        principals = list(ctx.principals)
        matches = [n.get("isACLMatch") for n in nodes if n.get("isACLMatch") is not None]
        # Overall: true if any searched node matched; false when none searched / none matched.
        is_acl_match = any(matches) if matches else False
        logger.info(
            "🔒 acl_summary",
            request_id=ctx.request_id,
            tenant_id=ctx.tenant_id,
            principals=principals,
            nodes_searched=len(nodes),
            isACLMatch=is_acl_match,
            nodes=nodes,
        )
    except Exception as exc:
        # Logging must never break a request.
        logger.warning("acl_summary_failed", error=str(exc))
    finally:
        _NODES.set(None)
