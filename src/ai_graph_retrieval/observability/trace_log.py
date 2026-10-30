"""End-to-end pipeline trace logging — helpers for full observability.

Every log line is a structlog JSON record that inherits the per-request
``request_id`` bound into contextvars by ``RequestContextMiddleware``, so all lines
for one request share a single correlation id and the complete execution flow can be
reconstructed from Kubernetes pod logs (``jq 'select(.request_id=="…")'``).

Secrecy rules enforced here:
  * config snapshots mask every secret value (``redact_settings``);
  * connection strings/endpoints are reduced to scheme+host (no embedded creds/paths);
  * prompt logging is metadata-only (role counts + sizes) — never prompt content.

Verbosity is controlled by the log level (``AGR_LOG_LEVEL`` → DEBUG/INFO/WARNING/ERROR)
with no code change: milestones log at INFO, per-item I/O at DEBUG.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import structlog

logger = structlog.get_logger("agr.trace")

# Field-name fragments that mark a value as secret → masked in any snapshot.
_SECRET_MARKERS = ("password", "api_key", "apikey", "secret", "token")
# Field-name fragments whose value is a connection string → keep scheme+host only.
_LOCATION_MARKERS = ("endpoint", "_uri", "_url", "_host")


def _is_secret(name: str) -> bool:
    n = name.lower()
    return any(m in n for m in _SECRET_MARKERS)


def _is_location(name: str) -> bool:
    n = name.lower()
    return any(m in n for m in _LOCATION_MARKERS)


def host_only(url: str) -> str:
    """scheme://host[:port] — drops path, query, and any embedded credentials.

    Fails CLOSED: this is a redaction helper, so a value it cannot parse is
    replaced wholesale rather than logged verbatim — returning the input on a
    parse failure would leak exactly the embedded credentials this exists to
    strip. A scheme-less value (a bare hostname) carries no userinfo and passes
    through.
    """
    try:
        s = urlsplit(url)
        if not s.scheme:
            return url
        return f"{s.scheme}://{s.hostname}" + (f":{s.port}" if s.port else "")
    except Exception:
        return "<unparseable-url>"


def redact_settings(settings: Any) -> dict[str, Any]:
    """Effective-config snapshot with secrets masked and endpoints reduced to host.
    Emitted once at startup so operators can see exactly what config is live without
    exposing credentials."""
    data = settings.model_dump() if hasattr(settings, "model_dump") else dict(settings)
    out: dict[str, Any] = {}
    for k, v in data.items():
        # Only mask STRING secret fields (real credentials). Numeric fields whose name
        # happens to contain a marker — e.g. *_max_tokens, *_token_budget, *_token_s —
        # are budgets/counts, not secrets, and are shown so config stays debuggable.
        if _is_secret(k) and isinstance(v, (str, type(None))):
            out[k] = "***set***" if isinstance(v, str) and v else "***unset***"
        elif v is not None and _is_location(k):
            out[k] = host_only(str(v))
        else:
            out[k] = v
    return out


def prompt_metadata(messages: list[dict]) -> dict[str, Any]:
    """Non-sensitive shape of an LLM message list: role counts + total chars. Never
    the content itself — prompt text is not logged."""
    roles: dict[str, int] = {}
    chars = 0
    for m in messages:
        roles[m.get("role", "?")] = roles.get(m.get("role", "?"), 0) + 1
        c = m.get("content")
        chars += len(c) if isinstance(c, str) else 0
    return {"messages": len(messages), "roles": roles, "prompt_chars": chars}
