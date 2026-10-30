"""The caller's identity — mandatory input for every graph read.

`AccessContext` is a frozen value object. It is a *positional argument of every
repository method*, which is the structural guarantee that no Cypher can run
without a tenant filter and a principal set (see docs/acl/neo4j-graph-acl-design.md §3.3).

Principals are either resolved server-side from the gateway identity headers
(WS1/WS2) or, in Phase-1 backward-compatibility mode, supplied verbatim as the
raw ES `_allow_access_control` tokens by the upstream caller.
"""
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AccessContext(BaseModel):
    """Tenant + principal token set for one request. Immutable."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(min_length=1)
    principals: tuple[str, ...] = Field(min_length=1)
    request_id: str = Field(min_length=1)
    # Per-request ACL-enforcement override. None ⇒ defer to
    # ``settings.acl_enforcement_enabled`` (the default). Set by the route from
    # the request's ``options.disabled_acl`` (True ⇒ False here = ACL off).
    acl_enforced: bool | None = None

    @field_validator("principals")
    @classmethod
    def _no_blank_principals(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(p.strip() for p in v if p and p.strip())
        if not cleaned:
            raise ValueError("principals must contain at least one non-empty token")
        return cleaned

    def log_fields(self) -> dict:
        """Safe-to-log summary — principal *count*, never the token values."""
        return {
            "tenant_id": self.tenant_id,
            "principal_count": len(self.principals),
            "request_id": self.request_id,
        }
