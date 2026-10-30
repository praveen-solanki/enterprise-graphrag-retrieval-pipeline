"""Liveness, readiness, and startup probes."""
from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ai_graph_retrieval.api.deps import get_container
from ai_graph_retrieval.container import Container

logger = logging.getLogger(__name__)

# Headroom for Neo4j HTTP Query API retries on transient gateway 5xx.
_READY_TIMEOUT_S = 3.0

_services: dict[str, bool] = {}
_ready = False

router = APIRouter(tags=["health"])


def register_service(name: str) -> None:
    """Register a service that must be initialized before startup probe passes."""
    _services[name] = False


def mark_initialized(name: str) -> None:
    """Mark a registered service as initialized."""
    if name in _services:
        _services[name] = True
        logger.info("component_initialized: %s", name)


def set_ready(ready: bool = True) -> None:
    global _ready
    _ready = ready
    logger.info("service_ready_state: %s", "ready" if ready else "not_ready")


def get_readiness_timeout_seconds() -> float:
    raw = os.environ.get("HEALTH_READINESS_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return _READY_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        return _READY_TIMEOUT_S


async def check_readiness(container: Container) -> dict:
    """Readiness probe.

    Store (Neo4j + Qdrant) reachability gates readiness only when
    ``settings.health_check_stores`` is on. It is OFF by default — dev stores
    behind an internal CA / flaky gateway were flapping the probe and bouncing
    the pod — so readiness then reflects only that the service process is up."""
    out: dict = {"probeType": "readiness", "status": "pass", "reason": []}

    if not container.settings.health_check_stores:
        return out

    probe = container.http_graph or container.graph
    ready_timeout = get_readiness_timeout_seconds()
    out["neo4j"] = "ok"
    try:
        await asyncio.wait_for(probe.verify(), timeout=ready_timeout)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.error("❌ readiness_neo4j_check_failed: %s", reason)
        out["neo4j"] = "unreachable"
        out["status"] = "fail"
        out["reason"] = [f"neo4j: {reason}"]
        return out

    if container.vector_store is not None:
        ok = False
        try:
            ok = await asyncio.wait_for(
                container.vector_store.verify(), timeout=ready_timeout
            )
        except Exception as exc:
            out["reason"].append(f"qdrant: {type(exc).__name__}: {exc}")
        else:
            if not ok:
                out["reason"].append("qdrant: unreachable")
        out["qdrant"] = "ok" if ok else "unreachable"
        if not ok:
            out["status"] = "fail"

    # Mongo backs the ingestion run_registry (per-request run selection). Only
    # probed when the feature is on (settings.mongo_uri set → container built the
    # client); when off there is nothing to check and no `mongo` key is emitted.
    if container.run_registry is not None:
        ok = False
        try:
            ok = await asyncio.wait_for(
                container.run_registry.verify(), timeout=ready_timeout
            )
        except Exception as exc:
            out["reason"].append(f"mongo: {type(exc).__name__}: {exc}")
        else:
            if not ok:
                out["reason"].append("mongo: unreachable")
        out["mongo"] = "ok" if ok else "unreachable"
        if not ok:
            out["status"] = "fail"

    if out["status"] == "fail":
        logger.error("❌ readiness_check_failed: %s", out["reason"])
    else:
        logger.info("✅ readiness_check_passed: %s", out)
    return out


@router.get("/")
async def index() -> dict:
    return {"status": "ok"}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/startUpProbe")
async def startup_probe() -> JSONResponse:
    """Startup gate. No logs per SRE checklist (item 7)."""
    pending = [name for name, ok in _services.items() if not ok]
    if pending:
        return JSONResponse({"status": "starting", "pending": pending}, status_code=503)
    return JSONResponse({"status": "ok"}, status_code=200)


@router.get("/livenessProbe")
async def liveness_probe() -> dict:
    """Liveness: event loop is alive. No logs, no external deps per SRE checklist (item 6)."""
    return {"status": "ok"}


@router.get("/readinessProbe")
async def readiness_probe(request: Request) -> JSONResponse:
    """Readiness: service ready + dependencies healthy. ERROR logs on failure, no log on success."""
    if not _ready:
        payload = {"probeType": "readiness", "status": "fail", "reason": ["service not ready"]}
        logger.error("❌ readiness_probe_service_not_ready")
        return JSONResponse(payload, status_code=503)

    pending = [name for name, ok in _services.items() if not ok]
    if pending:
        payload = {
            "probeType": "readiness",
            "status": "fail",
            "reason": [f"startup pending: {pending}"],
        }
        logger.error("❌ readiness_probe_startup_pending: %s", pending)
        return JSONResponse(payload, status_code=503)

    container = get_container(request)
    # Per-dep budget applies to Neo4j, Qdrant and Mongo sequentially; outer must cover all three.
    try:
        payload = await asyncio.wait_for(
            check_readiness(container),
            timeout=get_readiness_timeout_seconds() * 3,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        payload = {"probeType": "readiness", "status": "fail", "reason": [reason]}
        logger.error("❌ readiness_probe_exception: %s", reason)
        return JSONResponse(payload, status_code=503)

    status_code = 200 if payload.get("status") == "pass" else 503
    return JSONResponse(payload, status_code=status_code)
