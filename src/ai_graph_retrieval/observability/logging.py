"""Structured JSON logging (structlog), request-id aware via contextvars."""
from __future__ import annotations

import logging

import structlog

# Third-party loggers that dump high-volume DEBUG noise (heartbeats, pool
# churn, wire protocol) when the root level is DEBUG. Keep app DEBUG useful.
_NOISY_LOGGERS = (
    "uvicorn.access",  # our access middleware logs requests
    "pymongo",         # Atlas topology heartbeats every ~10s per host
    "pymongo.topology",
    "pymongo.connection",
    "pymongo.serverSelection",
    "pymongo.command",
)


def silence_noisy_loggers(min_level: int = logging.WARNING) -> None:
    """Clamp chatty library loggers so AGR_LOG_LEVEL=DEBUG stays readable."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(min_level)


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")
    silence_noisy_loggers()
