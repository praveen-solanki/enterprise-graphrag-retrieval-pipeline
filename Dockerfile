# ai-graph-retrieval — production image (Python/uv, FastAPI).
FROM python:3.11-slim-bookworm

WORKDIR /app

# venv on PATH so the `ai-graph-retrieval` console script (installed by uv sync)
# resolves at runtime; UV_* keep the build lean/reproducible.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

# Optional integrations to install (see [project.optional-dependencies]).
ARG EXTRAS=all

# --- Dependency layer (cached until pyproject.toml / uv.lock change) ------------------
# README.md is required because hatchling reads `readme = "README.md"` when it builds the
# project wheel in the next stage.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra "${EXTRAS}" --no-install-project --no-cache

# --- Project layer --------------------------------------------------------------------
COPY src/ ./src/
RUN uv sync --frozen --no-dev --extra "${EXTRAS}" --no-editable --no-cache

# --- Runtime --------------------------------------------------------------------------
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# The service binds 0.0.0.0:5000 (settings default / AGR_SERVICE_PORT).
EXPOSE 5000

# Console script entry (ai_graph_retrieval.main:run) — preserves the app's own
# uvicorn config, including log_config=None so structlog owns logging (a raw
# `uvicorn` CMD would install uvicorn's default logging and fight structlog).
CMD ["ai-graph-retrieval"]
