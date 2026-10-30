"""Process entrypoint: `ai-graph-retrieval` (console script) or
`python -m ai_graph_retrieval.main`."""
from __future__ import annotations

import sys

# A TLS-intercepting proxy replaces NVIDIA's certificate. truststore makes the
# client use the OS store, which already has that proxy's CA. Absent otherwise.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import uvicorn

from ai_graph_retrieval.settings import get_settings


def run() -> None:
    settings = get_settings()
    reload = "--reload" in sys.argv
    uvicorn.run(
        "ai_graph_retrieval.app:create_app",
        factory=True,
        host=settings.service_host,
        port=settings.service_port,
        reload=reload,
        log_config=None,  # structlog owns logging
    )


if __name__ == "__main__":
    run()
