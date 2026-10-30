"""Error taxonomy — every failure the service can surface, with HTTP mapping.

Handlers in `api/errors.py` translate these into consistent
``{code, message, request_id}`` JSON bodies. Anything not in this taxonomy is a
bug and surfaces as a sanitized 500.
"""


class GraphQueryError(Exception):
    """Base class. `safe_message` is what callers see; internals stay in logs."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, safe_message: str = "Internal error"):
        super().__init__(safe_message)
        self.safe_message = safe_message


class InvalidRequest(GraphQueryError):
    code = "invalid_request"
    http_status = 400


class ConversationConflict(GraphQueryError):
    """A conversation_id was presented by a different owner than it was bound to
    (E3 cross-session history-leak guard)."""

    code = "conversation_conflict"
    http_status = 409

    def __init__(self, safe_message: str = "conversation_id belongs to a different session"):
        super().__init__(safe_message)


class ServiceOverloaded(GraphQueryError):
    code = "overloaded"
    http_status = 503

    def __init__(self, safe_message: str = "Service is at capacity, retry shortly"):
        super().__init__(safe_message)


class RetrievalTimeout(GraphQueryError):
    code = "timeout"
    http_status = 504

    def __init__(self, safe_message: str = "Retrieval exceeded its time budget"):
        super().__init__(safe_message)


class GraphUnavailable(GraphQueryError):
    code = "graph_unavailable"
    http_status = 503

    def __init__(self, safe_message: str = "Knowledge graph is unavailable"):
        super().__init__(safe_message)


class LlmUnavailable(GraphQueryError):
    code = "llm_unavailable"
    http_status = 502

    def __init__(self, safe_message: str = "Language model backend is unavailable"):
        super().__init__(safe_message)
