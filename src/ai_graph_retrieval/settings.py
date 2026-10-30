"""Service configuration — the single source of truth for every tunable.

All values come from the environment (prefix ``AGR_``, plus a few conventional
unprefixed aliases such as ``NEO4J_URI``) or an optional ``.env`` file. Nothing
else in the codebase reads ``os.environ`` directly.
"""
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from itertools import pairwise

import structlog
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_logger = structlog.get_logger(__name__)

_EPSILON_S = 0.001  # smallest meaningful wall-clock budget; only 0/negative are broken

# Safe floor for every execution/digest knob that is used as a timeout or a slice bound.
# Below these, a value is not "unlimited" — it is actively wrong (a 0 budget expires
# before the first query; a negative slice drops the BEST-ranked items). A config below
# a floor is repaired and logged, never fatal — see Settings._normalise_execution_config.
# 0 is a legitimate value wherever it means "this mode does not get the capability"
# (e.g. Search's agent follow-up disabled), so those floors are 0.
_CONFIG_FLOORS: dict[str, float] = {
    # Wall-clock deadlines (asyncio.timeout).
    "search_budget_s": _EPSILON_S,
    "super_research_budget_s": _EPSILON_S,
    "synthesis_budget_s": _EPSILON_S,
    # Retrieval depth (list slice bounds).
    "rel_vocab_fetch": 1,
    "people_enumeration_limit": 1,
    "super_research_graph_chunk_cap": 1,
    # 0 is legitimate: "Search does not get graph chunk text".
    "search_graph_chunk_cap": 0,
    "source_docs_per_anchor": 1,
    "quotes_per_anchor": 1,
    "documents_per_title": 1,
    "quote_chars": 1,
    # Agent loop. Rounds may be 0 (no follow-up at all); patience may not.
    "agent_max_rounds": 1,
    "agent_stale_patience": 1,
    "search_agent_rounds": 0,
    "search_agent_patience": 1,
    "agent_tool_rows": 1,
    "agent_tool_rows_research": 1,
    # Post-lane passes. 0 means "this mode skips the pass".
    "search_scoring_budget": 0,
    "scoring_query_budget": 0,
    "enrich_chars_per_doc": 1,
    "document_chunk_limit": 1,
    "document_total_chars": 1,
    "answer_graph_cited_limit": 1,
    # Synthesis digest.
    "digest_max_entities": 1,
    "digest_max_facts": 1,
    "digest_max_claims": 1,
    "digest_max_quotes": 1,
    "digest_quote_char_budget": 1,
    "search_digest_quote_chars": 1,
    "super_research_digest_quote_chars": 1,
    # 0 is legitimate: it turns the truncated-fragment floor off.
    "digest_min_excerpt_chars": 0,
    "digest_entity_desc_chars": 1,
    "digest_fact_desc_chars": 1,
    "citation_snippet_chars": 1,
    # Conversation store.
    "graph_conversations_list_max": 1,
    "graph_store_timeout_ms": 1,
}


def _package_version() -> str:
    """The installed package version (pyproject ``version``), so trace
    ``service.version`` never drifts from a hand-copied literal."""
    try:
        return version("ai-graph-retrieval")
    except PackageNotFoundError:  # not installed as a dist (e.g. bare source run)
        return "0.0.0+unknown"


class GraphSource(BaseModel):
    """One knowledge graph the retrieval reads span (Neo4j database + its Qdrant
    collections + fulltext index name). Multi-graph = a list of these; every
    graph read/vector search fans out across them and the results are merged.
    All members live on the SAME Neo4j endpoint and Qdrant host — a source is
    just a different database + collection/index names, not a new connection."""

    model_config = ConfigDict(frozen=True)

    name: str  # provenance tag on returned evidence (e.g. "wikipedia", "arxiv")
    database: str  # Neo4j database (Query API /db/<database>/, or Bolt session db)
    entities_collection: str  # Qdrant entity-vector collection
    entity_fulltext_index: str = "entity_search"  # Neo4j fulltext index for entity lookup
    # Qdrant CHUNK-vector collection — one point per Neo4j chunk, the chunk_id in
    # the payload's `entity_id`. Empty ⇒ this source has no chunk index: it cannot
    # be similarity-searched and stays reachable by traversal only.
    chunks_collection: str = ""
    # Which graph's label wins when the SAME person is named differently in two
    # graphs (people layer only). Higher wins; equal ranks keep fan-out order.
    person_name_authority: int = 0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Aliased fields must also accept their field name as a constructor
        # kwarg (Settings(neo4j_uri=...) in tests/tools) — without this,
        # extra="ignore" silently discards such kwargs and the value would
        # leak in from the ambient env/.env instead.
        populate_by_name=True,
    )

    # ── Service ────────────────────────────────────────────────────────────
    environment: str = "dev"
    log_level: str = "INFO"
    service_host: str = "0.0.0.0"  # noqa: S104 — container-internal bind
    service_port: int = 5000
    internal_auth_token: str | None = Field(
        default=None,
        description="Optional shared secret; when set, callers must send it as x-internal-auth.",
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    # Reflects the caller's Origin and allows credentials. Set
    # cors_allow_origin_regex to an explicit allowlist before exposing the
    # service publicly.
    cors_enabled: bool = True
    cors_allow_origin_regex: str = ".*"

    # ── Caller identity ─────────────────────────────────────────────────────
    # Optional headers a fronting gateway may set. Absent ⇒ the defaults below:
    # one tenant, and the principals the ingestion stamps on public sources
    # (Wikipedia/arXiv documents carry ACL ["public"]).
    tenant_header: str = "x-tenant-id"
    user_header: str = "x-user-id"
    # Comma-separated principal tokens. Empty string ⇒ never read principals
    # from a header (always use default_principals).
    principals_header: str = "x-principals"
    default_tenant_id: str = "demo"
    default_principals: list[str] = Field(default_factory=lambda: ["public"])

    # ── ACL enforcement master switch ───────────────────────────────────────
    # Toggles principal filtering across BOTH stores:
    #   * Neo4j — every `any(p IN x.allow_access_control WHERE p IN $principals)`
    #     predicate is null-guarded (see cypher.acl_guarded), so with this OFF the
    #     repository binds `$principals = None` and the predicate is always true.
    #   * Qdrant — the ACL `MatchAny` leg is pushed into every chunk read when ON,
    #     and skipped when OFF.
    # Tenant isolation is not applied in store queries: each Neo4j database /
    # Qdrant collection is already scoped to a single tenant+run.
    acl_enforcement_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("ACL_ENFORCEMENT_ENABLED", "AGR_ACL_ENFORCEMENT_ENABLED"),
    )

    # Bind a conversation_id to its first owner and reject a mismatched owner.
    # Redis-backed when configured (cross-replica), in-memory fallback otherwise.
    enforce_conversation_ownership: bool = False
    conversation_ownership_ttl_s: float = 172800.0  # 2 days

    # ── Conversation ────────────────────────────────────────────────────────
    # When the entry classifier categorizes a turn as greeting / smalltalk /
    # capability / guardrail / out-of-scope, its reply is returned with no
    # retrieval and no synthesis call. Off ⇒ every turn runs the full pipeline.
    chat_smalltalk_shortcircuit: bool = True
    # Use the classifier's self-contained rewrite of an elliptical follow-up
    # ("who founded it?") as the retrieval query. Off ⇒ retrieve on the
    # verbatim question.
    enable_follow_up_support: bool = False
    # Conversational turns shown to the entry classifier for reference resolution.
    followup_history_turns: int = 3
    # Server-side conversation memory: when a request carries a conversation_id
    # and NO history, recent turns are loaded from Redis (or in-process) and each
    # answered turn is recorded back. Client-supplied history always wins.
    enable_multi_turn_conversation: bool = False
    conversation_memory_ttl_s: float = 172800.0  # 2 days, refreshed per turn
    # Chars stored per message — matches the API's HistoryTurn cap.
    conversation_memory_turn_chars: int = 4000

    # ── Clarifying questions ────────────────────────────────────────────────
    # The entry classifier writes 1-2 clarification questions only when a real
    # answer-changing ambiguity exists, and Search pauses on them (202 →
    # /retrieve/resume). Off ⇒ questions are classified but no mode pauses.
    search_clarification_enabled: bool = True
    clarification_session_ttl_s: float = 1800.0  # abandoned sessions self-clean
    # The research agent's request_clarification tool (Super Research only, see
    # profile.mid_research_clarify). This flag is the kill switch.
    mid_research_clarification_enabled: bool = True

    # ── Answer formatting ───────────────────────────────────────────────────
    # Synthesis attaches a [Document Title] chip to each factual statement and
    # the composer rewrites it into an inline link to the document's URL.
    inline_citations_enabled: bool = True
    # Apply the classifier's answer shape (timeline / table / per_member / prose).
    answer_shape_enabled: bool = True
    # Model-generated follow-up suggestions (fast tier); fail-open to [].
    suggested_actions_enabled: bool = True
    suggested_actions_max: int = 4
    # Far below synthesis_budget_s: a hung call loses the suggestions, never the answer.
    suggested_actions_timeout_s: float = 30.0

    # ── Optional people/HR graph layer ──────────────────────────────────────
    # Person-specific tools, prompts, profile/relationship enrichment and the
    # employment gate. Only meaningful for graphs whose ingestion connector writes
    # person nodes with employment status; Wikipedia/arXiv graphs do not.
    people_graph_enabled: bool = False
    # Upper bound on the names one person-identity read resolves. 0 disables it.
    person_identity_max_names: int = 400
    # Exclude people the source records as no longer active (people layer only).
    # An explicitly historical question keeps them (marked, never presented as
    # current).
    inactive_people_filter_enabled: bool = True

    # ═════════════════════════════ DATABASES ════════════════════════════════

    # ── Graph sources ───────────────────────────────────────────────────────
    # Connectors a request reads when it names none (``connectorType`` omitted).
    default_connectors: list[str] = Field(default_factory=lambda: ["wikipedia", "arxiv"])
    # Optional fixed sources, as JSON: [{"name": "wikipedia", "database": "...",
    # "entities_collection": "...", "chunks_collection": "..."}]. A connector with
    # a configured source uses it; otherwise its latest COMPLETED ingestion run is
    # resolved from run_registry (needs AGR_MONGO_URI).
    graph_sources_config: list[GraphSource] = Field(
        default_factory=list,
        validation_alias=AliasChoices("AGR_GRAPH_SOURCES", "graph_sources_config"),
    )
    # Fulltext index bound for a run resolved from run_registry (the record does
    # not store it). The ingestion creates `entity_search` in every run database.
    entity_fulltext_index: str = "entity_search"
    run_fulltext_index_by_connector: dict[str, str] = Field(default_factory=dict)
    # run_registry embedding_model → the model name to embed the query with, when
    # retrieval reaches the same model under a different name.
    embedding_model_to_deployment: dict[str, str] = Field(default_factory=dict)

    # ── Qdrant ──────────────────────────────────────────────────────────────
    qdrant_host: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_HOST", "QDRANT_URL", "AGR_QDRANT_HOST"))
    qdrant_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("QDRANT_API_KEY", "AGR_QDRANT_API_KEY"))
    qdrant_port: int | None = Field(
        default=None, validation_alias=AliasChoices("QDRANT_PORT", "AGR_QDRANT_PORT"))
    qdrant_timeout_s: float = 30.0
    qdrant_prefer_grpc: bool = False
    # TLS verification for an HTTPS Qdrant endpoint. Point qdrant_ca_cert at a
    # private CA's PEM, or set qdrant_verify_tls=false (dev only).
    qdrant_verify_tls: bool = Field(
        default=True, validation_alias=AliasChoices("QDRANT_VERIFY_TLS", "AGR_QDRANT_VERIFY_TLS")
    )
    qdrant_ca_cert: str | None = Field(
        default=None, validation_alias=AliasChoices("QDRANT_CA_CERT", "AGR_QDRANT_CA_CERT")
    )
    qdrant_entity_id_field: str = "entity_id"
    qdrant_acl_field: str = "acl"
    graph_acl_property: str = "allow_access_control"

    # Gate the live Neo4j/Qdrant connectivity probes (readiness endpoint +
    # background startup verification).
    health_check_stores: bool = Field(
        default=False,
        validation_alias=AliasChoices("HEALTH_CHECK_STORES", "AGR_HEALTH_CHECK_STORES"),
    )

    # ── MongoDB (run_registry + conversation store) ─────────────────────────
    mongo_uri: str | None = Field(
        default=None, validation_alias=AliasChoices("MONGO_URI", "AGR_MONGO_URI"))
    mongo_db: str = Field(default="", validation_alias=AliasChoices("MONGO_DB", "AGR_MONGO_DB"))
    run_registry_collection: str = "run_registry"
    run_registry_cache_ttl_s: float = 300.0
    # Durable conversation persistence: sessions (one doc per tenant+conversation)
    # and messages (a user + an ai_response doc per turn). Off unless mongo_uri is
    # set AND persistence is enabled.
    graph_conversations_collection: str = "graph_conversations"
    graph_messages_collection: str = "graph_messages"
    graph_conversation_persist_enabled: bool = True
    graph_conversations_list_max: int = 100  # clamp for the list endpoint's limit
    graph_store_timeout_ms: int = 3000  # serverSelectionTimeoutMS for the store client
    # After a store operation fails on transport, skip store calls for this long
    # instead of paying server selection again on every one. 0 disables.
    graph_store_retry_cooldown_s: float = Field(default=30.0, ge=0.0, allow_inf_nan=False)

    # ── Redis (optional: sessions, ownership, conversation memory) ──────────
    # Unset ⇒ every caller uses its in-process fallback (single replica only).
    redis_host: str | None = Field(
        default=None, validation_alias=AliasChoices("REDIS_HOST", "AGR_REDIS_HOST"))
    redis_port: int = Field(
        default=6379, validation_alias=AliasChoices("REDIS_PORT", "AGR_REDIS_PORT"))
    redis_username: str | None = Field(
        default=None, validation_alias=AliasChoices("REDIS_USERNAME", "AGR_REDIS_USERNAME"))
    redis_password: str | None = Field(
        default=None, validation_alias=AliasChoices("REDIS_PASSWORD", "AGR_REDIS_PASSWORD"))

    # ── Neo4j ───────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        validation_alias=AliasChoices("NEO4J_URI", "AGR_NEO4J_URI"))
    neo4j_username: str = Field(
        default="neo4j", validation_alias=AliasChoices("NEO4J_USERNAME", "AGR_NEO4J_USERNAME"))
    neo4j_password: str = Field(
        default="", validation_alias=AliasChoices("NEO4J_PASSWORD", "AGR_NEO4J_PASSWORD"))
    # Default database for connectivity checks. Per-request reads use the
    # resolved source's database. Non-optional: the HTTP transport puts it in
    # the URL path.
    neo4j_database: str = Field(
        default="neo4j",
        validation_alias=AliasChoices("NEO4J_DATABASE", "AGR_NEO4J_DATABASE"),
    )
    neo4j_pool_size: int = 50
    neo4j_connection_timeout_s: float = 5.0
    neo4j_tx_timeout_s: float = 120.0
    neo4j_http_keepalive_expiry_s: float = 60.0
    neo4j_http_max_keepalive: int = 20
    # Set to use the Neo4j Query API (HTTP v2) instead of Bolt.
    neo4j_http_url: str | None = Field(
        default=None, validation_alias=AliasChoices("NEO4J_HTTP_URL", "AGR_NEO4J_HTTP_URL"))

    # ═══════════════════════════ end DATABASES ══════════════════════════════

    # Hybrid entity-resolution blend weights (graph/repository.resolve_entities):
    # lexical fulltext hit + semantic ANN similarity + a small degree prior.
    entity_w_lexical: float = 0.6
    entity_w_semantic: float = 0.55
    entity_w_degree: float = 0.06

    # ── Excerpt ranking (which chunks the answer model actually reads) ───────
    # Blend of question similarity (retrieval/chunk_relevance) and the parent
    # document's trust score. Both zero ⇒ document order.
    quote_w_semantic: float = Field(default=0.70, ge=0.0, allow_inf_nan=False)
    quote_w_doc_trust: float = Field(default=0.30, ge=0.0, allow_inf_nan=False)
    # OFF restores document-order excerpts.
    quote_ranking_enabled: bool = True

    # ── Beam-bounded multi-hop traversal → chunks ───────────────────────────
    beam_hops: int = 2  # variable-length EXTRACTED_REL cap (int, substituted into Cypher)
    beam_leg_limit: int = 200  # (seed,neighbour) pairs kept before chunk fan-out (hub damping)
    beam_chunk_limit: int = 40  # chunks returned by one traversal
    # Recent-change chunks — claim-date → supporting chunk quotes.
    recent_change_chunk_limit: int = 20

    # ── LLM (any OpenAI-compatible endpoint: NVIDIA NIM, vLLM, …) ───────────
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    # NVIDIA_API_KEY is accepted so one .env can serve ingestion and retrieval.
    # Unset ⇒ "none" (a local vLLM needs no key).
    llm_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("AGR_LLM_API_KEY", "NVIDIA_API_KEY"))
    llm_model: str = "nvidia/nemotron-3.5-bolt-e4b"
    # Lightweight decisions (classification, re-ranking, suggestions). Empty ⇒ llm_model.
    fast_model: str = ""
    # max_tokens for each research-agent round.
    agent_max_tokens: int = 4096
    # None ⇒ omit the parameter (provider default). 0.0 keeps routing,
    # extraction and synthesis deterministic.
    llm_temperature: float | None = 0.0
    # False ⇒ never send llm_temperature (for models that reject sampling parameters).
    llm_send_temperature: bool = True
    llm_max_output_tokens: int | None = None  # floor for chat calls without an explicit cap
    llm_timeout_s: float = 600.0
    llm_max_retries: int = 4

    # ── Embeddings ──────────────────────────────────────────────────────────
    # Unset base URL / key ⇒ the LLM's.
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    # Must be the model the graph was ingested with. A run resolved from
    # run_registry overrides model and dimension per request
    # (core/run_config.EmbeddingSpace).
    embedding_model: str = "nvidia/nemotron-3.5-embed-1b"
    # Dimension the QUERY is embedded at. MUST equal the collection's dimension:
    # a mismatch is not degraded recall, it is ZERO hits.
    embedding_dimensions: int = 2048
    # NIM's asymmetric embed models need input_type ("query" for a search; the
    # ingestion embeds with "passage"). Empty ⇒ plain OpenAI request shape with
    # an explicit `dimensions` (OpenAI / vLLM models).
    embedding_input_type: str = "query"

    # ── Budgets ─────────────────────────────────────────────────────────────
    # Excerpt (verbatim chunk text) budget, in characters, per mode. The excerpts
    # are cut from a RELEVANCE-ordered list, so a larger budget reads further
    # down the relevance curve rather than padding with arbitrary chunks.
    search_digest_quote_chars: int = 100_000
    super_research_digest_quote_chars: int = 160_000
    # ── Per-request `options` override ceilings (RunConfig) ─────────────────
    # A caller-supplied depth knob is CLAMPED to these: the request can lower
    # cost/latency freely but can't push the pipeline past these bounds.
    # Effective values are echoed in options_applied.
    options_max_agent_rounds: int = 6
    options_max_graph_chunk_cap: int = 80
    search_budget_s: float = 1200.0
    super_research_budget_s: float = 1800.0
    # Synthesis runs AFTER (outside) the retrieval deadline; bounds the final
    # streamed synthesis pass on its own.
    synthesis_budget_s: float = 1800.0
    partial_on_timeout: bool = True  # synthesize from partial evidence on retrieval timeout
    router_timeout_s: float = 60.0
    # Each agent round re-sends the growing conversation, so rounds are the
    # dominant cost/token driver. Super Research's cap (via its profile).
    agent_max_rounds: int = 5
    agent_stale_patience: int = 2
    # Rows a list-returning agent tool hands back per call: Search, and the
    # research lane (whose job is breadth). Scaled per subject for a tool called
    # with several subjects at once (agent/tools.py _rows_per_subject).
    agent_tool_rows: int = 12
    agent_tool_rows_research: int = 50

    # ── Execution profiles: one depth table per mode ────────────────────────
    # Assembled in retrieval/profile.py. Super Research reuses agent_max_rounds /
    # agent_stale_patience / scoring_query_budget above.
    #
    # Graph chunks the agent tools may add TOTAL across all graphs, each
    # deduplicated against text already in the pool. The graph is the only source
    # of excerpt text, so these bound what the answer can quote. 0 for Search switches graph
    # chunk text off for that mode. Per-request: options.graph_chunk_cap.
    super_research_graph_chunk_cap: int = 50
    search_graph_chunk_cap: int = 20
    # Search is the lighter agent mode.
    search_agent_rounds: int = 2
    # FLAT rounds (no new evidence) that end the loop.
    search_agent_patience: int = 1
    # Post-loop claim top-up: claim-fetch up to this many pool entities that no
    # claim covers yet (one parallel batch, no extra agent rounds). 0 disables.
    agent_claim_enrich_top: int = 30
    # Of each fetched batch, keep only claims whose SUBJECT is the fetched entity,
    # at most this many — so one entity cannot fill the digest's claim window.
    agent_claim_enrich_per_entity: int = 5
    # Post-loop relationship top-up (people layer): edges between pooled people.
    agent_relationship_enrich_top: int = 30
    # Wall-clock bound on each read of that parallel batch.
    relationship_enrich_timeout_s: float = 10.0
    # Post-loop profile top-up (people layer): structured node fields for
    # surfaced-but-unprofiled entities. One batched query. 0 disables.
    agent_profile_enrich_top: int = 60
    # Post-loop mention top-up: entities the pool's graph chunks MENTION, ranked by
    # how many retrieved passages mention each. 0 disables.
    agent_mention_enrich_top: int = 40
    # Relationships AMONG the pool's entities (the induced subgraph), per graph.
    agent_induced_facts_limit: int = 200
    # The answer's graph (synthesis/answer_graph.py): row ceiling per graph on the
    # reads that fetch what the CITED documents mention.
    answer_graph_cited_limit: int = 300
    # Budgeted LLM neighbor re-ranking for Search (Super Research gets
    # scoring_query_budget). Fires only on a large, poorly separated fanout.
    search_scoring_budget: int = 2

    # Degenerate-answer floor: the fewest ALPHANUMERIC characters a real answer
    # carries. Below it the composer re-issues synthesis once and keeps the
    # better result. 0 disables the guard.
    synthesis_min_answer_chars: int = 12
    fast_max_tokens: int | None = 8000
    research_max_tokens: int | None = 8000
    # If a synthesis round stops on the max_tokens cap (finish_reason ==
    # "length"), continue for up to this many more rounds.
    synthesis_max_continuations: int = Field(default=20, ge=0)
    # Upper bound on ONE streamed `answer_delta` event's text (smooth token flow
    # in a UI). 0 passes provider chunks through unchanged.
    answer_delta_max_chars: int = Field(default=24, ge=0)
    # Upper bound on ONE trace_delta event's serialized size.
    trace_delta_max_bytes: int = Field(default=49_152, ge=1_024)

    # ── Synthesis digest caps: what the answer model is allowed to SEE ───────
    # Sized in CHARACTERS of prompt to stay well inside the model's context
    # window. A fact is one short line; an entity carries up to 600 chars of
    # description — hence the lopsided item counts.
    digest_max_entities: int = 100
    digest_max_facts: int = 600
    # Chars of derived description per item, truncated on a word boundary.
    digest_entity_desc_chars: int = 600
    digest_fact_desc_chars: int = 400
    digest_max_claims: int = 150
    digest_max_quotes: int = 80
    digest_quote_char_budget: int = 90_000
    # Spend the excerpt budget breadth-first across documents (one excerpt per
    # document per round, synthesis/dedup.interleave_by_document) instead of
    # filling greedily from the top-ranked documents. Same budget, same chunks.
    digest_round_robin_enabled: bool = True
    # Floor for an excerpt the character budget TRUNCATED: below it the excerpt is
    # skipped rather than rendered as a fragment. Genuinely short chunks are kept.
    digest_min_excerpt_chars: int = 200
    # Two documents opening with the same this-many normalized characters share
    # one round-robin group (republished copies). 0 disables.
    digest_near_duplicate_prefix_chars: int = 400

    # ── SLO targets (surfaced on /v1/graph/usage) ───────────────────────────
    slo_fast_p95_s: float = 5.0
    slo_research_p95_s: float = 60.0
    slo_first_token_s: float = 15.0

    # ── Concurrency ─────────────────────────────────────────────────────────
    max_concurrent_requests: int = 64
    admission_timeout_s: float = 2.0
    llm_max_concurrency: int = 16

    # ── Retrieval knobs ─────────────────────────────────────────────────────
    resolve_limit: int = 20
    fulltext_k: int = 60
    vector_overscan: int = 8
    neighbor_limit: int = 60  # per-entity edge fan-out (bounds context and hub explosion)
    claim_limit: int = 60
    # Dated facts pulled per temporal window.
    recent_limit: int = 100
    # Citable source documents fetched PER ANCHOR ENTITY (adding an anchor can
    # only add documents).
    source_docs_per_anchor: int = 8
    # Verbatim excerpts fetched PER ANCHOR ENTITY.
    quotes_per_anchor: int = 5
    # Documents returned PER REQUESTED TITLE by the title lookup.
    documents_per_title: int = 1
    # Per-chunk excerpt cap; the digest applies a boundary-aware total budget.
    quote_chars: int = 2500
    support_weight: float = 0.15
    default_temporal_months: int = 6
    history_max_turns: int = 6
    rel_vocab_ttl_s: float = 600.0
    # Rows the relationship-vocabulary read asks each graph for before merging.
    rel_vocab_fetch: int = 200
    # People an ENUMERATION read may return, per graph (people layer). A ceiling
    # against hub explosion, not a relevance cut.
    people_enumeration_limit: int = 200

    # ── Tiered neighbor scoring (research lane) ─────────────────────────────
    scoring_enabled: bool = Field(
        default=True,
        description=(
            "LLM re-ranking of ambiguous hub expansions in the research lane. Fires only "
            "when the fanout is large AND the cheap ranking is not cleanly separated AND "
            "per-request budget remains; always falls back to the cheap order on failure."
        ),
    )
    scoring_small_threshold: int = 30
    scoring_separation_margin: float = 0.05
    scoring_candidate_cap: int = 120
    scoring_batch_token_budget: int = 6000
    scoring_query_budget: int = 6
    scoring_timeout_s: float = 60.0

    # ── Agent document reading ──────────────────────────────────────────────
    # Chars of body text kept PER CHUNK when a document is read.
    enrich_chars_per_doc: int = 6000
    # Chunks a single document read may return, per graph (reported to the agent
    # as `complete`).
    document_chunk_limit: int = 50
    # Total characters a single document read may return across its chunks.
    document_total_chars: int = 60_000

    # CEILING for each citation's hover preview. It ends at the last complete
    # sentence within this budget.
    citation_snippet_chars: int = 150

    # ── Trust ranking weights (document citation order) ─────────────────────
    # `relevance` (fused chunk similarity for THIS query) leads; `multisource`
    # (graph mention count) is a corroboration nudge. Normalized by their sum.
    # NaN/inf/negatives are rejected at startup; zero disables a signal.
    trust_w_relevance: float = Field(default=0.40, ge=0.0, allow_inf_nan=False)
    trust_w_multisource: float = Field(default=0.25, ge=0.0, allow_inf_nan=False)
    trust_w_recency: float = Field(default=0.15, ge=0.0, allow_inf_nan=False)
    trust_w_support: float = Field(default=0.12, ge=0.0, allow_inf_nan=False)
    trust_w_authority: float = Field(default=0.08, ge=0.0, allow_inf_nan=False)
    trust_recency_halflife_days: float = 365.0
    # A question that NAMES its period has already stated its time scope, so
    # recency decay inside that window double-counts it. Multiplies
    # trust_w_recency when the classifier resolved a window; 1.0 disables.
    trust_recency_windowed_damping: float = Field(
        default=0.2, ge=0.0, le=1.0, allow_inf_nan=False
    )

    # ── Cost accounting (USD per 1M tokens; 0 = rate not configured) ────────
    # chat = llm_model; fast = fast_model. Fast and cached rates fall back to the
    # chat / full input rate when unset, so a missing rate never under-reports.
    cost_chat_input_per_1m: float = 0.0
    cost_chat_output_per_1m: float = 0.0
    cost_fast_input_per_1m: float | None = None
    cost_fast_output_per_1m: float | None = None
    cost_embedding_per_1m: float = 0.0
    cost_chat_cached_input_per_1m: float | None = None
    cost_fast_cached_input_per_1m: float | None = None

    # ── Tracing (OpenTelemetry, optional `tracing` extra) ───────────────────
    # Exports over OTLP/HTTP to a collector and/or Langfuse's OTLP endpoint.
    tracing_enabled: bool = False
    otel_service_name: str = Field(
        default="ai-graph-retrieval",
        validation_alias=AliasChoices("OTEL_SERVICE_NAME", "AGR_OTEL_SERVICE_NAME"),
    )
    service_version: str = Field(
        default_factory=_package_version,
        validation_alias=AliasChoices("SERVICE_VERSION", "AGR_SERVICE_VERSION"),
    )
    # OTLP/HTTP collector base URL (e.g. http://localhost:4318).
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "AGR_OTEL_EXPORTER_OTLP_ENDPOINT",
        ),
    )
    # Langfuse — all three required, else the Langfuse exporter is skipped.
    langfuse_public_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY", "AGR_LANGFUSE_PUBLIC_KEY"),
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY", "AGR_LANGFUSE_SECRET_KEY"),
    )
    langfuse_host: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL", "AGR_LANGFUSE_HOST"),
    )

    # ── Ledgers (non-blocking JSONL sinks; None disables file output) ───────
    usage_ledger_path: str | None = Field(
        default=None,
        description="JSONL file for per-request usage/cost entries (e.g. logs/usage.jsonl).",
    )
    audit_ledger_path: str | None = Field(
        default=None,
        description=(
            "JSONL file for per-request audit entries (docs served, principal hash — never "
            "raw tokens). Audit entries are ALWAYS also emitted as structured logs."
        ),
    )
    ledger_queue_size: int = 2048

    # ── Startup ─────────────────────────────────────────────────────────────
    warmup_enabled: bool = Field(
        default=True,
        description="Prime the embedding endpoint at startup (avoids first-query cold start).",
    )
    warmup_timeout_s: float = 10.0

    # ── Policy ──────────────────────────────────────────────────────────────
    strict_derived_text: bool = Field(
        default=False,
        description=(
            "When true, LLM-blended derived text (entity descriptions) is excluded from "
            "synthesis input."
        ),
    )

    @model_validator(mode="after")
    def _normalise_execution_config(self) -> "Settings":
        """Repair a nonsensical execution config instead of refusing to start.

        NOTHING here raises. Each problem below has exactly one safe repair, so it
        is applied and logged at WARNING with the field, the rejected value and the
        value actually used. Repairs only ever move a value in the SAFE direction.

        1. Values below a safe floor (deadlines and slice bounds, where 0 or
           negative is actively wrong, not "unlimited").
        2. An inverted budget ladder — Search given more wall-clock than Super
           Research. Repaired by raising the deeper mode.
        3. A digest excerpt budget too small to hold one anchor's excerpts.
        """
        if not self.fast_model.strip():
            object.__setattr__(self, "fast_model", self.llm_model)

        for name, floor in _CONFIG_FLOORS.items():
            value = getattr(self, name)
            if value < floor:
                self._repair(name, value, floor, "below its safe floor")

        for lower, higher in pairwise(("search_budget_s", "super_research_budget_s")):
            lo, hi = getattr(self, lower), getattr(self, higher)
            if hi < lo:
                self._repair(higher, hi, lo,
                             f"would make the budget_s ladder go backwards "
                             f"({lower}={lo})")

        needed = self.quotes_per_anchor * self.quote_chars
        if self.digest_quote_char_budget < needed:
            self._repair("digest_quote_char_budget", self.digest_quote_char_budget,
                         needed,
                         "cannot hold even one anchor's excerpt result "
                         "(quotes_per_anchor x quote_chars), so every anchor "
                         "contributing text would be undone by truncation")
        return self

    def _repair(self, field: str, given: object, used: object, why: str) -> None:
        """Apply one config repair and say so loudly. Never raises."""
        object.__setattr__(self, field, used)
        _logger.warning(
            "settings_auto_corrected", field=field, configured=given, using=used,
            reason=why,
            detail="the service started with the corrected value; fix the override to "
                   "silence this",
        )

    def graph_sources(self) -> list[GraphSource]:
        """The fixed graph sources from configuration (``AGR_GRAPH_SOURCES``).
        Empty ⇒ every request resolves its sources from run_registry."""
        return list(self.graph_sources_config)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings accessor (cached; env is read once)."""
    return Settings()
