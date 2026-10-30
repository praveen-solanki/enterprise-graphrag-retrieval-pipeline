# ai-graph-retrieval

ACL-aware GraphRAG retrieval over a Neo4j + Qdrant knowledge graph. It answers a
question with a streamed, cited answer grounded in the graph, plus the evidence and
a live trace of how it was found.

It reads the graphs written by
[ai-graph-ingestion](../ai-graph-ingestion) (Wikipedia and arXiv connectors), and
talks to any OpenAI-compatible LLM endpoint: NVIDIA NIM by default, or a local
vLLM / OpenAI.

## What it does

The caller picks the mode per request (`mode` is required; there is no default and
no automatic routing):

| Mode | Strategy | What runs |
|---|---|---|
| `search` | `search` | A bounded research-agent loop (≤ `AGR_SEARCH_AGENT_ROUNDS`) that resolves entities, walks relationships and reads graph chunks through its ACL-bound tools, followed by enrichment passes and one synthesis. |
| `super_research` | `deep_research` | The same flow with more rounds (≤ `AGR_AGENT_MAX_ROUNDS`), a larger graph-chunk allowance, wider tool rows and mid-research clarification. |

The agent is the intelligence layer: it decides what to resolve, expand,
claim-fetch and read through its toolset (`agent/tools.py`). Every strategy fills
one evidence pool and one live graph trace, and ends in the same streamed, cited
synthesis pass. Every Cypher read is ACL-anchored; there is no free-form Cypher
anywhere in the service.

Each request runs through fixed pipeline stages (`pipeline/`): **planning** (one
lightweight entry classification) → **gates** (may pause once, pre-retrieval, on
clarifying questions the classifier wrote) → **execution** (the mode's strategy
under one deadline) → **quality** (censor → contradictions → trust ranking) →
**synthesis** (streamed answer + suggested follow-ups) → **respond**.

A greeting, thanks, capability question or guardrail case is answered directly by
the entry classifier's reply; retrieval and the heavy model never run.

### Answer-shaping behaviour

- **Relevance-led ranking.** Sources are ranked by a trust blend led by the query
  relevance the retrieval computed, then corroboration, recency, support and
  authority (`AGR_TRUST_W_*`).
- **Inline citations.** Synthesis attaches a `[Document Title]` chip to each
  statement; the composer resolves chips to real documents and appends the
  `## Sources` list. A chip naming an unknown document is dropped, so a hallucinated
  citation never ships.
- **Conditional clarification.** The entry classifier writes clarifying questions
  only when a genuine answer-changing ambiguity exists; the request then returns
  `202` and continues through `/retrieve/resume`.
- **Suggested follow-ups.** Every answer carries a few model-generated next
  questions/actions (proposed only, never executed).
- **No-answer detection.** `answer_found` is evidence-based: zero evidence reports
  false, and the answer says so plainly.
- **Optional people layer.** Tools for people directories (profiles, reporting
  lines, inactive-person filtering) are off by default
  (`AGR_PEOPLE_GRAPH_ENABLED=false`); turn them on only for a connector that writes
  those properties.

## Graph sources

Each connector (`AGR_DEFAULT_CONNECTORS`, default `["wikipedia","arxiv"]`) resolves
to one Neo4j database + Qdrant collections:

1. `runIdSuffix` in the request → that exact ingestion run from MongoDB
   `run_registry`.
2. Otherwise a fixed source from `AGR_GRAPH_SOURCES` (JSON), if configured.
3. Otherwise the connector's latest `COMPLETED` run in `run_registry`.

A connector the request names that cannot be resolved is a `422`; default
connectors with no run yet are skipped (a `422` only if none resolves). The run's
embedding model and dimension are used to embed the query, so the query always
lands in the same vector space as the ingested passages.

## Run locally

Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras           # .venv + pinned deps (extras: tracing, mongo, redis)
cp .env.example .env           # set NVIDIA_API_KEY; store defaults match ai-graph-ingestion
uv run ai-graph-retrieval      # UI → http://localhost:5000/ui
```

Start the ingestion stack first (its `docker-compose.yml` publishes Neo4j, Qdrant
and MongoDB on localhost) and ingest at least one run.

With Docker instead:

```bash
docker compose up --build      # reads .env; reaches the stores via host.docker.internal
```

> Behind a TLS-intercepting proxy? Append `--native-tls` to the uv commands.

## Endpoints

| Method & path | Purpose |
|---|---|
| `POST /v1/graph/retrieve` | one-shot JSON retrieval; may answer `202 awaiting_clarification` |
| `POST /v1/graph/retrieve/stream` | SSE: `meta`, `step`, `trace_delta`, `answer_delta`, `clarification_request`, `final`, `error` |
| `POST /v1/graph/retrieve/resume` · `.../resume/stream` | answer a clarification and get the answer |
| `GET /v1/graph/conversations` · `.../{id}/messages` | stored conversations (needs `AGR_MONGO_URI`) |
| `GET /v1/graph/usage` | cumulative token + cost report |
| `GET /ui` | page to ask a question and draw the answer graph. The tenant dropdown is the completed ingestion runs (`GET /v1/graph/tenants`), newest first |
| `GET /` · `/health` · `/startUpProbe` · `/livenessProbe` · `/readinessProbe` | probes |

```bash
curl -s localhost:5000/v1/graph/retrieve \
  -H 'content-type: application/json' \
  -H 'x-tenant-id: demo' \
  -d '{
    "question": "Who introduced the transformer architecture?",
    "mode": "search",
    "connectorType": ["wikipedia", "arxiv"]
  }'
```

### Identity and ACLs

- **Tenant:** the `x-tenant-id` header (`AGR_TENANT_HEADER`), falling back to
  `AGR_DEFAULT_TENANT_ID` (`demo`, the ingestion's example tenant).
- **Principals:** the `x-principals` header (comma-separated, `AGR_PRINCIPALS_HEADER`),
  falling back to `AGR_DEFAULT_PRINCIPALS` (`["public"]`, which is what the
  Wikipedia/arXiv connectors stamp on every document).
- **User:** the `x-user-id` header binds a conversation to its first owner.

ACL enforcement is on by default (`ACL_ENFORCEMENT_ENABLED=true`). The principals
header is trusted as-is, so expose the service only behind a gateway that sets it,
or set `AGR_PRINCIPALS_HEADER=` to ignore it. Set `AGR_INTERNAL_AUTH_TOKEN` to
require a shared secret (`x-internal-auth`) on every call.

### Response shape

```jsonc
{
  "mode": "search",
  "answer_found": true,
  "answer": "…introduced in 2017 [Attention Is All You Need]…\n\n## Sources\n- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)",
  // The documents the answer actually cited, ranked best first.
  "citations": [{"doc_id": "…", "title": "Attention Is All You Need",
                 "url": "https://arxiv.org/abs/1706.03762", "graph": "arxiv",
                 "relevance": 0.92, "trust": 0.71}],
  "evidence":  { "entities": [], "facts": [], "claims": [], "quotes": [], "sources": [] },
  "trace":     { "nodes": [], "edges": [] },
  "answer_graph": { "nodes": [], "edges": [] },
  "suggested_actions": [{"action_id": "…", "label": "Compare with RNN encoders"}],
  "options_applied": { "run": { "run_selection": "resolved",
                                "sources": [{"name": "arxiv", "run_suffix": "…", "selected_by": "latest"}] } },
  "usage": {"llm_calls": 6, "cost_usd": 0.0}, "duration_ms": 18400
}
```

`cost_usd` is computed from `AGR_COST_*` rates (USD per 1M tokens), which default to
0 because pricing depends on where the model is hosted.

## Configuration

All settings live in `settings.py` (env prefix `AGR_`; connection settings also
accept their conventional unprefixed names such as `NEO4J_URI`, `QDRANT_URL`,
`MONGO_URI`, `NVIDIA_API_KEY`). The main ones:

| Setting | Default |
|---|---|
| `AGR_LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` |
| `AGR_LLM_MODEL` / `AGR_FAST_MODEL` | `nvidia/nemotron-3.5-bolt-e4b` / same as `AGR_LLM_MODEL` |
| `AGR_EMBEDDING_MODEL` / `AGR_EMBEDDING_DIMENSIONS` | `nvidia/nemotron-3.5-embed-1b` / `2048` |
| `AGR_EMBEDDING_INPUT_TYPE` | `query` (ingestion embeds with `passage`) |
| `NEO4J_URI` or `NEO4J_HTTP_URL` | `bolt://localhost:7687` |
| `QDRANT_URL` | unset (vector search disabled) |
| `MONGO_URI` / `MONGO_DB` | unset (run resolution needs `AGR_GRAPH_SOURCES` instead) |
| `REDIS_HOST` | unset (in-process session store) |
| `AGR_TRACING_ENABLED` | `false` |

### Tracing

Install the `tracing` extra, set `AGR_TRACING_ENABLED=true`, and point it at an
OTLP/HTTP collector (`OTEL_EXPORTER_OTLP_ENDPOINT`, e.g. Jaeger or an OTel
Collector) and/or Langfuse (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
`LANGFUSE_HOST`, exported to Langfuse's OTLP endpoint). Every LLM call is a span
with model, tokens and computed cost, nested under the request's root span.

## Layout

```
src/ai_graph_retrieval/
├── settings.py        # all config (env prefix AGR_)
├── container.py       # composition root (no globals anywhere)
├── orchestrator.py    # request lifecycle: sequences the pipeline stages
├── pipeline/          # one module per stage: planning → gates → execution →
│                      #   quality → synthesis → respond (+ recording)
├── core/              # domain: access context, models, events, progress, errors
├── graph/             # Neo4j adapter — cypher.py holds every query + ACL guards
├── connectors/        # outbound I/O: neo4j, qdrant, llm, mongo, redis
├── retrieval/         # router, per-mode strategies, chunk relevance, trust ranking
├── agent/             # OpenAI Agents SDK research loop + ACL-bound tools
├── clarification/     # ambiguity gate, clarify/resume sessions
├── conversation/      # optional multi-turn memory + conversation store
├── actions/           # model-generated follow-ups (proposed, never executed)
├── synthesis/         # evidence digest → streamed answer + citation resolution
├── observability/     # structlog logging, OpenTelemetry tracing, ACL audit
└── api/               # FastAPI routes, schemas, middleware, error mapping
```

## Tests

```bash
uv run pytest                      # full suite — fast, no network
uv run ruff check src tests scripts
```

The DI structure lets the repository, strategies, ranking, synthesis and agent
runtime be tested without a live Neo4j or LLM.

## Design notes

- **ACL property = `acl`.** Chunks and documents carry their ACL tokens on `acl`;
  claims are visible through `(:Claim)-[:SUPPORTED_BY]->(:Chunk)` and relationships
  through `r.chunk_ids` (visible if any supporting chunk is visible).
- **Tenant isolation is database/collection-scoped.** Each Neo4j database and
  Qdrant collection holds a single tenant + run, so reads do not filter on
  `tenant_id`; the tenant still scopes run resolution, ownership and audit.
- **CORS** reflects the caller's `Origin` (`AGR_CORS_ALLOW_ORIGIN_REGEX` defaults to
  `.*`). Set an explicit allowlist before exposing the service publicly.
