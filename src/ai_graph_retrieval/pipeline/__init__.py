"""The retrieval pipeline, one stage per module.

`RetrievalOrchestrator` is a thin coordinator; every phase of a request lives
here as an independently constructable, independently testable stage with one
job each:

    planning    — ONE lightweight entry classification resolved into an
                  executable decision: conversational triage (greeting/guardrail/
                  out-of-scope short-circuit with a model-written reply),
                  follow-up rewrite, and the lane strategy + budget; routing
                  never executes anything
    gates       — the conditional pre-retrieval pause (the classifier's own
                  clarification questions) before spending the retrieval budget
    execution   — runs the API seed + the mode's strategy under its budget
    quality     — censor → contradictions → trust ranking over the evidence pool
    synthesis   — composes the answer + suggests next actions, on its own budget
    respond     — shapes the one response payload (normal + short-circuit paths)
    recording   — the non-blocking audit + usage records every request leaves
    state       — the mutable per-request record the stages fill in

Stages hold only long-lived collaborators (settings, clients, ledgers); all
per-request data flows through explicit arguments — nothing request-scoped is
stored on a stage, so one instance of each serves every request concurrently.
Routing (deciding what to run) is fully separated from execution (running it):
`ExecutionPlanner` returns a decision, `EvidenceCollector` carries it out, and
new lanes or execution profiles plug in at that seam without touching either.
"""
