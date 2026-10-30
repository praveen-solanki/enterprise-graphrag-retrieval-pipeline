"""Vector similarity — one implementation, used everywhere a cosine is needed.

There were three copies of this function (the API-seed anchor ranking, the agent's
claim ordering, and the digest's graph-chunk ranking) and they had already drifted:
two clamped the result to [0, 1] and one returned the raw quotient. Three copies of
a scoring primitive means three definitions of what a score MEANS, which is exactly
the kind of divergence that makes two rankings disagree for reasons nobody can see.

CLAMPED to [0, 1] deliberately. A negative cosine means "pointing away", which for
every caller here is the same thing as "not similar" — none of them wants an
ordering among degrees of irrelevance — and a clamped range makes a score readable
as a similarity anywhere it is logged or compared. Real text embeddings from one
model very rarely produce a negative cosine at all, so this is a definition, not a
behaviour knob.

Stdlib only, no settings, no state: safe to import from any layer.
"""
from __future__ import annotations


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity of two equal-length vectors, clamped to [0, 1].

    Returns 0.0 when either vector is missing or has zero norm, so a caller can
    treat "no vector" and "no similarity" identically and never has to guard the
    call. Lengths are zipped without a strictness check: a width mismatch is a
    configuration error the vector store rejects at query time, and scoring is not
    the place to raise it.
    """
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))
