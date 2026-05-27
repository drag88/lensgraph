"""Cite node — VALIDATION, not snapping (design §5 finding #9).

For each ``Citation`` the model emitted in ``state['parsed'].citations``:

  * Compute overlap_ratio against every reranked chunk with the same
    ``video_id``. The ratio is

        overlap_seconds / max(citation_span_seconds, epsilon)

    using a half-open interval (so a zero-length citation is treated
    as invalid).
  * If max overlap >= 0.5 → ``ValidatedCitation(matched_chunk_id=...)``.
  * Else → ``InvalidCitation(reason=...)``. The reason surfaces the
    model-side hallucination so phase-2's CitationAccuracy metric
    catches it.

If the parser flagged ``parse_ok=False`` OR the model set
``abstain=True``, we emit an abstention ``AnswerResult`` with no
citations populated — the trace still records what the model said via
``raw_answer`` upstream.
"""

from __future__ import annotations

from generate.parser import Citation
from generate.state import (
    AgentState,
    AnswerResult,
    InvalidCitation,
    RetrievedChunk,
    ValidatedCitation,
)

_MIN_OVERLAP = 0.5


def _overlap_ratio(cit: Citation, chunk: RetrievedChunk) -> float:
    """Fraction of the citation's span covered by the chunk's span.

    Uses ``max(end-start, epsilon)`` to keep zero-length citations from
    triggering a div-by-zero (and returning a spurious ratio of inf).
    """
    span = cit.end_sec - cit.start_sec
    if span <= 0:
        return 0.0
    overlap = max(0.0, min(cit.end_sec, chunk.end_sec) - max(cit.start_sec, chunk.start_sec))
    return overlap / span


def _validate_one(
    cit: Citation, reranked: list[RetrievedChunk]
) -> tuple[ValidatedCitation | None, InvalidCitation | None]:
    """Either-or: a citation is valid (matched) or invalid (no overlap)."""
    best_chunk: RetrievedChunk | None = None
    best_ratio = 0.0
    for c in reranked:
        if c.video_id != cit.video_id:
            continue
        r = _overlap_ratio(cit, c)
        if r > best_ratio:
            best_ratio = r
            best_chunk = c
    if best_chunk is not None and best_ratio >= _MIN_OVERLAP:
        return (
            ValidatedCitation(
                video_id=cit.video_id,
                start_sec=cit.start_sec,
                end_sec=cit.end_sec,
                answer_claim_index=cit.answer_claim_index,
                matched_chunk_id=best_chunk.chunk_id,
            ),
            None,
        )
    reason = (
        f"no chunk with >=50% overlap on video_id={cit.video_id} "
        f"(best ratio={best_ratio:.2f})"
    )
    return (
        None,
        InvalidCitation(
            video_id=cit.video_id,
            start_sec=cit.start_sec,
            end_sec=cit.end_sec,
            answer_claim_index=cit.answer_claim_index,
            reason=reason,
        ),
    )


def cite(state: AgentState) -> AgentState:
    """Validate citations, build the terminal ``AnswerResult``."""
    parsed = state.get("parsed")
    reranked = state.get("reranked") or []
    iteration = state.get("iteration", 0)
    trace_id = state.get("trace_id", "")
    query = state.get("query", "")
    corpus_id = state.get("corpus_id", "")

    # Abstention (parse failure OR model-emitted abstain).
    if parsed is None or not parsed.parse_ok or parsed.abstain:
        state["final"] = AnswerResult(
            trace_id=trace_id,
            query=query,
            corpus_id=corpus_id,
            answer=None,
            claims=[],
            valid_citations=[],
            invalid_citations=[],
            abstain=True,
            parse_ok=bool(parsed and parsed.parse_ok),
            iterations=iteration,
        )
        return state

    valid: list[ValidatedCitation] = []
    invalid: list[InvalidCitation] = []
    for cit in parsed.citations:
        v, iv = _validate_one(cit, reranked)
        if v is not None:
            valid.append(v)
        if iv is not None:
            invalid.append(iv)

    state["final"] = AnswerResult(
        trace_id=trace_id,
        query=query,
        corpus_id=corpus_id,
        answer=parsed.answer or None,
        claims=list(parsed.claims),
        valid_citations=valid,
        invalid_citations=invalid,
        abstain=False,
        parse_ok=True,
        iterations=iteration,
    )
    return state
