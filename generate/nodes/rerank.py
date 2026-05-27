"""Rerank node — collapse fused candidates to top-8 via BGE cross-encoder.

``retrieve.rerank.rerank`` strictly expects ``list[FusedResult]``. The
LangGraph state carries ``RetrievedChunk`` (a Pydantic mirror of
``FusedResult`` + post-rerank fields) so we round-trip back to
``FusedResult`` here. The mapping is one-to-one for the fields the
reranker cares about (text + channel_ranks for trace provenance).
"""

from __future__ import annotations

from generate.state import AgentState, RetrievedChunk
from retrieve import rerank as rerank_mod
from retrieve.types import FusedResult

_TOP_K = 8


def _to_fused(rc: RetrievedChunk) -> FusedResult:
    """Mirror a ``RetrievedChunk`` back into ``FusedResult`` for the
    reranker. ``score`` re-uses ``rrf_score`` when present (Retrieve set
    it from the RRF pass) else the plain ``score``."""
    rrf_score = rc.rrf_score if rc.rrf_score is not None else rc.score
    return FusedResult(
        chunk_id=rc.chunk_id,
        video_id=rc.video_id,
        start_sec=rc.start_sec,
        end_sec=rc.end_sec,
        text=rc.text,
        score=rrf_score,
        rank=rc.rank,
        channel_ranks=dict(rc.channel_ranks),
    )


def rerank(state: AgentState, *, conn) -> AgentState:
    """Cross-encoder rerank the retrieved candidates; keep top-8."""
    retrieved = state.get("retrieved") or []
    fused = [_to_fused(rc) for rc in retrieved]
    query = state.get("refined_query") or state["query"]
    reranked = rerank_mod.rerank(conn, query, fused, top_k=_TOP_K)
    state["reranked"] = [RetrievedChunk.from_reranked(r) for r in reranked]
    return state
