"""Shared types for the retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelResult:
    """One retrieval result from a single channel.

    `score` semantics depend on the channel (cosine similarity for dense,
    inner-product for sparse, ts_rank for bm25, MaxSim for multi-vector)
    but `rank` is always 1-indexed.
    """

    chunk_id: int
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    score: float
    rank: int  # 1-indexed


@dataclass(frozen=True)
class FusedResult:
    """Output of retrieve.rrf.fuse(). Same metadata shape as
    ChannelResult plus the per-channel rank provenance that motivated
    the chunk's inclusion in the fused list.

    `channel_ranks` contains exactly the channels that ranked this chunk
    (the channels in the fuse() input whose result list included it).
    Channels that did NOT rank the chunk are ABSENT from the dict — never
    represented as 0, None, or sentinel values. Consumers (Verify node,
    trace logging) rely on dict.keys() being the contributing channel set.

    Metadata (video_id, start_sec, end_sec, text) is sourced from the
    FIRST-SEEN ChannelResult for a given chunk_id across the input
    channels. The chunks-table primary key guarantees these fields are
    identical across channels for the same chunk_id, so the choice is
    cosmetic — documented here for posterity.
    """

    chunk_id: int
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    score: float  # fused RRF score
    rank: int  # 1-indexed position in fused list
    channel_ranks: dict[str, int]  # channel_name -> rank in that channel


@dataclass(frozen=True)
class RerankedResult:
    """Output of retrieve.rerank.rerank(). Carries the cross-encoder score
    in `rerank_score` and preserves the pre-rerank RRF score in `rrf_score`
    so the trace can show how the reranker re-ordered (or didn't re-order)
    the fused list.

    `score` is set to `rerank_score` for parallelism with the other result
    types (callers that only want "the ranking score" stay polymorphic);
    `rank` is the 1-indexed position in the reranked output.

    `channel_ranks` is forwarded as-is from the input FusedResult — the
    reranker does not invent channel attribution; it only reorders.
    """

    chunk_id: int
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    score: float  # == rerank_score; mirrored for callers that handle multiple result shapes
    rank: int  # 1-indexed position in reranked list
    channel_ranks: dict[str, int]
    rerank_score: float  # cross-encoder sigmoid in [0, 1]
    rrf_score: float  # pre-rerank fused score, preserved for trace
