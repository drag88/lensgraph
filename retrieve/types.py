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
