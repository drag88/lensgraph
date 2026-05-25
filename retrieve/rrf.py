"""Reciprocal Rank Fusion over per-channel ChannelResult lists.

RRF combines independent retrieval rankings into one fused ranking
without requiring score normalisation across heterogeneous channels
(BM25 ts_rank, cosine, inner-product, MaxSim). Each channel contributes
1 / (k + rank) per chunk it ranks; missing-from-channel = 0 contribution.

Returns FusedResult instances with `score` set to the fused RRF score,
`rank` set to the 1-indexed position in the fused output, and
`channel_ranks` preserving the exact per-channel input ranks for the
channels that contributed to the chunk's fusion.
"""

from __future__ import annotations

from collections.abc import Mapping

from retrieve.types import ChannelResult, FusedResult

RRF_K = 60


def fuse(
    channels: Mapping[str, list[ChannelResult]],
    *,
    k: int = RRF_K,
    top_k: int = 8,
) -> list[FusedResult]:
    """Fuse per-channel results via RRF. Returns top_k fused chunks.

    `channels`: {channel_name: per-channel ranked list}. Channel name is
        informational — RRF does not weight or distinguish channels by
        name. Empty per-channel lists are tolerated; missing channel names
        likewise (a chunk seen in only one channel just gets one
        contribution).

    `k`: RRF constant. Standard 60 per Cormack et al. (2009). Larger k
        flattens the contribution curve (early ranks matter less); smaller
        k sharpens it. Don't tune this without an eval run.

    `top_k`: maximum fused results returned. Tie-broken by chunk_id for
        determinism.

    Algorithm:
      1. Walk every channel's results; accumulate per-chunk rank dict.
      2. Sum 1/(k+rank) over each chunk's ranks.
      3. Sort by score DESC, chunk_id ASC.
      4. Truncate to top_k; assign 1-indexed fused rank.
    """
    if not channels:
        return []

    by_chunk: dict[int, dict] = {}
    for channel_name, results in channels.items():
        for r in results:
            entry = by_chunk.setdefault(r.chunk_id, {"meta": r, "ranks_by_channel": {}})
            entry["ranks_by_channel"][channel_name] = r.rank

    scored: list[tuple[float, int, ChannelResult, dict[str, int]]] = []
    for chunk_id, entry in by_chunk.items():
        rrf_score = sum(1.0 / (k + rank) for rank in entry["ranks_by_channel"].values())
        scored.append((rrf_score, chunk_id, entry["meta"], entry["ranks_by_channel"]))

    scored.sort(key=lambda x: (-x[0], x[1]))

    return [
        FusedResult(
            chunk_id=meta.chunk_id,
            video_id=meta.video_id,
            start_sec=meta.start_sec,
            end_sec=meta.end_sec,
            text=meta.text,
            score=score,
            rank=i + 1,
            channel_ranks=dict(ranks_by_channel),
        )
        for i, (score, _chunk_id, meta, ranks_by_channel) in enumerate(scored[:top_k])
    ]
