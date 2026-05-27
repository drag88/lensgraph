"""Retrieve node — five-channel fetch + RRF fusion.

Calls each retrieval channel with ``top_k=30`` then fuses via RRF down
to ``top_k=30`` fused candidates. The downstream Rerank node collapses
that to top-8.

Source query priority: ``state['refined_query']`` (set by Verify on a
loop-back) overrides ``state['query']`` on iterations > 0. On iteration
0, the refined_query is None and the original query is used.

Synthesis with >1 sub_query: each sub_query runs all five channels, and
fuse() receives a channel dict keyed by ``"q<i>:<channel>"`` so the
fused score sums contributions across sub_queries. Single sub_query
collapses to the plain channel naming.

Visual channel gracefully returns ``[]`` when no frames have been
ingested for the corpus — :func:`retrieve.visual.retrieve` short-circuits
on an empty pooled prefilter (see ``retrieve/visual.py``).
"""

from __future__ import annotations

from typing import Any

from generate.state import AgentState, RetrievedChunk
from retrieve import bm25, dense, multivec, rrf, sparse, visual
from retrieve.types import ChannelResult

_CHANNEL_TOP_K = 30
_FUSE_TOP_K = 30
_CHANNELS: tuple[tuple[str, Any], ...] = (
    ("bm25", bm25.retrieve),
    ("dense", dense.retrieve),
    ("sparse", sparse.retrieve),
    ("multivec", multivec.retrieve),
    ("visual", visual.retrieve),
)


def _run_channels(conn, q: str) -> dict[str, list[ChannelResult]]:
    """Run all five channels for one query; return ``{name: results}``."""
    out: dict[str, list[ChannelResult]] = {}
    for name, fn in _CHANNELS:
        out[name] = fn(conn, q, top_k=_CHANNEL_TOP_K)
    return out


def retrieve(state: AgentState, *, conn) -> AgentState:
    """Run the five retrieval channels + RRF; write
    ``state['retrieved']``."""
    src_query = state.get("refined_query") or state["query"]
    sub_queries = state.get("sub_queries") or [src_query]

    # On a loop-back (refined_query set), the refinement supersedes any
    # synthesis decomposition — Verify wants its own targeted re-search.
    if state.get("refined_query"):
        sub_queries = [src_query]

    if len(sub_queries) == 1:
        channels = _run_channels(conn, sub_queries[0])
    else:
        channels = {}
        for i, sq in enumerate(sub_queries):
            for name, results in _run_channels(conn, sq).items():
                channels[f"q{i}:{name}"] = results

    fused = rrf.fuse(channels, top_k=_FUSE_TOP_K)
    state["retrieved"] = [RetrievedChunk.from_fused(r) for r in fused]
    return state
