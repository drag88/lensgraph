"""Functional tests for retrieve.rerank — loads BGE-reranker-v2-m3 once.

Marked slow so the fast suite stays fast. Step 26 functional coverage:
the reranker takes a list of FusedResult, returns top_k RerankedResult
in monotonic score order, and never invents chunks outside the input
set.

The first test triggers FlagReranker download (~2.3 GB on first ever
run; cached after that). Subsequent tests reuse the same singleton.
"""

from __future__ import annotations

import pytest

from retrieve.rerank import rerank
from retrieve.types import FusedResult

pytestmark = pytest.mark.slow


def _fused(chunk_id: int, text: str, rank: int = 1, score: float = 0.0) -> FusedResult:
    return FusedResult(
        chunk_id=chunk_id,
        video_id=f"vid-{chunk_id}",
        start_sec=0.0,
        end_sec=10.0,
        text=text,
        score=score,
        rank=rank,
        channel_ranks={"bm25": rank},
    )


# A handful of real-looking text snippets. The first is the strongest
# match for "cheese" so a sane reranker should put it in the top-1.
_CANDIDATES = [
    _fused(1, "Cheddar cheese is aged in caves for at least nine months.", rank=1),
    _fused(2, "Python's GIL serializes thread access to the interpreter.", rank=2),
    _fused(3, "PostgreSQL B-tree indexes use a balanced multi-way tree.", rank=3),
    _fused(4, "Brie and camembert are soft cheeses with bloomy rinds.", rank=4),
    _fused(5, "TLS 1.3 mandates forward secrecy via ephemeral key exchange.", rank=5),
    _fused(6, "Mozzarella is traditionally made from buffalo milk.", rank=6),
    _fused(7, "Rust's borrow checker enforces single-mutable-reference at compile time.", rank=7),
    _fused(8, "Parmesan cheese is grated over pasta dishes.", rank=8),
    _fused(9, "Kafka consumer groups partition topic load across replicas.", rank=9),
    _fused(10, "Gouda matures with a wax coating to control humidity.", rank=10),
]


def test_rerank_returns_top_k_in_monotonic_score_order():
    out = rerank(conn=None, query="cheese", candidates=_CANDIDATES, top_k=5)
    assert len(out) == 5
    assert [r.rank for r in out] == [1, 2, 3, 4, 5]
    scores = [r.rerank_score for r in out]
    for prev, cur in zip(scores, scores[1:], strict=False):
        assert cur <= prev, f"rerank scores must be non-increasing, got {scores!r}"
    # Sanity: the score mirrored into .score equals rerank_score.
    assert all(r.score == r.rerank_score for r in out)


def test_rerank_preserves_input_set():
    """The top_k chunks are a subset of the input chunk_ids — no fabrication."""
    out = rerank(conn=None, query="cheese", candidates=_CANDIDATES, top_k=5)
    input_ids = {c.chunk_id for c in _CANDIDATES}
    output_ids = {r.chunk_id for r in out}
    assert output_ids.issubset(input_ids)


def test_rerank_preserves_rrf_score_provenance():
    """RerankedResult.rrf_score carries the pre-rerank FusedResult.score so
    the trace can reconstruct the re-ordering."""
    candidates = [
        _fused(101, "Cheddar cheese aged in caves.", rank=1, score=0.0833),
        _fused(102, "Python's GIL serializes threads.", rank=2, score=0.0820),
    ]
    out = rerank(conn=None, query="cheese", candidates=candidates, top_k=2)
    by_id = {r.chunk_id: r for r in out}
    assert by_id[101].rrf_score == pytest.approx(0.0833)
    assert by_id[102].rrf_score == pytest.approx(0.0820)


def test_rerank_empty_returns_empty():
    """Edge case at the slow boundary — paranoid mirror of the fast test
    in case the lazy path ever regressed without anyone noticing."""
    assert rerank(conn=None, query="anything", candidates=[]) == []


def test_rerank_smaller_than_top_k_returns_all_sorted():
    candidates = [
        _fused(201, "Cheddar cheese is aged in caves.", rank=1),
        _fused(202, "Brie has a bloomy rind.", rank=2),
        _fused(203, "Python's GIL serializes threads.", rank=3),
    ]
    out = rerank(conn=None, query="cheese", candidates=candidates, top_k=5)
    assert len(out) == 3
    assert [r.rank for r in out] == [1, 2, 3]
    scores = [r.rerank_score for r in out]
    for prev, cur in zip(scores, scores[1:], strict=False):
        assert cur <= prev


def test_rerank_carries_channel_ranks_through():
    """channel_ranks is forwarded as-is — the reranker reorders, it does
    not invent channel attribution."""
    candidates = [
        FusedResult(
            chunk_id=301,
            video_id="vid-301",
            start_sec=0.0,
            end_sec=10.0,
            text="Parmesan cheese is grated over pasta.",
            score=0.05,
            rank=1,
            channel_ranks={"bm25": 1, "dense": 4, "visual": 2},
        ),
        FusedResult(
            chunk_id=302,
            video_id="vid-302",
            start_sec=0.0,
            end_sec=10.0,
            text="Random unrelated text.",
            score=0.04,
            rank=2,
            channel_ranks={"sparse": 3},
        ),
    ]
    out = rerank(conn=None, query="cheese", candidates=candidates, top_k=2)
    by_id = {r.chunk_id: r for r in out}
    assert by_id[301].channel_ranks == {"bm25": 1, "dense": 4, "visual": 2}
    assert by_id[302].channel_ranks == {"sparse": 3}
