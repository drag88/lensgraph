"""Smoke tests for the four text retrieval channels.

Functional only — confirms each channel returns well-shaped results against
real ingested data. No quality assertions, no fusion, no reranker.

Requires the main lensgraph DB to be populated (via `make ingest-all`).
If empty, the suite skips with a pointer to the ingest command.
"""

from __future__ import annotations

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from retrieve import bm25, dense, multivec, sparse
from retrieve.types import ChannelResult

pytestmark = pytest.mark.slow

# Two keywords both present in the ingested corpus. BM25 (websearch_to_tsquery)
# AND-joins terms, so a long natural-language question often matches zero
# chunks even when the topic is well-covered. This pair hits multiple chunks
# under all four channels. Smoke-test only — not a quality signal.
DEV_GOLD_QUERY = "embedding retrieval"


@pytest.fixture(scope="module")
def conn():
    c = psycopg.connect(resolve_dsn(), autocommit=True)
    chunks_n = c.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if chunks_n == 0:
        c.close()
        pytest.skip(
            "main lensgraph DB has no chunks; "
            "run `make ingest-all CORPUS=ai_engineering_v0` first"
        )
    yield c
    c.close()


@pytest.fixture(scope="module")
def channels():
    """All four channel `retrieve` callables, name-keyed for parametrization."""
    return {
        "bm25": bm25.retrieve,
        "dense": dense.retrieve,
        "sparse": sparse.retrieve,
        "multivec": multivec.retrieve,
    }


@pytest.mark.parametrize("name", ["bm25", "dense", "sparse", "multivec"])
def test_channel_returns_non_empty_for_dev_gold_query(conn, channels, name):
    results = channels[name](conn, DEV_GOLD_QUERY, top_k=10)
    assert len(results) > 0, f"{name} returned no results for {DEV_GOLD_QUERY!r}"
    assert all(isinstance(r, ChannelResult) for r in results)


@pytest.mark.parametrize("name", ["bm25", "dense", "sparse", "multivec"])
def test_channel_ranks_are_1_indexed_and_monotonic(conn, channels, name):
    results = channels[name](conn, DEV_GOLD_QUERY, top_k=10)
    if not results:
        pytest.skip(f"{name} returned empty")
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


@pytest.mark.parametrize("name", ["bm25", "dense", "sparse", "multivec"])
def test_channel_result_carries_required_fields(conn, channels, name):
    results = channels[name](conn, DEV_GOLD_QUERY, top_k=5)
    if not results:
        pytest.skip(f"{name} returned empty")
    for r in results:
        assert isinstance(r.chunk_id, int) and r.chunk_id > 0
        assert isinstance(r.video_id, str) and r.video_id
        assert r.start_sec >= 0 and r.end_sec > r.start_sec
        assert isinstance(r.text, str) and len(r.text) > 0
        assert isinstance(r.score, float)
