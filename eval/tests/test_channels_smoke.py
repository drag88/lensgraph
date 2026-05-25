"""Smoke tests for the four text retrieval channels.

Functional only — confirms each channel returns well-shaped, monotonically
ranked results against real ingested data. No quality assertions, no
fusion, no reranker.

Two query shapes exercise each channel:

- A real ``dev_gold`` natural-language question (primary). This is the
  regression test for the BM25 AND-zeroes-out bug: long lemmatised
  questions used to match zero chunks under ``websearch_to_tsquery``;
  the OR fallback in ``retrieve.bm25`` is what keeps this passing.

- A hand-picked two-keyword phrase (secondary). Both terms exist in the
  corpus and hit multiple chunks under every channel.

Requires the main lensgraph DB to be populated (via `make ingest-all`).
If empty, the suite skips with a pointer to the ingest command.
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from retrieve import bm25, dense, multivec, sparse
from retrieve.types import ChannelResult

pytestmark = pytest.mark.slow

DEV_GOLD_PATH = Path("eval/corpora/ai_engineering_v0/dev_gold.jsonl")

# Two keywords both present in the ingested corpus. Kept as a secondary
# smoke against the keyword path of each channel.
KEYWORD_QUERY = "embedding retrieval"

CHANNELS = ["bm25", "dense", "sparse", "multivec"]


@pytest.fixture(scope="module")
def conn():
    c = psycopg.connect(resolve_dsn(), autocommit=True)
    chunks_n = c.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if chunks_n == 0:
        c.close()
        pytest.skip(
            "main lensgraph DB has no chunks; run `make ingest-all CORPUS=ai_engineering_v0` first"
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


@pytest.fixture(scope="module")
def dev_gold_question() -> str:
    """First non-empty question from the committed dev_gold corpus.

    Loaded fresh on every run so a curation update flows into the smoke
    suite without test edits. If the file is missing or empty, fail
    loudly — the corpus is a project invariant.
    """
    lines = [ln for ln in DEV_GOLD_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        pytest.fail(f"{DEV_GOLD_PATH} has no examples")
    q = json.loads(lines[0])["question"]
    if not q:
        pytest.fail("first dev_gold example has empty question")
    return q


@pytest.mark.parametrize("name", CHANNELS)
def test_channel_returns_non_empty_for_real_dev_gold_question(
    conn, channels, dev_gold_question, name
):
    results = channels[name](conn, dev_gold_question, top_k=10)
    assert len(results) > 0, (
        f"{name} returned no results for real dev_gold question {dev_gold_question!r}"
    )
    assert all(isinstance(r, ChannelResult) for r in results)


@pytest.mark.parametrize("name", CHANNELS)
def test_channel_returns_non_empty_for_keyword_query(conn, channels, name):
    results = channels[name](conn, KEYWORD_QUERY, top_k=10)
    assert len(results) > 0, f"{name} returned no results for {KEYWORD_QUERY!r}"
    assert all(isinstance(r, ChannelResult) for r in results)


@pytest.mark.parametrize("name", CHANNELS)
def test_channel_ranks_are_1_indexed_and_monotonic(conn, channels, name):
    results = channels[name](conn, KEYWORD_QUERY, top_k=10)
    if not results:
        pytest.skip(f"{name} returned empty")
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


@pytest.mark.parametrize("name", CHANNELS)
def test_channel_scores_non_increasing_by_rank(conn, channels, name):
    results = channels[name](conn, KEYWORD_QUERY, top_k=10)
    if not results:
        pytest.skip(f"{name} returned empty")
    scores = [r.score for r in results]
    for prev, cur in zip(scores, scores[1:], strict=False):
        assert cur <= prev, f"{name} scores must be non-increasing by rank, got {scores!r}"


@pytest.mark.parametrize("name", CHANNELS)
def test_channel_result_carries_required_fields(conn, channels, name):
    results = channels[name](conn, KEYWORD_QUERY, top_k=5)
    if not results:
        pytest.skip(f"{name} returned empty")
    for r in results:
        assert isinstance(r.chunk_id, int) and r.chunk_id > 0
        assert isinstance(r.video_id, str) and r.video_id
        assert r.start_sec >= 0 and r.end_sec > r.start_sec
        assert isinstance(r.text, str) and len(r.text) > 0
        assert isinstance(r.score, float)
