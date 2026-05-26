"""Catastrophic-regression smoke for retrieve.rerank against real data.

Step 26 calls for a tight single-example assertion that would catch a
catastrophic rerank bug (reverse-order, off-by-one, dropping the gold)
without needing to publish a full eval run. The test picks the
``tengyu-rag-library-analogy`` example from the committed dev_gold
corpus, runs the full RRF pipeline (BM25 + dense + sparse + multivec +
visual stage 1+2 where available), enumerates every chunk whose span
overlaps the gold span, then asserts:

  1. At least one gold-overlapping chunk is in the top-5 of the RRF
     fused output.
  2. At least one gold-overlapping chunk is STILL in the top-5 of the
     reranked output.

Why "at least one of the set" rather than "a single canonical chunk":
fixed-window chunking with 5s overlap turns a 118s gold span into 5-6
overlapping chunks, and the chunk with the largest IoU is not always
the chunk with the strongest semantic match — adjacent chunks may
contain the analogy's anchor sentence verbatim while the max-IoU chunk
spans the wind-down of the previous topic. The catastrophic-regression
case is "NONE of the gold-overlapping chunks reach top-5", which would
indicate retrieval is fundamentally broken; ranking any one of them at
position 1 is healthy behavior, not a regression.

Both halves still matter: (1) catches a retrieval regression upstream
of the reranker; (2) catches the reranker silently re-ordering all
gold-overlapping chunks out of the top-5.

Skips cleanly if the corpus is not populated — slice 2 can ship with
this test skipped, but the assertion logic is in place for the moment
the corpus is loaded. Visual channel is included only if `frames` AND
`frame_patches` are populated; otherwise the smoke uses the four text
channels (still a valid catastrophic-regression detector — the gold
example is a transcript-only question by modality).
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from retrieve import bm25, dense, multivec, sparse, visual
from retrieve.rerank import rerank
from retrieve.rrf import fuse

pytestmark = pytest.mark.slow

DEV_GOLD_PATH = Path("eval/corpora/ai_engineering_v0/dev_gold.jsonl")
TARGET_EXAMPLE_ID = "tengyu-rag-library-analogy"


@pytest.fixture(scope="module")
def conn():
    c = psycopg.connect(resolve_dsn(), autocommit=True)
    chunks_n = c.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if chunks_n == 0:
        c.close()
        pytest.skip(
            "lensgraph DB has no chunks; run "
            "`make ingest-all CORPUS=ai_engineering_v0` first"
        )
    yield c
    c.close()


@pytest.fixture(scope="module")
def gold_example() -> dict:
    """Load the target example from dev_gold. Fail loudly if it is gone —
    a rename in the corpus invalidates the smoke and demands a deliberate
    revisit (see CLAUDE.md hard rule 2 / data contracts for the verified
    flag invariant)."""
    for line in DEV_GOLD_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        if ex.get("id") == TARGET_EXAMPLE_ID:
            return ex
    pytest.fail(
        f"dev_gold no longer contains {TARGET_EXAMPLE_ID!r}; "
        f"update test_rerank_smoke.py to a new W_CYk2ogcDI single-clip example."
    )


def _gold_chunk_ids(conn, ex: dict) -> set[int]:
    """Return every chunk_id whose [start_sec, end_sec) span has non-zero
    overlap with the example's gold span. Empty set means the corpus is
    not chunked for this video — the caller should skip."""
    span = ex["gold_spans"][0]
    video_id = ex["video_id"]
    g_start = float(span["start_sec"])
    g_end = float(span["end_sec"])
    rows = conn.execute(
        """
        SELECT chunk_id
        FROM chunks
        WHERE video_id = %s
          AND end_sec > %s::real
          AND start_sec < %s::real
        """,
        (video_id, g_start, g_end),
    ).fetchall()
    return {int(r[0]) for r in rows}


def _has_visual_data(conn) -> bool:
    """Visual channel is only meaningful when frames AND frame_patches are
    populated. The pooled column alone (slice 1) is not enough for
    `retrieve.visual.retrieve()` to score chunks."""
    frames_n = conn.execute(
        "SELECT count(*) FROM frames WHERE pooled_embedding IS NOT NULL"
    ).fetchone()[0]
    patches_n = conn.execute("SELECT count(*) FROM frame_patches").fetchone()[0]
    return frames_n > 0 and patches_n > 0


def test_known_gold_chunk_in_top_5_pre_and_post_rerank(conn, gold_example):
    gold_chunk_ids = _gold_chunk_ids(conn, gold_example)
    if not gold_chunk_ids:
        pytest.skip(
            f"no chunk overlaps gold span for {gold_example['id']!r} "
            f"(video {gold_example['video_id']!r}); chunking not yet run for this video"
        )

    question = gold_example["question"]

    channels = {
        "bm25": bm25.retrieve(conn, question, top_k=30),
        "dense": dense.retrieve(conn, question, top_k=30),
        "sparse": sparse.retrieve(conn, question, top_k=30),
        "multivec": multivec.retrieve(conn, question, top_k=30),
    }
    if _has_visual_data(conn):
        channels["visual"] = visual.retrieve(conn, question, top_k=30)

    fused = fuse(channels, top_k=8)
    pre_top5_ids = [r.chunk_id for r in fused[:5]]
    pre_hits = set(pre_top5_ids) & gold_chunk_ids
    assert pre_hits, (
        f"no gold-overlapping chunk reached the top-5 of the RRF fused "
        f"output (pre-rerank). Gold-overlapping chunks: "
        f"{sorted(gold_chunk_ids)!r}. Got top-5: {pre_top5_ids!r}. "
        f"This indicates a retrieval regression upstream of the reranker."
    )

    reranked = rerank(conn, question, fused, top_k=8)
    post_top5_ids = [r.chunk_id for r in reranked[:5]]
    post_hits = set(post_top5_ids) & gold_chunk_ids
    assert post_hits, (
        f"gold-overlapping chunks present pre-rerank ({sorted(pre_hits)!r}) "
        f"but ALL DROPPED from top-5 post-rerank. Pre-rerank top-5: "
        f"{pre_top5_ids!r}; post-rerank top-5: {post_top5_ids!r}. "
        f"This is the catastrophic-regression case step 26 is built to catch."
    )
