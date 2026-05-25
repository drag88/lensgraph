"""Per-video readiness tests against real Postgres.

Slow: requires live Postgres reachable at $POSTGRES_DSN. Per-module
ephemeral DB with migrations pre-applied. Per-test TRUNCATE for isolation.

The regression these tests guard against: bakeoff_prep used to report a
channel "ready" if its rowcount was > 0, hiding partial coverage. The new
contract is per-chunk strictness — a channel is ready iff its rowcount
equals chunks_n.

Tests insert partial state via direct SQL (not the repos) because the
repos enforce the consistency we're deliberately breaking here.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import psycopg
import pytest
from pgvector import SparseVector
from pgvector.psycopg import register_vector

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos.talks import Talk
from db.repos.talks import upsert as upsert_talk
from embed.bge_m3 import VOCAB_SIZE
from ingest.readiness import for_video

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_readiness"


def _swap_db(dsn: str, new_db: str) -> str:
    head, _, _ = dsn.rpartition("/")
    return f"{head}/{new_db}"


@pytest.fixture(scope="module")
def test_db():
    admin = resolve_dsn()
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        c.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    test_dsn = _swap_db(admin, TEST_DB_NAME)
    apply(test_dsn)
    yield test_dsn
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")


@pytest.fixture
def conn(test_db):
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("TRUNCATE chunk_token_embeds, sparse_embeds, dense_embeds, chunks, talks CASCADE")
        register_vector(c)
        yield c


def _make_talk(video_id: str) -> Talk:
    return Talk(
        video_id=video_id,
        title="Readiness test",
        speaker="S",
        url="https://example.com/v",
        duration_sec=30,
        format_tags=["narrative"],
        license="cc-by",
        captions_source="manual_transcript",
        transcript_path="transcripts/nope.vtt",
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _insert_chunks(conn: psycopg.Connection, video_id: str, n: int) -> list[int]:
    """Insert n chunks for video_id at non-overlapping spans. Returns chunk_ids."""
    ids: list[int] = []
    for i in range(n):
        row = conn.execute(
            """
            INSERT INTO chunks (
                video_id, chunking_strategy, start_sec, end_sec,
                text, token_count, frame_secs
            )
            VALUES (%s, 'fixed_window', %s, %s, %s, 3, '{}')
            RETURNING chunk_id
            """,
            (video_id, float(i * 10), float(i * 10 + 5), f"chunk text {i}"),
        ).fetchone()
        ids.append(int(row[0]))
    return ids


def _insert_dense(conn: psycopg.Connection, chunk_ids: list[int]) -> None:
    vec = np.zeros(1024, dtype=np.float32)
    for cid in chunk_ids:
        conn.execute(
            "INSERT INTO dense_embeds (chunk_id, embedding) VALUES (%s, %s)",
            (cid, vec),
        )


def _insert_sparse(conn: psycopg.Connection, chunk_ids: list[int]) -> None:
    sv = SparseVector({0: 1.0}, VOCAB_SIZE)
    for cid in chunk_ids:
        conn.execute(
            "INSERT INTO sparse_embeds (chunk_id, embedding) VALUES (%s, %s)",
            (cid, sv),
        )


def _insert_tokens(conn: psycopg.Connection, chunk_ids: list[int]) -> None:
    """Insert two token rows per chunk_id so DISTINCT collapses to len(chunk_ids)."""
    vec = np.zeros(1024, dtype=np.float32)
    for cid in chunk_ids:
        conn.execute(
            "INSERT INTO chunk_token_embeds (chunk_id, position, embedding) "
            "VALUES (%s, 0, %s), (%s, 1, %s)",
            (cid, vec, cid, vec),
        )


# -- happy path -----------------------------------------------------------


def test_readiness_complete_when_all_channels_match_chunks_n(conn):
    upsert_talk(conn, _make_talk("vid-ok"))
    ids = _insert_chunks(conn, "vid-ok", 3)
    _insert_dense(conn, ids)
    _insert_sparse(conn, ids)
    _insert_tokens(conn, ids)

    r = for_video(conn, "vid-ok")
    assert r.talks_present is True
    assert r.chunks_n == 3
    assert r.dense_n == 3
    assert r.sparse_n == 3
    assert r.tokens_n == 3
    assert r.complete is True


# -- partial coverage regressions -----------------------------------------


def test_partial_dense_reports_incomplete(conn):
    """The regression the user called out: 2 dense rows against 3 chunks must
    NOT report ready. Old boolean formulation said ✓ here."""
    upsert_talk(conn, _make_talk("vid-partial-dense"))
    ids = _insert_chunks(conn, "vid-partial-dense", 3)
    _insert_dense(conn, ids[:2])
    _insert_sparse(conn, ids)
    _insert_tokens(conn, ids)

    r = for_video(conn, "vid-partial-dense")
    assert r.chunks_n == 3
    assert r.dense_n == 2
    assert r.dense_n < r.chunks_n
    assert r.complete is False


def test_partial_sparse_reports_incomplete(conn):
    upsert_talk(conn, _make_talk("vid-partial-sparse"))
    ids = _insert_chunks(conn, "vid-partial-sparse", 3)
    _insert_dense(conn, ids)
    _insert_sparse(conn, ids[:2])
    _insert_tokens(conn, ids)

    r = for_video(conn, "vid-partial-sparse")
    assert r.chunks_n == 3
    assert r.sparse_n == 2
    assert r.complete is False


def test_partial_tokens_reports_incomplete(conn):
    """tokens_n is DISTINCT chunk_id count, not raw row count — so 2 chunks
    with token rows still reads as tokens_n=2 vs chunks_n=3."""
    upsert_talk(conn, _make_talk("vid-partial-tokens"))
    ids = _insert_chunks(conn, "vid-partial-tokens", 3)
    _insert_dense(conn, ids)
    _insert_sparse(conn, ids)
    _insert_tokens(conn, ids[:2])

    r = for_video(conn, "vid-partial-tokens")
    assert r.chunks_n == 3
    assert r.tokens_n == 2
    assert r.complete is False


# -- empty / missing ------------------------------------------------------


def test_zero_chunks_reports_incomplete(conn):
    upsert_talk(conn, _make_talk("vid-no-chunks"))

    r = for_video(conn, "vid-no-chunks")
    assert r.talks_present is True
    assert r.chunks_n == 0
    assert r.dense_n == 0
    assert r.sparse_n == 0
    assert r.tokens_n == 0
    assert r.complete is False


def test_missing_talk_reports_incomplete(conn):
    r = for_video(conn, "vid-ghost")
    assert r.talks_present is False
    assert r.chunks_n == 0
    assert r.complete is False
