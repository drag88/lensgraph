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
from db.repos import ingest_step_status as iss
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
        # `talks CASCADE` would clean frames + ingest_step_status via FK,
        # but listing both explicitly documents what each test touches.
        c.execute(
            "TRUNCATE chunk_token_embeds, sparse_embeds, dense_embeds, "
            "chunks, frames, ingest_step_status, talks CASCADE"
        )
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


# -- frame_sample status (visual readiness, informational only) ------------


def _insert_frames(conn: psycopg.Connection, video_id: str, n: int) -> None:
    """Insert n frame rows for video_id via raw SQL. pooled_embedding stays NULL."""
    for i in range(n):
        conn.execute(
            """
            INSERT INTO frames (video_id, frame_sec, image_path, sha256)
            VALUES (%s, %s, %s, %s)
            """,
            (video_id, float(i), f"/tmp/{video_id}/{i:04d}.jpg", f"{i:064x}"),
        )


def test_readiness_frame_sample_status_none_when_no_row(conn):
    """No ingest_step_status row → status is None, frames_n == 0."""
    upsert_talk(conn, _make_talk("vid-no-frame-status"))

    r = for_video(conn, "vid-no-frame-status")
    assert r.frame_sample_status is None
    assert r.frames_n == 0


def test_readiness_frame_sample_status_pending(conn):
    upsert_talk(conn, _make_talk("vid-frame-pending"))
    iss.upsert(conn, "vid-frame-pending", step="frame_sample", status="pending")

    r = for_video(conn, "vid-frame-pending")
    assert r.frame_sample_status == "pending"
    assert r.frames_n == 0


def test_readiness_frame_sample_status_skipped(conn):
    upsert_talk(conn, _make_talk("vid-frame-skipped"))
    iss.upsert(conn, "vid-frame-skipped", step="frame_sample", status="skipped")

    r = for_video(conn, "vid-frame-skipped")
    assert r.frame_sample_status == "skipped"
    assert r.frames_n == 0


def test_readiness_frame_sample_status_completed_with_frame_count(conn):
    upsert_talk(conn, _make_talk("vid-frame-done"))
    iss.upsert(conn, "vid-frame-done", step="frame_sample", status="completed")
    _insert_frames(conn, "vid-frame-done", 7)

    r = for_video(conn, "vid-frame-done")
    assert r.frame_sample_status == "completed"
    assert r.frames_n == 7


def test_readiness_frame_sample_does_not_affect_complete_property(conn):
    """Load-bearing decoupling: visual status MUST NOT gate text readiness.
    If text channels are all caught up, `complete` stays True even when the
    frame sampler has not finished — or has not started."""
    upsert_talk(conn, _make_talk("vid-text-done-frames-pending"))
    ids = _insert_chunks(conn, "vid-text-done-frames-pending", 3)
    _insert_dense(conn, ids)
    _insert_sparse(conn, ids)
    _insert_tokens(conn, ids)
    iss.upsert(conn, "vid-text-done-frames-pending", step="frame_sample", status="pending")

    r = for_video(conn, "vid-text-done-frames-pending")
    assert r.frame_sample_status == "pending"
    assert r.frames_n == 0
    assert r.complete is True
