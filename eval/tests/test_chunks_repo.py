"""Chunks repository tests.

Slow: requires a live Postgres reachable at $POSTGRES_DSN. Spins an ephemeral
test DB per module, applies all migrations once, then exercises the repo
functions against it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

from chunking.types import Chunk
from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos import chunks as chunks_repo
from db.repos.talks import Talk
from db.repos.talks import upsert as upsert_talk

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_chunks_repo"
VIDEO_ID = "chunks-test-vid"


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
    """Per-test connection with a clean slate. TRUNCATE cascades through chunks."""
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("TRUNCATE talks CASCADE")
        _seed_talk(c)
        yield c


def _make_talk() -> Talk:
    return Talk(
        video_id=VIDEO_ID,
        title="Chunks Test Talk",
        speaker="Speaker",
        url="https://example.com/v",
        duration_sec=600,
        format_tags=["narrative"],
        license="mit",
        captions_source="youtube_auto",
        transcript_path="transcripts/chunks-test.vtt",
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _seed_talk(conn: psycopg.Connection) -> None:
    upsert_talk(conn, _make_talk())


def _make_chunk(
    start: float, end: float, text: str = "hello", strategy: str = "fixed_window"
) -> Chunk:
    return Chunk(
        video_id=VIDEO_ID,
        start_sec=start,
        end_sec=end,
        text=text,
        chunking_strategy=strategy,
        frame_secs=[start + 1.0],
        token_count=len(text.split()),
    )


def test_upsert_inserts_new_chunks_returns_ids(conn):
    inputs = [
        _make_chunk(0.0, 30.0, "first"),
        _make_chunk(30.0, 60.0, "second"),
        _make_chunk(60.0, 90.0, "third"),
    ]
    ids = chunks_repo.upsert(conn, inputs)
    assert len(ids) == 3
    assert all(isinstance(i, int) and i > 0 for i in ids)
    count = conn.execute("SELECT count(*) FROM chunks WHERE video_id = %s", (VIDEO_ID,)).fetchone()[
        0
    ]
    assert count == 3


def test_upsert_idempotent_on_same_natural_key(conn):
    ch = _make_chunk(0.0, 30.0, "original")
    first_ids = chunks_repo.upsert(conn, [ch])
    second_ids = chunks_repo.upsert(conn, [ch])
    assert first_ids == second_ids
    count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert count == 1


def test_upsert_overwrites_text_on_conflict(conn):
    chunks_repo.upsert(conn, [_make_chunk(0.0, 30.0, "v1")])
    chunks_repo.upsert(conn, [_make_chunk(0.0, 30.0, "v2")])
    text = conn.execute(
        "SELECT text FROM chunks WHERE video_id = %s AND start_sec = 0.0 AND end_sec = 30.0",
        (VIDEO_ID,),
    ).fetchone()[0]
    assert text == "v2"


def test_get_text_returns_chunk_text(conn):
    [chunk_id] = chunks_repo.upsert(conn, [_make_chunk(0.0, 30.0, "round-trip body")])
    assert chunks_repo.get_text(conn, chunk_id, VIDEO_ID) == "round-trip body"


def test_get_text_missing_returns_none(conn):
    assert chunks_repo.get_text(conn, 999_999, VIDEO_ID) is None
