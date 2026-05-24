"""Talks + ingest_step_status repository tests.

Slow: requires a live Postgres reachable at $POSTGRES_DSN. Spins an
ephemeral test DB per module, applies all migrations once, then exercises
the repo functions against it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos import ingest_step_status as iss
from db.repos.talks import Talk, get, upsert

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_talks_repo"


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
    """Per-test connection with a clean slate. TRUNCATE wipes volatile state
    so tests do not leak rows into each other."""
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("TRUNCATE ingest_step_status, talks CASCADE")
        yield c


def _make_talk(video_id: str = "rt-test", **overrides) -> Talk:
    defaults: dict = dict(
        video_id=video_id,
        title="Test Talk",
        speaker="Speaker",
        url="https://example.com/v",
        duration_sec=120,
        format_tags=["narrative", "slides_heavy"],
        license="mit",
        captions_source="youtube_auto",
        transcript_path="transcripts/test.vtt",
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Talk(**defaults)


# -- talks ----------------------------------------------------------------


def test_get_missing_returns_none(conn):
    assert get(conn, "nope") is None


def test_round_trip_required_fields_only(conn):
    talk = _make_talk()
    upsert(conn, talk)
    assert get(conn, talk.video_id) == talk


def test_round_trip_all_fields_incl_chapter_slice(conn):
    talk = _make_talk(
        video_id="chapter-test",
        source_video_id="parent-vid",
        source_start_sec=10.5,
        source_end_sec=130.5,
        notes="chapter slice with notes",
        ingested_at=datetime(2025, 1, 2, 9, 0, 0, tzinfo=UTC),
    )
    upsert(conn, talk)
    assert get(conn, talk.video_id) == talk


def test_upsert_overwrites_on_conflict(conn):
    upsert(conn, _make_talk())
    upsert(conn, _make_talk(title="Revised", duration_sec=240))
    fetched = get(conn, "rt-test")
    assert fetched is not None
    assert fetched.title == "Revised"
    assert fetched.duration_sec == 240


def test_upsert_format_tags_round_trips_array(conn):
    upsert(conn, _make_talk(format_tags=["whiteboard", "code_heavy", "live_demo"]))
    fetched = get(conn, "rt-test")
    assert fetched is not None
    assert fetched.format_tags == ["whiteboard", "code_heavy", "live_demo"]


# -- ingest_step_status ---------------------------------------------------


def test_iss_upsert_inserts_pending_row(conn):
    upsert(conn, _make_talk(video_id="iss-test"))
    iss.upsert(conn, "iss-test", step="fetch")
    row = conn.execute(
        "SELECT status, attempts, started_at, completed_at "
        "FROM ingest_step_status WHERE video_id=%s AND step=%s AND entity_id=0",
        ("iss-test", "fetch"),
    ).fetchone()
    assert row == ("pending", 0, None, None)


def test_iss_upsert_overwrites_status_only(conn):
    upsert(conn, _make_talk(video_id="iss-test"))
    iss.upsert(conn, "iss-test", step="fetch", status="failed")
    iss.upsert(conn, "iss-test", step="fetch", status="pending")
    row = conn.execute(
        "SELECT status FROM ingest_step_status WHERE video_id=%s AND step=%s",
        ("iss-test", "fetch"),
    ).fetchone()
    assert row[0] == "pending"


def test_iss_claim_transitions_pending_to_in_progress(conn):
    for v in ("a", "b", "c"):
        upsert(conn, _make_talk(video_id=v))
        iss.upsert(conn, v, step="fetch")
    claimed = iss.claim_for_update(conn, "fetch", batch_size=2)
    assert len(claimed) == 2
    for video_id, _ in claimed:
        row = conn.execute(
            "SELECT status, attempts, started_at FROM ingest_step_status "
            "WHERE video_id=%s AND step='fetch' AND entity_id=0",
            (video_id,),
        ).fetchone()
        status, attempts, started_at = row
        assert status == "in_progress"
        assert attempts == 1
        assert started_at is not None


def test_iss_claim_does_not_double_claim_in_progress(conn):
    for v in ("a", "b", "c"):
        upsert(conn, _make_talk(video_id=v))
        iss.upsert(conn, v, step="fetch")
    first = iss.claim_for_update(conn, "fetch", batch_size=2)
    second = iss.claim_for_update(conn, "fetch", batch_size=10)
    assert len(first) == 2
    assert len(second) == 1
    assert {v for v, _ in first}.isdisjoint({v for v, _ in second})


def test_iss_mark_completed_sets_completed_at(conn):
    upsert(conn, _make_talk(video_id="iss-test"))
    iss.upsert(conn, "iss-test", step="fetch")
    iss.claim_for_update(conn, "fetch", batch_size=1)
    iss.mark_completed(conn, "iss-test", step="fetch")
    row = conn.execute(
        "SELECT status, completed_at FROM ingest_step_status "
        "WHERE video_id=%s AND step='fetch' AND entity_id=0",
        ("iss-test",),
    ).fetchone()
    assert row[0] == "completed"
    assert row[1] is not None


def test_iss_mark_failed_persists_error_payload(conn):
    upsert(conn, _make_talk(video_id="iss-test"))
    iss.upsert(conn, "iss-test", step="fetch")
    iss.claim_for_update(conn, "fetch", batch_size=1)
    payload = {"reason": "yt-dlp 404", "attempt_count": 1}
    iss.mark_failed(conn, "iss-test", step="fetch", error_payload=payload)
    row = conn.execute(
        "SELECT status, completed_at, error_payload FROM ingest_step_status "
        "WHERE video_id=%s AND step='fetch' AND entity_id=0",
        ("iss-test",),
    ).fetchone()
    assert row[0] == "failed"
    assert row[1] is not None
    assert row[2] == payload


def test_iss_entity_id_fanout_for_per_chunk_steps(conn):
    """Fan-out steps (embed_text per chunk) use entity_id != 0."""
    upsert(conn, _make_talk(video_id="iss-test"))
    iss.upsert(conn, "iss-test", step="embed_text", entity_id=1)
    iss.upsert(conn, "iss-test", step="embed_text", entity_id=2)
    rows = conn.execute(
        "SELECT entity_id, status FROM ingest_step_status "
        "WHERE video_id=%s AND step='embed_text' ORDER BY entity_id",
        ("iss-test",),
    ).fetchall()
    assert rows == [(1, "pending"), (2, "pending")]
