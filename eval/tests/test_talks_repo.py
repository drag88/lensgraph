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


def _setup_pending(conn, video_id: str, step: str = "fetch", entity_id: int = 0):
    """Ensure a talk + a pending status row exist for the given identity."""
    upsert(conn, _make_talk(video_id=video_id))
    iss.upsert(conn, video_id, step=step, entity_id=entity_id)


def _read_row(conn, video_id: str, step: str = "fetch", entity_id: int = 0):
    return conn.execute(
        "SELECT status, attempts, started_at, completed_at, error_payload "
        "FROM ingest_step_status WHERE video_id=%s AND step=%s AND entity_id=%s",
        (video_id, step, entity_id),
    ).fetchone()


def test_iss_upsert_inserts_pending_row(conn):
    _setup_pending(conn, "iss-test")
    row = _read_row(conn, "iss-test")
    status, attempts, started_at, completed_at, error_payload = row
    assert status == "pending"
    assert attempts == 0
    assert started_at is None
    assert completed_at is None
    assert error_payload is None


def test_iss_upsert_overwrites_status_only(conn):
    _setup_pending(conn, "iss-test")
    iss.upsert(conn, "iss-test", step="fetch", status="failed")
    iss.upsert(conn, "iss-test", step="fetch", status="pending")
    assert _read_row(conn, "iss-test")[0] == "pending"


# claim_for_update — exact-row identity contract


def test_iss_claim_missing_row_raises(conn):
    """Worker dequeued a payload referencing a row that doesn't exist —
    contract violation, raise rather than silently succeed."""
    with conn.transaction(), pytest.raises(iss.IngestStepNotFoundError):
        iss.claim_for_update(conn, "ghost-vid", step="fetch")


def test_iss_claim_pending_transitions_to_in_progress(conn):
    _setup_pending(conn, "iss-test")
    with conn.transaction():
        result = iss.claim_for_update(conn, "iss-test", step="fetch")
    assert result == iss.ClaimResult(status="in_progress", attempts=1)
    status, attempts, started_at, _, _ = _read_row(conn, "iss-test")
    assert status == "in_progress"
    assert attempts == 1
    assert started_at is not None


def test_iss_claim_on_completed_returns_without_increment(conn):
    """A re-delivered PGMQ job for already-completed work must NOT re-run."""
    _setup_pending(conn, "iss-test")
    iss.upsert(conn, "iss-test", step="fetch", status="completed")
    before = _read_row(conn, "iss-test")
    with conn.transaction():
        result = iss.claim_for_update(conn, "iss-test", step="fetch")
    after = _read_row(conn, "iss-test")
    assert result.status == "completed"
    assert result.attempts == before[1]
    assert after[1] == before[1]  # attempts unchanged
    assert after[0] == "completed"  # status unchanged


def test_iss_claim_on_skipped_returns_without_increment(conn):
    """Skipped (e.g. asr on a video with manual captions) — same as completed."""
    _setup_pending(conn, "iss-test")
    iss.upsert(conn, "iss-test", step="fetch", status="skipped")
    with conn.transaction():
        result = iss.claim_for_update(conn, "iss-test", step="fetch")
    assert result.status == "skipped"
    assert _read_row(conn, "iss-test")[0] == "skipped"


def test_iss_claim_on_failed_transitions_to_in_progress(conn):
    """Failed rows are eligible for retry: claim transitions + bumps attempts."""
    _setup_pending(conn, "iss-test")
    iss.upsert(conn, "iss-test", step="fetch", status="failed")
    with conn.transaction():
        result = iss.claim_for_update(conn, "iss-test", step="fetch")
    assert result.status == "in_progress"
    assert result.attempts == 1
    assert _read_row(conn, "iss-test")[0] == "in_progress"


def test_iss_claim_on_in_progress_re_transitions_for_redelivery(conn):
    """PGMQ visibility timeout expiry + worker crash leaves status in_progress.
    A subsequent claim must re-transition and increment attempts."""
    _setup_pending(conn, "iss-test")
    with conn.transaction():
        first = iss.claim_for_update(conn, "iss-test", step="fetch")
    assert first.attempts == 1
    with conn.transaction():
        second = iss.claim_for_update(conn, "iss-test", step="fetch")
    assert second.status == "in_progress"
    assert second.attempts == 2
    assert _read_row(conn, "iss-test")[1] == 2


def test_iss_claim_for_video_b_does_not_touch_video_a(conn):
    """The job-identity contract: claiming B's row must leave A untouched.

    Regression check for the pre-fix bug where claim_for_update picked
    'some pending row of step X' — under contention, a worker handling B's
    job could transition A's row instead.
    """
    for v in ("vid-a", "vid-b"):
        _setup_pending(conn, v)
    with conn.transaction():
        result = iss.claim_for_update(conn, "vid-b", step="fetch")
    assert result.status == "in_progress"
    a_status, a_attempts, a_started, _, _ = _read_row(conn, "vid-a")
    b_status, b_attempts, b_started, _, _ = _read_row(conn, "vid-b")
    assert (a_status, a_attempts, a_started) == ("pending", 0, None)
    assert b_status == "in_progress"
    assert b_attempts == 1
    assert b_started is not None


# mark_completed / mark_failed — only transition in_progress rows


def test_iss_mark_completed_transitions_in_progress(conn):
    _setup_pending(conn, "iss-test")
    with conn.transaction():
        iss.claim_for_update(conn, "iss-test", step="fetch")
    assert iss.mark_completed(conn, "iss-test", step="fetch") is True
    status, _, _, completed_at, _ = _read_row(conn, "iss-test")
    assert status == "completed"
    assert completed_at is not None


def test_iss_mark_completed_no_op_when_not_in_progress(conn):
    """Calling mark_completed on a pending row is a no-op — returns False;
    the row stays pending."""
    _setup_pending(conn, "iss-test")
    assert iss.mark_completed(conn, "iss-test", step="fetch") is False
    status, _, _, completed_at, _ = _read_row(conn, "iss-test")
    assert status == "pending"
    assert completed_at is None


def test_iss_mark_failed_persists_error_payload(conn):
    _setup_pending(conn, "iss-test")
    with conn.transaction():
        iss.claim_for_update(conn, "iss-test", step="fetch")
    payload = {"reason": "yt-dlp 404", "attempt_count": 1}
    assert iss.mark_failed(conn, "iss-test", step="fetch", error_payload=payload) is True
    status, _, _, completed_at, error_payload = _read_row(conn, "iss-test")
    assert status == "failed"
    assert completed_at is not None
    assert error_payload == payload


def test_iss_mark_failed_no_op_when_not_in_progress(conn):
    _setup_pending(conn, "iss-test")
    assert iss.mark_failed(conn, "iss-test", step="fetch", error_payload={"x": 1}) is False
    status, _, _, _, error_payload = _read_row(conn, "iss-test")
    assert status == "pending"
    assert error_payload is None


def test_iss_entity_id_fanout_for_per_chunk_steps(conn):
    """Fan-out steps (embed_text per chunk) use entity_id != 0 — each row is
    independently claimable."""
    upsert(conn, _make_talk(video_id="iss-test"))
    iss.upsert(conn, "iss-test", step="embed_text", entity_id=1)
    iss.upsert(conn, "iss-test", step="embed_text", entity_id=2)
    with conn.transaction():
        result = iss.claim_for_update(conn, "iss-test", step="embed_text", entity_id=1)
    assert result.status == "in_progress"
    # entity_id=2 untouched
    assert _read_row(conn, "iss-test", step="embed_text", entity_id=2)[0] == "pending"
