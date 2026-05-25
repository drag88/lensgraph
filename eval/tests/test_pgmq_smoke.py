"""PGMQ smoke: enqueue, transactional consume, ack of a no-op job.

Slow: requires live Postgres reachable at $POSTGRES_DSN. Each test runs
against an ephemeral DB with all migrations applied and the phase-1 queue
topology pre-created.

The point of these tests is the contract, not the surface: prove that
worker reads + status transitions + PGMQ archives commit atomically, that
already-completed work is acked without re-running, and that the
exact-row claim path (the prior fix) is what `process_one` actually uses.
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos import ingest_step_status as iss
from db.repos.talks import Talk, upsert
from queues import pgmq_client, workers

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_pgmq"
QUEUE = "ingest_fetch"


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
    # Create queues once per module — pgmq.create is idempotent so per-test
    # ensure_queues would also be fine, but module-scope amortises the cost.
    with psycopg.connect(test_dsn, autocommit=True) as c:
        pgmq_client.ensure_queues(c)
    yield test_dsn
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")


@pytest.fixture
def conn(test_db):
    """Per-test connection with a clean slate. TRUNCATE drops talks +
    ingest_step_status; pgmq.purge_queue clears each phase-1 queue."""
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("TRUNCATE ingest_step_status, talks CASCADE")
        for q in pgmq_client.QUEUES:
            pgmq_client.purge(c, q)
        yield c


def _make_talk(video_id: str) -> Talk:
    return Talk(
        video_id=video_id,
        title="Smoke",
        speaker="S",
        url="https://example.com/v",
        duration_sec=60,
        format_tags=["narrative"],
        license="mit",
        captions_source="youtube_auto",
        transcript_path="transcripts/smoke.vtt",
        transcript_sha256="a" + "0" * 63,
        accessed_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _seed(conn, video_id: str, step: str = "fetch") -> int:
    """Upsert talk + pending status row, enqueue a payload. Return msg_id."""
    upsert(conn, _make_talk(video_id))
    iss.upsert(conn, video_id, step=step)
    return pgmq_client.send(conn, QUEUE, {"video_id": video_id, "step": step, "entity_id": 0})


# -- pgmq_client primitives ------------------------------------------------


def test_send_returns_positive_msg_id(conn):
    msg_id = pgmq_client.send(conn, QUEUE, {"hello": "world"})
    assert msg_id > 0


def test_read_returns_payload_with_msg_id(conn):
    msg_id = pgmq_client.send(conn, QUEUE, {"video_id": "v", "step": "fetch"})
    msgs = pgmq_client.read(conn, QUEUE, vt=30, qty=1)
    assert len(msgs) == 1
    assert msgs[0].msg_id == msg_id
    assert msgs[0].message == {"video_id": "v", "step": "fetch"}


def test_archive_removes_from_visible_queue(conn):
    msg_id = pgmq_client.send(conn, QUEUE, {"x": 1})
    msgs = pgmq_client.read(conn, QUEUE, vt=30, qty=1)
    assert pgmq_client.archive(conn, QUEUE, msgs[0].msg_id) is True
    # No more visible messages (vt could not have expired in <1s anyway).
    again = pgmq_client.read(conn, QUEUE, vt=30, qty=1)
    assert again == []
    # Sanity: the msg_id we archived was the one we read.
    assert msg_id == msgs[0].msg_id


def test_send_batch_returns_msg_ids_in_order(conn):
    payloads = [
        {"video_id": "a", "step": "embed_text", "entity_id": 1},
        {"video_id": "a", "step": "embed_text", "entity_id": 2},
        {"video_id": "a", "step": "embed_text", "entity_id": 3},
    ]
    msg_ids = pgmq_client.send_batch(conn, QUEUE, payloads)
    assert len(msg_ids) == 3
    assert all(isinstance(m, int) and m > 0 for m in msg_ids)
    # Round-trip: read them back and confirm the payloads match (order on dequeue
    # is FIFO by msg_id, which is also the order we sent them).
    msgs = pgmq_client.read(conn, QUEUE, vt=30, qty=3)
    assert [m.msg_id for m in msgs] == msg_ids
    assert [m.message for m in msgs] == payloads


def test_send_batch_empty_input_no_db_call(conn):
    # Empty input short-circuits to [] without hitting the database.
    assert pgmq_client.send_batch(conn, QUEUE, []) == []
    # And no messages were enqueued.
    assert pgmq_client.read(conn, QUEUE, vt=1, qty=1) == []


# -- worker process_one ---------------------------------------------------


def test_process_one_empty_queue_returns_false(conn):
    handler_calls: list[dict] = []

    def handler(c, p):
        handler_calls.append(p)

    with conn.transaction():
        result = workers.process_one(conn, QUEUE, handler)
    assert result is False
    assert handler_calls == []


def test_process_one_happy_path_completes_and_archives(conn):
    """Enqueue → consume → claim → handler → mark_completed → archive,
    all atomic in the caller's transaction."""
    _seed(conn, "smoke-vid")
    handler_calls: list[dict] = []

    def handler(c, p):
        handler_calls.append(p)

    with conn.transaction():
        result = workers.process_one(conn, QUEUE, handler)

    assert result is True
    assert len(handler_calls) == 1
    assert handler_calls[0]["video_id"] == "smoke-vid"

    # Status row is completed with completed_at set.
    row = conn.execute(
        "SELECT status, attempts, completed_at FROM ingest_step_status "
        "WHERE video_id='smoke-vid' AND step='fetch' AND entity_id=0"
    ).fetchone()
    status, attempts, completed_at = row
    assert status == "completed"
    assert attempts == 1
    assert completed_at is not None

    # Queue is empty — archived.
    assert pgmq_client.read(conn, QUEUE, vt=1, qty=1) == []


def test_process_one_already_completed_acks_without_running_handler(conn):
    """A redelivered job for already-completed work must NOT re-invoke the
    handler. The worker observes status=completed via claim_for_update and
    archives the message without doing any work."""
    _seed(conn, "already-done")
    iss.upsert(conn, "already-done", step="fetch", status="completed")

    handler_calls: list[dict] = []

    def handler(c, p):
        handler_calls.append(p)

    with conn.transaction():
        result = workers.process_one(conn, QUEUE, handler)

    assert result is True
    assert handler_calls == []  # handler NEVER ran

    # Status still completed; attempts not bumped.
    row = conn.execute(
        "SELECT status, attempts FROM ingest_step_status "
        "WHERE video_id='already-done' AND step='fetch'"
    ).fetchone()
    assert row == ("completed", 0)

    # Queue is empty — archived.
    assert pgmq_client.read(conn, QUEUE, vt=1, qty=1) == []


def test_process_one_handler_crash_rolls_back_consume(conn):
    """Transactional consume: if the handler raises inside the caller's
    transaction, the rollback undoes the read, the claim, and the
    (would-be) archive. The message remains visible for the next worker
    and the status row stays at its pre-claim state."""
    _seed(conn, "crash-vid")

    def boom(c, p):
        raise RuntimeError("simulated handler crash")

    with pytest.raises(RuntimeError, match="simulated handler crash"):
        with conn.transaction():
            workers.process_one(conn, QUEUE, boom)

    # Status row rolled back to its pre-claim state (pending, attempts=0).
    row = conn.execute(
        "SELECT status, attempts, started_at FROM ingest_step_status "
        "WHERE video_id='crash-vid' AND step='fetch'"
    ).fetchone()
    assert row == ("pending", 0, None)

    # Message is back on the queue (the read was rolled back).
    msgs = pgmq_client.read(conn, QUEUE, vt=30, qty=1)
    assert len(msgs) == 1
    assert msgs[0].message["video_id"] == "crash-vid"


def test_process_one_uses_exact_row_claim(conn):
    """Regression: a payload for video B must not transition video A's row.

    Builds on the repo-level safety test by exercising it through the
    full process_one pipeline — both videos pending, enqueue only B,
    process_one transitions only B.
    """
    _seed(conn, "vid-a")
    pgmq_client.purge(conn, QUEUE)  # drop A's enqueue; we only want B's job
    _seed(conn, "vid-b")

    def handler(c, p):
        pass

    with conn.transaction():
        workers.process_one(conn, QUEUE, handler)

    a_row = conn.execute(
        "SELECT status, attempts FROM ingest_step_status WHERE video_id='vid-a'"
    ).fetchone()
    b_row = conn.execute(
        "SELECT status, attempts FROM ingest_step_status WHERE video_id='vid-b'"
    ).fetchone()
    assert a_row == ("pending", 0)
    assert b_row == ("completed", 1)


# -- worker contract: failure-mode rollbacks ------------------------------


def test_process_one_raises_when_mark_completed_returns_false(conn):
    """If the handler sabotages the status row (e.g. an upstream code path
    flips status away from in_progress), mark_completed returns False.
    process_one MUST raise so the surrounding transaction rolls back —
    silently archiving would commit "work done but never recorded".
    """
    _seed(conn, "saboteur-vid")

    def saboteur_handler(c, p):
        # Externally flip the row out of in_progress while still in the txn.
        iss.upsert(c, "saboteur-vid", step="fetch", status="failed")

    with pytest.raises(workers.WorkerCompletionFailedError):
        with conn.transaction():
            workers.process_one(conn, QUEUE, saboteur_handler)

    # Transaction rolled back: status back to pre-claim, message visible again.
    row = conn.execute(
        "SELECT status, attempts FROM ingest_step_status "
        "WHERE video_id='saboteur-vid' AND step='fetch'"
    ).fetchone()
    assert row == ("pending", 0)
    msgs = pgmq_client.read(conn, QUEUE, vt=30, qty=1)
    assert len(msgs) == 1
    assert msgs[0].message["video_id"] == "saboteur-vid"


def test_process_one_raises_when_archive_returns_false(conn, monkeypatch):
    """If archive reports the message did not archive, process_one MUST
    raise. Committing the status transition while the message stays visible
    would cause double-work on the next dequeue."""
    _seed(conn, "archive-fail-vid")

    def handler(c, p):
        pass

    # Force archive to report failure for this test only.
    monkeypatch.setattr(pgmq_client, "archive", lambda c, q, mid: False)

    with pytest.raises(workers.WorkerArchiveFailedError):
        with conn.transaction():
            workers.process_one(conn, QUEUE, handler)

    # Transaction rolled back: status back to pending, message visible again.
    row = conn.execute(
        "SELECT status, attempts FROM ingest_step_status "
        "WHERE video_id='archive-fail-vid' AND step='fetch'"
    ).fetchone()
    assert row == ("pending", 0)
    msgs = pgmq_client.read(conn, QUEUE, vt=30, qty=1)
    assert len(msgs) == 1


def test_process_one_already_completed_branch_raises_on_archive_failure(conn, monkeypatch):
    """The already-completed / skipped branch also has to honour the archive
    contract — silently swallowing a failed archive there would leave a
    completed row plus a visible message, causing redelivery forever."""
    _seed(conn, "completed-archive-fail")
    iss.upsert(conn, "completed-archive-fail", step="fetch", status="completed")

    def handler(c, p):
        pass  # never runs in this branch

    monkeypatch.setattr(pgmq_client, "archive", lambda c, q, mid: False)

    with pytest.raises(workers.WorkerArchiveFailedError):
        with conn.transaction():
            workers.process_one(conn, QUEUE, handler)

    # Status was 'completed' before; rollback restored it (the saboteur upsert
    # is inside the txn). The visible message is back.
    msgs = pgmq_client.read(conn, QUEUE, vt=30, qty=1)
    assert len(msgs) == 1
