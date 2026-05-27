"""Slow tests for ``db/repos/traces.py``.

Live Postgres required. Module-scoped ephemeral DB with migrations
applied, per-test TRUNCATE for isolation.

Proves:
  - ``start_trace`` writes a row with default ``status='running'`` and
    ``ended_at IS NULL``.
  - ``end_trace`` flips status, sets ``ended_at``, and persists final
    fields.
  - ``flush_spans`` batch-inserts N spans in a single call; jsonb
    columns round-trip back as dicts.
  - Empty input to ``flush_spans`` is a no-op (no DB call → no error).
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos import traces as traces_repo
from generate.trace import new_uuid7

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_traces"


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
        c.execute("TRUNCATE trace_spans, traces CASCADE")
        yield c


def test_start_trace_writes_row_with_running_status(conn):
    trace_id = new_uuid7()
    traces_repo.start_trace(
        conn,
        trace_id=trace_id,
        query="how does Tengyu compare RAG?",
        corpus_id="ai_engineering_v0",
        generator_model_id="Qwen/Qwen3-235B-A22B-Instruct",
    )
    row = conn.execute(
        "SELECT query, corpus_id, generator_model_id, status, "
        "started_at, ended_at FROM traces WHERE trace_id = %s",
        (trace_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == "how does Tengyu compare RAG?"
    assert row[1] == "ai_engineering_v0"
    assert row[2] == "Qwen/Qwen3-235B-A22B-Instruct"
    assert row[3] == "running"
    assert row[4] is not None  # started_at defaulted to now()
    assert row[5] is None  # ended_at NULL until end_trace


def test_end_trace_updates_terminal_fields(conn):
    trace_id = new_uuid7()
    traces_repo.start_trace(
        conn,
        trace_id=trace_id,
        query="q",
        corpus_id="c",
        generator_model_id="m",
    )
    traces_repo.end_trace(
        conn,
        trace_id=trace_id,
        status="completed",
        final_answer="the answer",
        final_citations=[{"video_id": "v", "start_sec": 1.0, "end_sec": 5.0}],
        iterations=1,
        latency_ms=1234,
    )
    row = conn.execute(
        "SELECT status, final_answer, final_citations, iterations, "
        "latency_ms, ended_at FROM traces WHERE trace_id = %s",
        (trace_id,),
    ).fetchone()
    assert row[0] == "completed"
    assert row[1] == "the answer"
    assert row[2] == [{"video_id": "v", "start_sec": 1.0, "end_sec": 5.0}]
    assert row[3] == 1
    assert row[4] == 1234
    assert row[5] is not None


def test_flush_spans_batch_inserts_three_rows(conn):
    trace_id = new_uuid7()
    traces_repo.start_trace(
        conn, trace_id=trace_id, query="q", corpus_id="c", generator_model_id=None
    )

    now = datetime.now(UTC)
    spans = [
        {
            "span_id": new_uuid7(),
            "trace_id": trace_id,
            "parent_span_id": None,
            "node_name": "plan",
            "iteration": 0,
            "started_at": now,
            "ended_at": now,
            "input": {"query": "q"},
            "output": {"question_type": "single_clip"},
            "metadata": {"latency_ms": 5},
            "status": "ok",
        },
        {
            "span_id": new_uuid7(),
            "trace_id": trace_id,
            "parent_span_id": None,
            "node_name": "retrieve",
            "iteration": 0,
            "started_at": now,
            "ended_at": now,
            "input": {"query": "q"},
            "output": {"retrieved_chunk_ids": [1, 2, 3]},
            "metadata": {"latency_ms": 42},
            "status": "ok",
        },
        {
            "span_id": new_uuid7(),
            "trace_id": trace_id,
            "parent_span_id": None,
            "node_name": "rerank",
            "iteration": 0,
            "started_at": now,
            "ended_at": now,
            "input": {"retrieved_chunk_ids": [1, 2, 3]},
            "output": {"reranked_chunk_ids": [2, 1, 3]},
            "metadata": {"latency_ms": 110},
            "status": "ok",
        },
    ]
    traces_repo.flush_spans(conn, spans)

    rows = conn.execute(
        "SELECT node_name, iteration, input, output, metadata, status "
        "FROM trace_spans WHERE trace_id = %s ORDER BY started_at, node_name",
        (trace_id,),
    ).fetchall()
    assert len(rows) == 3
    node_names = sorted(r[0] for r in rows)
    assert node_names == ["plan", "rerank", "retrieve"]

    # jsonb round-trip — psycopg decodes back to dicts.
    by_name = {r[0]: r for r in rows}
    assert by_name["retrieve"][2] == {"query": "q"}
    assert by_name["retrieve"][3] == {"retrieved_chunk_ids": [1, 2, 3]}
    assert by_name["retrieve"][4] == {"latency_ms": 42}
    assert all(r[5] == "ok" for r in rows)


def test_flush_spans_empty_input_is_noop(conn):
    # Should not raise, should not execute any SQL.
    traces_repo.flush_spans(conn, [])
    rows = conn.execute("SELECT count(*) FROM trace_spans").fetchone()
    assert rows[0] == 0
