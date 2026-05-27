"""eval_runs repository tests.

Slow: requires a live Postgres reachable at $POSTGRES_DSN. Spins an
ephemeral test DB per module, applies all migrations once, then exercises
the repo against it. Mirrors the fixture pattern in
``test_embeds_repo.py`` — clean slate per test via per-test connection
+ unique run_ids per test (no TRUNCATE between tests because no test
shares a run_id).

What this proves:
  * ``insert_run`` round-trips every column we care about, including
    ``summary jsonb``.
  * ``insert_result`` PK violation on a duplicate ``(run_id,
    example_id)`` is exactly the contract we want.
  * ``lock_bakeoff_winner`` mutates ``summary`` in place to mark
    ``winner_locked=true`` + ``component`` + ``winner_candidate_id``,
    and is idempotent (re-running with the same args is a no-op, no
    exception).
  * ``load_bakeoff_winner`` returns ``None`` when no row is locked, the
    locked id when one is, and the MOST-RECENT lock when multiple
    exist for the same component (per design §5 SQL: ``ORDER BY
    created_at DESC LIMIT 1``).
"""

from __future__ import annotations

import uuid
from datetime import date

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from db.repos import eval_runs

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_eval_runs_repo"


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
        yield c


def _rid() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def test_insert_run_round_trip(conn):
    run_id = _rid()
    summary = {
        "component": "minimal_generation_sweep",
        "candidate_id": "qwen3-235b-a22b-instruct",
        "parse_ok": True,
        "latency_ms": 412,
    }
    candidate_set = {"hello": "world", "k": [1, 2]}
    eval_runs.insert_run(
        conn,
        run_id=run_id,
        run_date=date(2026, 5, 27),
        code_path="minimal_generation",
        chunking_strategy="fixed_window",
        embedding_model_id="bge-m3-all-channels",
        generator_model_id="qwen3-235b-a22b-instruct",
        judge_model_id=None,
        judge_prompt_hash=None,
        candidate_set_yaml=candidate_set,
        summary=summary,
    )
    row = conn.execute(
        """
        SELECT run_id, run_date, code_path, chunking_strategy,
               embedding_model_id, generator_model_id, judge_model_id,
               judge_prompt_hash, candidate_set_yaml, summary
          FROM eval_runs WHERE run_id = %s
        """,
        (run_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == run_id
    assert row[1] == date(2026, 5, 27)
    assert row[2] == "minimal_generation"
    assert row[3] == "fixed_window"
    assert row[4] == "bge-m3-all-channels"
    assert row[5] == "qwen3-235b-a22b-instruct"
    assert row[6] is None
    assert row[7] is None
    assert row[8] == candidate_set
    assert row[9] == summary


def test_insert_result_round_trip_and_pk_violation(conn):
    run_id = _rid()
    eval_runs.insert_run(
        conn,
        run_id=run_id,
        run_date=date(2026, 5, 27),
        code_path="minimal_generation",
        chunking_strategy="fixed_window",
        embedding_model_id="bge-m3-all-channels",
        candidate_set_yaml={"x": 1},
        summary={},
    )
    eval_runs.insert_result(
        conn,
        run_id=run_id,
        example_id="example-1",
        system_output={"answer": "x"},
        metrics={"parse_ok": True},
    )
    row = conn.execute(
        "SELECT system_output, metrics FROM eval_results "
        "WHERE eval_run_id = %s AND example_id = %s",
        (run_id, "example-1"),
    ).fetchone()
    assert row is not None
    assert row[0] == {"answer": "x"}
    assert row[1] == {"parse_ok": True}

    # Re-insert same (run_id, example_id) → PK violation.
    with pytest.raises(psycopg.errors.UniqueViolation):
        eval_runs.insert_result(
            conn,
            run_id=run_id,
            example_id="example-1",
            system_output={"answer": "y"},
            metrics={"parse_ok": False},
        )


def test_lock_bakeoff_winner_marks_summary_and_is_idempotent(conn):
    run_id = _rid()
    eval_runs.insert_run(
        conn,
        run_id=run_id,
        run_date=date(2026, 5, 27),
        code_path="embeddings_bakeoff",
        chunking_strategy="fixed_window",
        embedding_model_id="bge-m3-all-channels",
        candidate_set_yaml={"x": 1},
        summary={},
    )
    eval_runs.lock_bakeoff_winner(
        conn,
        component="text_embeddings",
        candidate_id="bge-m3-all-channels",
        run_id=run_id,
    )
    s = conn.execute(
        "SELECT summary FROM eval_runs WHERE run_id = %s", (run_id,)
    ).fetchone()[0]
    assert s["winner_locked"] is True
    assert s["component"] == "text_embeddings"
    assert s["winner_candidate_id"] == "bge-m3-all-channels"

    # Idempotent: re-call same args is a no-op (no exception).
    eval_runs.lock_bakeoff_winner(
        conn,
        component="text_embeddings",
        candidate_id="bge-m3-all-channels",
        run_id=run_id,
    )
    s2 = conn.execute(
        "SELECT summary FROM eval_runs WHERE run_id = %s", (run_id,)
    ).fetchone()[0]
    assert s2 == s


def test_load_bakeoff_winner_returns_most_recent(conn):
    # No lock yet → None.
    assert eval_runs.load_bakeoff_winner(conn, component="generator") is None

    # Lock an older row.
    old = _rid()
    eval_runs.insert_run(
        conn,
        run_id=old,
        run_date=date(2026, 5, 1),
        code_path="minimal_generation",
        chunking_strategy="fixed_window",
        embedding_model_id="bge-m3-all-channels",
        candidate_set_yaml={"x": 1},
        summary={},
    )
    eval_runs.lock_bakeoff_winner(
        conn,
        component="generator",
        candidate_id="qwen3-235b-a22b-instruct",
        run_id=old,
    )
    assert (
        eval_runs.load_bakeoff_winner(conn, component="generator")
        == "qwen3-235b-a22b-instruct"
    )

    # Lock a newer row with a different candidate → newest wins.
    newer = _rid()
    eval_runs.insert_run(
        conn,
        run_id=newer,
        run_date=date(2026, 5, 20),
        code_path="minimal_generation",
        chunking_strategy="fixed_window",
        embedding_model_id="bge-m3-all-channels",
        candidate_set_yaml={"x": 1},
        summary={},
    )
    eval_runs.lock_bakeoff_winner(
        conn,
        component="generator",
        candidate_id="deepseek-v3.2",
        run_id=newer,
    )
    assert (
        eval_runs.load_bakeoff_winner(conn, component="generator")
        == "deepseek-v3.2"
    )


def test_load_bakeoff_winner_ignores_non_minimal_generation_code_path(conn):
    """Design §5 SQL: generator/judge winner resolution must filter by
    code_path='minimal_generation'. A phase-3 langgraph_loop run with
    winner_locked=true must NEVER be picked up as the canonical generator
    selection — the loop adds Plan/Verify/Cite confounds the bakeoff is
    built to isolate.

    Setup: lock a minimal_generation winner (gemma) THEN lock a more
    recent langgraph_loop winner (qwen). Without the filter, the MRU
    rule would return qwen; with the filter, gemma wins. The newest row
    here is langgraph_loop so the test catches a missing filter — if
    load_bakeoff_winner didn't apply ``code_path``, MRU would pick
    langgraph_loop's qwen.
    """
    older_mg = _rid()
    eval_runs.insert_run(
        conn,
        run_id=older_mg,
        run_date=date(2026, 5, 1),
        code_path="minimal_generation",
        chunking_strategy="fixed_window",
        embedding_model_id="bge-m3-all-channels",
        candidate_set_yaml={"x": 1},
        summary={},
    )
    eval_runs.lock_bakeoff_winner(
        conn,
        component="generator",
        candidate_id="gemma-4-31b",
        run_id=older_mg,
    )

    newer_lg = _rid()
    eval_runs.insert_run(
        conn,
        run_id=newer_lg,
        run_date=date(2026, 5, 20),
        code_path="langgraph_loop",
        chunking_strategy="fixed_window",
        embedding_model_id="bge-m3-all-channels",
        candidate_set_yaml={"x": 1},
        summary={},
    )
    eval_runs.lock_bakeoff_winner(
        conn,
        component="generator",
        candidate_id="qwen3-235b-a22b-instruct",
        run_id=newer_lg,
    )

    # Filtered: only minimal_generation rows count → gemma wins.
    assert (
        eval_runs.load_bakeoff_winner(
            conn, component="generator", code_path="minimal_generation"
        )
        == "gemma-4-31b"
    ), "code_path filter must keep the langgraph_loop lock from leaking through"

    # Unfiltered (None) returns MRU across all code_paths — used by tests
    # only; production callers MUST pass code_path.
    assert (
        eval_runs.load_bakeoff_winner(conn, component="generator")
        == "qwen3-235b-a22b-instruct"
    )
