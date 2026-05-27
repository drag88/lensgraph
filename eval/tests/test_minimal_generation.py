"""Tests for the minimal_generation bakeoff harness.

Slow: writes to live Postgres. Provider HTTP is mocked via
``monkeypatch.setattr(providers, "chat_completion", fake)``.

What the harness must do (design §6):
  * Build a prompt from (query, top-k retrieved chunks).
  * Call ``providers.chat_completion(candidate_id, ...)``.
  * Parse the response with the SHARED ``parse_generation_output``.
  * Write EXACTLY ONE ``eval_runs`` row with
    ``code_path='minimal_generation'``.
  * Bubble ``parse_ok``, ``latency_ms``, and ``abstain`` into
    ``summary jsonb`` so phase 2's selection rule can read it.
"""

from __future__ import annotations

import json

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply
from eval.runners import minimal_generation as mg
from eval.runners import providers
from eval.runners.minimal_generation import RetrievedChunkLite
from eval.runners.providers import ProviderError, ProviderResponse

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_minimal_generation"


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


def _chunks() -> list[RetrievedChunkLite]:
    return [
        RetrievedChunkLite(
            chunk_id=1,
            video_id="W_CYk2ogcDI",
            start_sec=168.0,
            end_sec=200.0,
            text="Tengyu describes a library where books are facts.",
        ),
        RetrievedChunkLite(
            chunk_id=2,
            video_id="W_CYk2ogcDI",
            start_sec=200.0,
            end_sec=240.0,
            text="Retrieval is the lookup; the index points at the book.",
        ),
    ]


def test_happy_path_writes_eval_runs_row(conn, monkeypatch):
    payload = json.dumps(
        {
            "answer": "library analogy",
            "claims": [{"text": "library = index"}],
            "citations": [
                {
                    "video_id": "W_CYk2ogcDI",
                    "start_sec": 168.0,
                    "end_sec": 200.0,
                    "answer_claim_index": 0,
                }
            ],
            "abstain": False,
        }
    )
    fake = ProviderResponse(
        raw_text=payload,
        latency_ms=412,
        provider_model_id="Qwen/Qwen3-235B-A22B-Instruct",
    )
    monkeypatch.setattr(providers, "chat_completion", lambda *a, **k: fake)

    result = mg.generate_for_bakeoff(
        conn=conn,
        example_id="tengyu-rag-library-analogy",
        query="How does Tengyu Ma use a library analogy?",
        chunks=_chunks(),
        candidate_id="qwen3-235b-a22b-instruct",
        run_id="run-happy-1",
    )

    assert result["parse_ok"] is True
    assert result["latency_ms"] == 412
    assert result["run_id"] == "run-happy-1"

    row = conn.execute(
        "SELECT code_path, generator_model_id, summary "
        "FROM eval_runs WHERE run_id = %s",
        ("run-happy-1",),
    ).fetchone()
    assert row is not None
    code_path, gen_id, summary = row
    assert code_path == "minimal_generation"
    assert gen_id == "qwen3-235b-a22b-instruct"
    assert summary["component"] == "minimal_generation_sweep"
    assert summary["candidate_id"] == "qwen3-235b-a22b-instruct"
    assert summary["parse_ok"] is True
    assert summary["latency_ms"] == 412
    assert summary["abstain"] is False

    res = conn.execute(
        "SELECT system_output, metrics FROM eval_results "
        "WHERE eval_run_id = %s AND example_id = %s",
        ("run-happy-1", "tengyu-rag-library-analogy"),
    ).fetchone()
    assert res is not None
    sys_out, metrics = res
    assert sys_out["answer"] == "library analogy"
    assert metrics["parse_ok"] is True
    assert metrics["latency_ms"] == 412


def test_malformed_response_still_writes_row(conn, monkeypatch):
    fake = ProviderResponse(
        raw_text="not json {{",
        latency_ms=88,
        provider_model_id="Qwen/Qwen3-235B-A22B-Instruct",
    )
    monkeypatch.setattr(providers, "chat_completion", lambda *a, **k: fake)

    result = mg.generate_for_bakeoff(
        conn=conn,
        example_id="garbage-example",
        query="any",
        chunks=_chunks(),
        candidate_id="qwen3-235b-a22b-instruct",
        run_id="run-malformed-1",
    )

    assert result["parse_ok"] is False
    assert result["latency_ms"] == 88

    summary = conn.execute(
        "SELECT summary FROM eval_runs WHERE run_id = %s",
        ("run-malformed-1",),
    ).fetchone()[0]
    assert summary["parse_ok"] is False
    assert summary["latency_ms"] == 88


def test_provider_error_propagates_without_writing_row(conn, monkeypatch):
    def boom(*a, **k):
        raise ProviderError("simulated rate limit exhaustion")

    monkeypatch.setattr(providers, "chat_completion", boom)

    with pytest.raises(ProviderError):
        mg.generate_for_bakeoff(
            conn=conn,
            example_id="rate-limited",
            query="any",
            chunks=_chunks(),
            candidate_id="qwen3-235b-a22b-instruct",
            run_id="run-error-1",
        )

    # No eval_runs row written.
    row = conn.execute(
        "SELECT 1 FROM eval_runs WHERE run_id = %s",
        ("run-error-1",),
    ).fetchone()
    assert row is None
