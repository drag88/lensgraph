"""Step-35a RED: forward-contract scaffold for the phase-2 embeddings
bakeoff.

Per design step 35a: the script reads channel measurements, applies the
ADR 004 selection rule (cheapest within 3pp meeting minimums), and calls
``db.repos.eval_runs.lock_bakeoff_winner('text_embeddings', <id>, run_id=...)``.

Phase 1 ships ONLY the scaffolding so the contract is testable from
week 5; phase 2 swaps the measurements source for a real sweep.
"""

from __future__ import annotations

import json
import subprocess
import sys

import psycopg
import pytest

from db.conn import dsn as resolve_dsn
from db.migrate import apply

pytestmark = pytest.mark.slow

TEST_DB_NAME = "lensgraph_test_run_embeddings_bakeoff"


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


def test_lock_writes_winner_locked_true_row(test_db, tmp_path, monkeypatch):
    """The script must apply the selection rule and write a
    ``winner_locked=true`` row with component='text_embeddings' +
    winner_candidate_id matching the yaml ``id`` field.

    Synthetic measurements: bge-m3 meets minimums and is cheapest;
    voyage-3-large is higher quality but pricier; gemini-embedding-2
    fails minimums. ADR 004 selection rule picks bge-m3 because:
      * it meets minimums,
      * it's within 3pp of the leader (voyage),
      * it's the cheapest of the within-3pp set.
    """
    measurements_path = tmp_path / "measurements.json"
    measurements_path.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "bge-m3-all-channels",
                    "score": 0.82,
                    "price_per_mtok_usd": 0.0,
                    "meets_minimums": True,
                },
                {
                    "candidate_id": "voyage-3-large",
                    "score": 0.84,
                    "price_per_mtok_usd": 0.18,
                    "meets_minimums": True,
                },
                {
                    "candidate_id": "gemini-embedding-2",
                    "score": 0.74,
                    "price_per_mtok_usd": 0.15,
                    "meets_minimums": False,
                },
            ]
        )
    )

    monkeypatch.setenv("POSTGRES_DSN", test_db)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_embeddings_bakeoff",
            "--measurements-json",
            str(measurements_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"script failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"

    with psycopg.connect(test_db, autocommit=True) as c:
        row = c.execute(
            """
            SELECT code_path, summary
              FROM eval_runs
             WHERE summary->>'winner_locked' = 'true'
               AND summary->>'component' = 'text_embeddings'
            """,
        ).fetchone()
    assert row is not None, "no winner_locked row found"
    code_path, summary = row
    assert code_path == "embeddings_bakeoff"
    assert summary["winner_candidate_id"] == "bge-m3-all-channels"
