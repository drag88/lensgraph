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


def test_real_measurements_against_dev_gold():
    """Phase-2 contract: invoking the script with NO ``--measurements-json``
    must run a real per-channel TR@5 sweep against the live DB's dev_gold
    corpus, write the per-channel floats into ``eval_runs.summary``, and
    lock ``bge-m3-all-channels`` iff the embeddings minimum
    (``timestamp_recall_at_5_vector_only_min`` = 0.75 in the candidates yaml)
    is met on the best-of vector channels.

    This test runs against the LIVE ``lensgraph`` DB because the real sweep
    needs ingested chunks + embeds. It captures the run_ids the subprocess
    appends and deletes them at the end so re-runs are idempotent.
    """
    live_dsn = resolve_dsn()

    def _emb_run_ids() -> set[str]:
        with psycopg.connect(live_dsn, autocommit=True) as c:
            rows = c.execute(
                "SELECT run_id FROM eval_runs WHERE code_path = 'embeddings_bakeoff'",
            ).fetchall()
        return {r[0] for r in rows}

    before = _emb_run_ids()

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.run_embeddings_bakeoff"],
        capture_output=True,
        text=True,
        check=False,
    )

    # We assert on the row regardless of exit code: a failing-minimum run
    # still writes a row (per Mission 2 contingency) but exits non-zero so
    # downstream `&&` chains know not to promote the embedder.
    new_ids = _emb_run_ids() - before
    try:
        assert len(new_ids) == 1, (
            f"expected exactly one new embeddings_bakeoff row, got {new_ids}; "
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
        run_id = next(iter(new_ids))

        with psycopg.connect(live_dsn, autocommit=True) as c:
            row = c.execute(
                "SELECT code_path, summary FROM eval_runs WHERE run_id = %s",
                (run_id,),
            ).fetchone()
        assert row is not None
        code_path, summary = row
        assert code_path == "embeddings_bakeoff"

        # Real per-channel TR@5 numbers (NOT the synthetic stub).
        per_channel = summary.get("per_channel_tr_at_5")
        assert isinstance(per_channel, dict), f"missing per_channel_tr_at_5: {summary}"
        for channel in ("dense", "sparse", "multivec", "rrf_4ch"):
            assert channel in per_channel, f"missing channel {channel}: {per_channel}"
            score = per_channel[channel]
            assert isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0, (
                f"channel {channel} score out of range: {score}"
            )

        # Selection rule: best of dense/sparse/multivec vs the yaml minimum.
        vector_best = max(per_channel["dense"], per_channel["sparse"], per_channel["multivec"])
        min_threshold = float(summary.get("embeddings_min_threshold", 0.75))
        if vector_best >= min_threshold:
            assert summary.get("winner_locked") is True, (
                f"min met ({vector_best:.3f} >= {min_threshold}) but winner_locked != True"
            )
            assert summary.get("component") == "text_embeddings"
            assert summary.get("winner_candidate_id") == "bge-m3-all-channels"
            assert proc.returncode == 0, (
                f"min met but exit != 0\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )
        else:
            assert summary.get("winner_locked") is not True, (
                f"min FAILED ({vector_best:.3f} < {min_threshold}) but row was locked anyway"
            )
            assert proc.returncode != 0, (
                "min FAILED but exit was 0 — caller would think the lock succeeded"
            )

        # Hard rule: per-channel measurements land in eval_runs.summary
        # AND in eval_results. Assert one row per dev_gold example was
        # written under this run_id, with per-channel pass@5 + the
        # actual top-k chunks in system_output.
        n_examples = int(summary["n_examples"])
        with psycopg.connect(live_dsn, autocommit=True) as c:
            rows = c.execute(
                """
                SELECT example_id, system_output, metrics
                  FROM eval_results
                 WHERE eval_run_id = %s
                """,
                (run_id,),
            ).fetchall()
        assert len(rows) == n_examples, (
            f"expected {n_examples} eval_results rows under {run_id}, got {len(rows)}"
        )
        for example_id, system_output, metrics in rows:
            assert "gold_span" in system_output, system_output
            assert "top_k_per_channel" in system_output
            for channel in ("dense", "sparse", "multivec", "rrf_4ch"):
                assert channel in system_output["top_k_per_channel"], (
                    f"{example_id} missing top-k for channel {channel}"
                )
                assert isinstance(system_output["top_k_per_channel"][channel], list)
            pass_map = metrics["pass_at_5_per_channel"]
            for channel in ("dense", "sparse", "multivec", "rrf_4ch"):
                assert channel in pass_map, f"{example_id} metrics missing {channel}"
                assert isinstance(pass_map[channel], bool)
            assert metrics["tr_at_5_best_vector"] in (0.0, 1.0)
    finally:
        # CASCADE on eval_runs.run_id reaps the matching eval_results rows.
        if new_ids:
            with psycopg.connect(live_dsn, autocommit=True) as c:
                c.execute(
                    "DELETE FROM eval_runs WHERE run_id = ANY(%s)",
                    (list(new_ids),),
                )
