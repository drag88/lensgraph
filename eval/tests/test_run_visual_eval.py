"""Slow integration test for ``scripts/run_visual_eval.py``.

Runs against the LIVE ``lensgraph`` DB. Two skip gates make the test
honest about what it can prove today:

* Visual gold is empty → skip (no gold to score).
* Frames / frame_patches tables empty → skip (no substrate to query).

Today's repo state hits both: ``visual_gold.jsonl`` is a committed
empty scaffold, and the MP4 + ColQwen patch ingest hasn't been run.
We still exercise the SKIP path itself (assert the script exits 0 with
a clear message and writes no row) so the gate code stays honest as
the substrate fills in.

The "happy path" — when both substrate + gold are populated — is
covered by the same test body via the ``substrate_ready`` boolean: if
True, the test asserts the actual run wrote a row + N eval_results
rows + the script reported the metrics.
"""

from __future__ import annotations

import subprocess
import sys

import psycopg
import pytest

from db.conn import dsn as resolve_dsn

pytestmark = pytest.mark.slow


def _check_substrate() -> tuple[int, int]:
    with psycopg.connect(resolve_dsn(), autocommit=True) as c:
        frames = c.execute(
            "SELECT count(*) FROM frames WHERE pooled_embedding IS NOT NULL"
        ).fetchone()[0]
        patches = c.execute("SELECT count(*) FROM frame_patches").fetchone()[0]
    return frames, patches


def _count_visual_eval_rows() -> set[str]:
    with psycopg.connect(resolve_dsn(), autocommit=True) as c:
        rows = c.execute(
            "SELECT run_id FROM eval_runs WHERE code_path = 'visual_eval'"
        ).fetchall()
    return {r[0] for r in rows}


def test_visual_eval_skips_or_runs_against_live_db():
    """Skip cleanly when prerequisites are absent; otherwise run the
    real eval, assert the row + per-example writes, and self-clean.

    This test is intentionally one path, not two: the skip messages
    are part of the contract — a stale "no frames" or "no gold" check
    can silently let the gate degrade, so the test asserts the
    SKIPPED text the script emits."""
    from eval.runners.measure_visual import load_visual_gold

    gold = load_visual_gold()
    n_frames, n_patches = _check_substrate()
    substrate_ready = n_frames > 0 and n_patches > 0
    gold_ready = len(gold) > 0

    before = _count_visual_eval_rows()

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.run_visual_eval"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"script must exit 0 even when skipping; stdout={proc.stdout}\nstderr={proc.stderr}"
    )

    after = _count_visual_eval_rows()
    new_ids = after - before
    try:
        if not gold_ready:
            assert "SKIPPED: visual_gold.jsonl has 0 single_clip" in proc.stdout, proc.stdout
            assert new_ids == set(), (
                f"skip path must write nothing; got new rows {new_ids}"
            )
            return
        if not substrate_ready:
            assert "SKIPPED: visual substrate not ingested" in proc.stdout, proc.stdout
            assert new_ids == set(), (
                f"skip path must write nothing; got new rows {new_ids}"
            )
            return

        # Happy path: both gold + substrate populated. Assert one new
        # eval_runs row + N matching eval_results rows.
        assert len(new_ids) == 1, (
            f"expected exactly one new visual_eval row, got {new_ids}; "
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
        run_id = next(iter(new_ids))
        with psycopg.connect(resolve_dsn(), autocommit=True) as c:
            row = c.execute(
                "SELECT code_path, summary FROM eval_runs WHERE run_id = %s",
                (run_id,),
            ).fetchone()
            assert row is not None
            code_path, summary = row
            assert code_path == "visual_eval"
            assert summary["component"] == "visual_retrieval"
            assert summary["candidate_id"] == "colqwen2.5"
            assert "visual_frame_recall_at_k" in summary
            assert "visual_chunk_tr_at_k" in summary
            assert summary["n_examples"] == len(gold)

            results = c.execute(
                "SELECT example_id, metrics FROM eval_results WHERE eval_run_id = %s",
                (run_id,),
            ).fetchall()
        assert len(results) == len(gold), (
            f"expected {len(gold)} eval_results rows under {run_id}, got {len(results)}"
        )
        for _ex_id, metrics in results:
            assert "frame_pass_at_k" in metrics
            assert "chunk_pass_at_k" in metrics
            assert isinstance(metrics["frame_pass_at_k"], bool)
            assert isinstance(metrics["chunk_pass_at_k"], bool)
    finally:
        # CASCADE on eval_runs.run_id reaps the matching eval_results rows.
        if new_ids:
            with psycopg.connect(resolve_dsn(), autocommit=True) as c:
                c.execute(
                    "DELETE FROM eval_runs WHERE run_id = ANY(%s)",
                    (list(new_ids),),
                )
