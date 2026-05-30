"""Slow integration tests for ``scripts/run_visual_eval.py``.

Two tests:

1. ``test_visual_eval_skips_or_runs_against_live_db`` — runs against
   the LIVE ``lensgraph`` DB. Two skip gates make the test honest:
   visual_gold empty → skip; frames/frame_patches empty → skip. Today
   the repo hits both, so we assert the SKIP path itself (script exits
   0, writes no row, prints the right message). When the substrate +
   gold fill in, the same test body asserts the real run.

2. ``test_skip_when_visual_gold_video_has_no_substrate`` — uses a
   throwaway test DB. Stages a talk with frames + patches for
   ``unrelated-video``, but the visual_gold.jsonl points at
   ``video-without-frames``. The per-video substrate check must skip
   (NOT measure with degenerate zeros) and write nothing to eval_runs.
   This is the regression test for the global-vs-per-video readiness
   fix.
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

_TEST_DB_NAME = "lensgraph_test_run_visual_eval"


def _swap_db(dsn: str, new_db: str) -> str:
    head, _, _ = dsn.rpartition("/")
    return f"{head}/{new_db}"


@pytest.fixture(scope="module")
def test_db():
    admin = resolve_dsn()
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {_TEST_DB_NAME}")
        c.execute(f"CREATE DATABASE {_TEST_DB_NAME}")
    test_dsn = _swap_db(admin, _TEST_DB_NAME)
    apply(test_dsn)
    yield test_dsn
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {_TEST_DB_NAME}")


def _check_substrate() -> tuple[int, int]:
    with psycopg.connect(resolve_dsn(), autocommit=True) as c:
        frames = c.execute(
            "SELECT count(*) FROM frames WHERE pooled_embedding IS NOT NULL"
        ).fetchone()[0]
        patches = c.execute("SELECT count(*) FROM frame_patches").fetchone()[0]
    return frames, patches


def _count_visual_eval_rows() -> set[str]:
    with psycopg.connect(resolve_dsn(), autocommit=True) as c:
        rows = c.execute("SELECT run_id FROM eval_runs WHERE code_path = 'visual_eval'").fetchall()
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
            assert new_ids == set(), f"skip path must write nothing; got new rows {new_ids}"
            return
        if not substrate_ready:
            assert "SKIPPED: visual substrate not ingested" in proc.stdout, proc.stdout
            assert new_ids == set(), f"skip path must write nothing; got new rows {new_ids}"
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
            # VisualAnswerGrounding is now a structured audit sub-object
            # (dict) or None — the old scalar key is gone (no shim).
            assert "visual_answer_grounding_at_k" not in metrics
            assert "visual_answer_grounding" in metrics
            vag = metrics["visual_answer_grounding"]
            assert vag is None or isinstance(vag, dict)
            if isinstance(vag, dict):
                # Every evaluable row's payload is self-defending: it names
                # the frames it judged and carries the pass flag + reason.
                assert set(vag) >= {
                    "passed",
                    "evaluated_frame_ids",
                    "evaluated_image_paths",
                    "ocr_excerpts",
                    "matched_term",
                    "failure_reason",
                    "judge_kind",
                }
                assert isinstance(vag["passed"], bool)
                # A passing row must name a frame and a term; a failing row
                # must name a reason. No silent pass with empty evidence.
                if vag["passed"]:
                    assert vag["evaluated_frame_ids"]
                    assert vag["matched_term"] is not None
                    assert vag["failure_reason"] is None
                else:
                    assert vag["failure_reason"] is not None
    finally:
        # CASCADE on eval_runs.run_id reaps the matching eval_results rows.
        if new_ids:
            with psycopg.connect(resolve_dsn(), autocommit=True) as c:
                c.execute(
                    "DELETE FROM eval_runs WHERE run_id = ANY(%s)",
                    (list(new_ids),),
                )


def _insert_talk(conn: psycopg.Connection, video_id: str) -> None:
    conn.execute(
        """
        INSERT INTO talks (
            video_id, title, speaker, url, duration_sec, format_tags,
            license, captions_source, transcript_path, transcript_sha256,
            accessed_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """,
        (
            video_id,
            f"Title {video_id}",
            "Speaker",
            f"https://example.com/{video_id}",
            600,
            ["slides_heavy"],
            "youtube_standard",
            "youtube_auto",
            f"transcripts/{video_id}.vtt",
            "0" * 64,
            "2026-05-27T00:00:00Z",
        ),
    )


def test_skip_when_visual_gold_video_has_no_substrate(test_db, tmp_path, monkeypatch):
    """The fix for finding #1: per-video substrate readiness, not a
    global frame count.

    Stage frames + patches for ``unrelated-video`` (so a global check
    would have passed) but point visual_gold at ``video-without-frames``.
    The script must skip with the per-video missing message and write
    no eval_runs row."""
    from pgvector.psycopg import register_vector

    visual_video = "video-without-frames"
    unrelated_video = "unrelated-video"

    with psycopg.connect(test_db, autocommit=True) as c:
        register_vector(c)
        _insert_talk(c, unrelated_video)
        _insert_talk(c, visual_video)
        # Frames + patches exist ONLY for unrelated-video.
        vec = [0.1] * 128  # ColQwen pooled vec dim from migration 0003
        fid = c.execute(
            """INSERT INTO frames (video_id, frame_sec, image_path, sha256, pooled_embedding)
               VALUES (%s, %s, %s, %s, %s) RETURNING frame_id""",
            (unrelated_video, 10.0, "/tmp/x.png", "a" * 64, vec),
        ).fetchone()[0]
        c.execute(
            "INSERT INTO frame_patches (frame_id, patch_index, embedding) VALUES (%s, %s, %s)",
            (fid, 0, vec),
        )

    # visual_gold points at the video with no substrate. Modality must
    # include a visual tag (load_visual_gold enforces this).
    gold_path = tmp_path / "visual_gold.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "id": "vis-no-frames",
                "question": "what does the slide show at the start of the talk?",
                "video_id": visual_video,
                "split": "dev",
                "question_type": "single_clip",
                "gold_spans": [{"start_sec": 100, "end_sec": 120}],
                "modality": ["slide"],
                "difficulty": "easy",
                "curator": "tester",
                "curated_at": "2026-05-27T00:00:00Z",
                "verified": True,
            }
        )
    )

    monkeypatch.setenv("POSTGRES_DSN", test_db)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.runners.run_visual_eval",
            "--visual-gold",
            str(gold_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"per-video skip must exit 0; stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "visual substrate not ingested for visual_gold video_id(s)" in proc.stdout, proc.stdout
    assert visual_video in proc.stdout, proc.stdout

    with psycopg.connect(test_db, autocommit=True) as c:
        n_runs = c.execute(
            "SELECT count(*) FROM eval_runs WHERE code_path = 'visual_eval'"
        ).fetchone()[0]
        n_results = c.execute("SELECT count(*) FROM eval_results").fetchone()[0]
    assert n_runs == 0, f"per-video skip must write no eval_runs row; got {n_runs}"
    assert n_results == 0, f"per-video skip must write no eval_results rows; got {n_results}"
