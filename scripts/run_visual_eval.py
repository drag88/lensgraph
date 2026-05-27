"""Visual retrieval eval runner — separate gate from bakeoff #1.

Reads ``eval/corpora/ai_engineering_v0/visual_gold.jsonl``, runs the
visual metrics from ``eval/runners/measure_visual.py`` against the live
DB's ColQwen frame + patch tables, and writes one ``eval_runs`` row
(``code_path='visual_eval'``) plus one ``eval_results`` row per visual
gold example — same eval_runs + eval_results contract as bakeoff #1
per the phase-2 hard rule.

Skip semantics — visual retrieval needs BOTH (a) verified visual gold
examples AND (b) ingested MP4 frames + ColQwen patches. If either is
missing, the runner emits a clear "SKIPPED: <reason>" to stdout and
exits 0. This is intentional: the gate exists so the user can plug in
verified examples + run ingest on real videos without bringing up
infrastructure for a degenerate measurement.

This runner does NOT lock a winner. Visual retrieval has no
single-candidate ADR 004 lock today — ColQwen is the only candidate
(``eval/config/model_candidates.yaml::candidates.visual_retrieval``)
and the eval is a quality gate, not a selection.

CLI:

    python -m scripts.run_visual_eval
        [--no-lift]       # skip the 5ch-vs-4ch RRF lift comparison
        [--k INT]         # top-k for both metrics (default 5)
        [--dry-run]       # print but do not write DB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import date
from pathlib import Path

import psycopg
import yaml

from db.conn import dsn as resolve_dsn
from db.repos import eval_runs

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "eval" / "config" / "model_candidates.yaml"
)


def _yaml_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frames_ingested(conn: psycopg.Connection) -> tuple[int, int]:
    """Return (frames with pooled embedding, frame_patches rows). The
    visual channel needs both to be non-zero for the eval to score
    anything meaningful."""
    frames = conn.execute(
        "SELECT count(*) FROM frames WHERE pooled_embedding IS NOT NULL"
    ).fetchone()[0]
    patches = conn.execute("SELECT count(*) FROM frame_patches").fetchone()[0]
    return frames, patches


def _write_per_example_results(
    conn: psycopg.Connection,
    *,
    run_id: str,
    per_example_detail: dict[str, dict],
) -> None:
    """Emit one ``eval_results`` row per visual gold example under
    ``run_id``. ``system_output`` carries the gold span + per-channel
    top-k frames/chunks; ``metrics`` carries the pass booleans."""
    for example_id, detail in per_example_detail.items():
        metrics = {
            "frame_pass_at_k": detail["frame_pass_at_k"],
            "chunk_pass_at_k": detail["chunk_pass_at_k"],
        }
        if "lift" in detail:
            metrics["lift"] = detail["lift"]
        eval_runs.insert_result(
            conn,
            run_id=run_id,
            example_id=example_id,
            system_output={
                "gold_span": detail["gold_span"],
                "top_k_frames": detail["top_k_frames"],
                "top_k_chunks": detail["top_k_chunks"],
            },
            metrics=metrics,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-lift", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from eval.runners.measure_visual import (
        load_visual_gold,
        measure_visual,
    )

    examples = load_visual_gold()
    if not examples:
        sys.stdout.write(
            "SKIPPED: visual_gold.jsonl has 0 single_clip examples. "
            "Stage verified visual examples (modality includes slide / "
            "screen_code / diagram / whiteboard) before the visual eval "
            "can score anything. See eval/reports/2026-05-27_visual_eval/"
            "methodology.mdx for the staging workflow.\n"
        )
        return 0

    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        n_frames, n_patches = _frames_ingested(conn)
    if n_frames == 0 or n_patches == 0:
        sys.stdout.write(
            f"SKIPPED: visual substrate not ingested "
            f"(frames_with_pooled_embedding={n_frames}, frame_patches={n_patches}). "
            "Run the MP4 ingest + ColQwen patch worker before invoking this eval. "
            f"Visual gold examples present: {len(examples)}.\n"
        )
        return 0

    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg_sha = _yaml_sha256(_CONFIG_PATH)

    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        summary, per_example_detail = measure_visual(
            conn,
            examples,
            k=args.k,
            include_lift=not args.no_lift,
        )

    summary.update(
        {
            "component": "visual_retrieval",
            "candidate_id": "colqwen2.5",
            "measurement_mode": "real_visual_gold_sweep",
            "candidate_set_yaml_sha256": cfg_sha,
        }
    )

    run_id = f"visual-eval-{uuid.uuid4().hex[:12]}"
    if args.dry_run:
        sys.stdout.write(
            json.dumps(
                {
                    "would_write_run_id": run_id,
                    "summary": summary,
                    "n_per_example": len(per_example_detail),
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        return 0

    # Single transaction: insert_run + per-example eval_results commit
    # together. Same atomicity contract as bakeoff #1. No lock call —
    # visual eval is a quality gate, not a selection.
    with psycopg.connect(resolve_dsn()) as conn:
        eval_runs.insert_run(
            conn,
            run_id=run_id,
            run_date=date.today(),
            code_path="visual_eval",
            chunking_strategy="fixed_window",
            embedding_model_id="colqwen2.5",
            candidate_set_yaml=cfg,
            summary=summary,
        )
        _write_per_example_results(
            conn, run_id=run_id, per_example_detail=per_example_detail
        )

    sys.stdout.write(
        f"visual_eval written: {run_id} "
        f"(frame_recall@{args.k}={summary['visual_frame_recall_at_k']:.3f}, "
        f"chunk_tr@{args.k}={summary['visual_chunk_tr_at_k']:.3f}, "
        f"n={summary['n_examples']})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
