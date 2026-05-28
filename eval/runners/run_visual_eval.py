"""Visual retrieval eval runner — separate gate from bakeoff #1.

Reads ``eval/corpora/ai_engineering_v0/visual_gold.jsonl``, runs the
visual metrics from ``eval/runners/measure_visual.py`` against the live
DB's ColQwen frame + patch tables, and writes one ``eval_runs`` row
(``code_path='visual_eval'``) plus one ``eval_results`` row per visual
gold example — same eval_runs + eval_results contract as bakeoff #1
per the phase-2 hard rule.

This module lives under ``eval/runners/`` (not ``scripts/``) because it
defines an eval run, writes ``eval_runs`` / ``eval_results``, and is
referenced from the eval methodology MDX — i.e. it is eval logic, not
operational glue. ``scripts/run_visual_eval.py`` is kept as a tiny
back-compat shim that delegates to ``main`` here.

Skip semantics — visual retrieval needs BOTH (a) verified visual gold
examples AND (b) ingested MP4 frames + ColQwen patches **for every
video_id referenced by visual_gold**. Per-video readiness is checked;
a global frame count is not enough (frames for unrelated talks would
let the script proceed and silently score zeros). If either condition
fails, the runner emits a clear ``SKIPPED: ...`` to stdout and exits 0.

This runner does NOT lock a winner. Visual retrieval has no
single-candidate ADR 004 lock today; the eval is a quality gate, not
a selection. The candidate id is resolved from
``eval/config/model_candidates.yaml::candidates.visual_retrieval.options``
(filter ``provider == 'local'``; exactly one expected) rather than
hard-coded, per the project's "model IDs from yaml" rule.

CLI:

    python -m eval.runners.run_visual_eval
        [--visual-gold PATH]   # override gold file (tests; default = corpus path)
        [--no-lift]            # skip the 5ch-vs-4ch RRF lift comparison
        [--k INT]              # top-k for both metrics (default 5)
        [--dry-run]            # print but do not write DB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import psycopg
import yaml

from db.conn import dsn as resolve_dsn
from db.repos import eval_runs

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "eval"
    / "config"
    / "model_candidates.yaml"
)


def _yaml_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_visual_candidate_id(cfg: dict) -> str:
    """Resolve the single 'current' visual candidate id from the yaml.

    The visual_retrieval candidate set today lists ColQwen2.5 alongside
    Gemini Embedding 2 as a single-vector baseline (kept for the
    writeup). The eval gate runs against the production stack, which
    is the local ColPali-backed implementation. The discriminator is
    ``provider == 'local'`` — the baseline lives under a hosted
    provider. Require exactly one match so adding a second local
    candidate forces an explicit choice rather than a silent default.
    """
    options = (
        cfg.get("candidates", {})
        .get("visual_retrieval", {})
        .get("options", [])
        or []
    )
    local = [o for o in options if isinstance(o, dict) and o.get("provider") == "local"]
    if len(local) != 1:
        raise RuntimeError(
            "Expected exactly 1 local visual_retrieval candidate in "
            f"eval/config/model_candidates.yaml; found {len(local)} "
            f"(ids: {[o.get('id') for o in local]}). The visual eval scaffold "
            "assumes one ColQwen-class candidate; amend this resolver before "
            "introducing a second."
        )
    cid = local[0].get("id")
    if not isinstance(cid, str) or not cid:
        raise RuntimeError(
            f"Local visual_retrieval candidate has no usable 'id' field: {local[0]!r}"
        )
    return cid


def substrate_per_video(
    conn: psycopg.Connection, video_ids: Iterable[str]
) -> dict[str, dict[str, int]]:
    """Return per-video counts of pooled-embedding frames and frame
    patches joined through frames. The visual channel needs BOTH > 0
    for a given ``video_id`` to score anything meaningful for examples
    on that video."""
    out: dict[str, dict[str, int]] = {}
    for vid in sorted(set(video_ids)):
        n_frames = conn.execute(
            "SELECT count(*) FROM frames WHERE video_id = %s AND pooled_embedding IS NOT NULL",
            (vid,),
        ).fetchone()[0]
        n_patches = conn.execute(
            """SELECT count(*) FROM frame_patches fp
                 JOIN frames f USING (frame_id)
                WHERE f.video_id = %s""",
            (vid,),
        ).fetchone()[0]
        out[vid] = {"pooled_frames": int(n_frames), "patches": int(n_patches)}
    return out


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
            "answer_term_hit_at_k": detail.get("answer_term_hit_at_k"),
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
    ap.add_argument(
        "--visual-gold",
        type=Path,
        default=None,
        help="Override the visual_gold.jsonl path (used by tests). "
        "Default = eval/corpora/ai_engineering_v0/visual_gold.jsonl.",
    )
    ap.add_argument("--no-lift", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from eval.runners.measure_visual import (
        _VISUAL_GOLD_PATH,
        load_visual_evidence,
        load_visual_gold,
        measure_visual,
    )

    gold_path = args.visual_gold if args.visual_gold is not None else _VISUAL_GOLD_PATH
    examples = load_visual_gold(gold_path)
    evidence_by_example = load_visual_evidence(gold_path)
    if not examples:
        sys.stdout.write(
            "SKIPPED: visual_gold.jsonl has 0 single_clip examples. "
            "Stage verified visual examples (modality includes slide / "
            "screen_code / diagram / whiteboard) before the visual eval "
            "can score anything. See eval/reports/2026-05-27_visual_eval/"
            "methodology.mdx for the staging workflow.\n"
        )
        return 0

    needed_videos = sorted({ex.video_id for ex in examples})
    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        substrate = substrate_per_video(conn, needed_videos)
    missing = sorted(
        vid
        for vid, counts in substrate.items()
        if counts["pooled_frames"] == 0 or counts["patches"] == 0
    )
    if missing:
        # Per-video skip — distinct from "no frames at all" because frames
        # may exist for unrelated talks and a global check would have
        # silently passed, scoring degenerate zeros on the visual_gold set.
        details = ", ".join(
            f"{v}(frames={substrate[v]['pooled_frames']},patches={substrate[v]['patches']})"
            for v in missing
        )
        sys.stdout.write(
            "SKIPPED: visual substrate not ingested for visual_gold video_id(s): "
            f"{details}. Run MP4 ingest + ColQwen patch worker for those videos "
            f"before invoking this eval. Visual gold examples present: {len(examples)}.\n"
        )
        return 0

    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg_sha = _yaml_sha256(_CONFIG_PATH)
    candidate_id = resolve_visual_candidate_id(cfg)

    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        summary, per_example_detail = measure_visual(
            conn,
            examples,
            k=args.k,
            include_lift=not args.no_lift,
            evidence_by_example=evidence_by_example,
        )

    summary.update(
        {
            "component": "visual_retrieval",
            "candidate_id": candidate_id,
            "measurement_mode": "real_visual_gold_sweep",
            "candidate_set_yaml_sha256": cfg_sha,
            "substrate_per_video": substrate,
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
            embedding_model_id=candidate_id,
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
