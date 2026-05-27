"""Embeddings bakeoff — phase-2 real measurement runner.

Applies the ADR 004 v3.1 selection rule (cheapest candidate within
``selection_rule.tie_break_pp`` of the leader AND meeting
``candidates.text_embeddings.minimums``) and locks the winner via
``db.repos.eval_runs.lock_bakeoff_winner('text_embeddings', ...)``.

Two measurement modes:

* **Real (default)** — runs a per-channel ``TimestampRecall@5`` sweep over
  ``dev_gold.jsonl`` via ``eval/runners/measure_embeddings.py``, applies
  the selection rule against the embeddings minimum, and writes one row
  to ``eval_runs`` (``code_path='embeddings_bakeoff'``). Locks the winner
  iff the best vector-channel score clears the minimum.

* **From-file** — ``--measurements-json PATH`` reads measurements from a
  JSON file. Used by the step-35a RED test to exercise the selection
  rule deterministically without a populated DB.

Failing-minimum behaviour: the row is still written (honest recording of
the failing measurement so the methodology writeup can quote it), but
``winner_locked`` is left false and the script exits with code 1. That
signals downstream callers ("ingest gating on the locked embedder") not
to promote the candidate, and prompts an ADR 004 amendment per Mission 2
in the phase-2 handoff.

CLI:

    python -m scripts.run_embeddings_bakeoff
        [--measurements-json PATH]   # bypass real sweep; tests only
        [--dry-run]                  # print but don't write DB
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


def _select_winner(measurements: list[dict], tie_break_pp: float) -> dict:
    eligible = [m for m in measurements if m.get("meets_minimums")]
    if not eligible:
        raise RuntimeError("no candidate meets minimums — selection cannot proceed")
    leader_score = max(m["score"] for m in eligible)
    tie_band = leader_score - (tie_break_pp / 100.0)
    within_band = [m for m in eligible if m["score"] >= tie_band]
    # Cheapest within the band; tie-break by score descending then id for determinism.
    within_band.sort(
        key=lambda m: (m["price_per_mtok_usd"], -m["score"], m["candidate_id"])
    )
    return within_band[0]


def _yaml_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _embeddings_min_threshold(cfg: dict) -> float:
    return float(
        cfg["candidates"]["text_embeddings"]["minimums"][
            "timestamp_recall_at_5_vector_only_min"
        ]
    )


def _run_real_measurement(cfg: dict) -> tuple[list[dict], dict]:
    """Run a real per-channel TR@5 sweep against the live DB.

    Returns ``(measurements_for_selection, extra_summary)`` where the
    measurements list is shaped for ``_select_winner`` and the extra
    summary holds the per-channel breakdown the methodology writeup
    quotes.
    """
    from eval.runners.measure_embeddings import (
        load_dev_gold_single_clip,
        measure_all_channels,
    )

    examples = load_dev_gold_single_clip()
    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        sweep = measure_all_channels(conn, examples)

    min_threshold = _embeddings_min_threshold(cfg)
    vector_best = sweep["vector_best_score"]
    meets = vector_best >= min_threshold

    measurement = {
        "candidate_id": "bge-m3-all-channels",
        "score": vector_best,
        "price_per_mtok_usd": 0.0,
        "meets_minimums": meets,
    }
    extra = {
        **sweep,
        "embeddings_min_threshold": min_threshold,
        "meets_minimums": meets,
        "measurement_mode": "real_dev_gold_sweep",
    }
    return [measurement], extra


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurements-json", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    tie_break_pp = float(cfg["selection_rule"]["tie_break_pp"])
    cfg_sha = _yaml_sha256(_CONFIG_PATH)

    if args.measurements_json is None:
        measurements, extra_summary = _run_real_measurement(cfg)
    else:
        measurements = json.loads(args.measurements_json.read_text(encoding="utf-8"))
        extra_summary = {"measurement_mode": "from_file"}

    extra_summary.update(
        {
            "measurements": measurements,
            "tie_break_pp": tie_break_pp,
            "candidate_set_yaml_sha256": cfg_sha,
        }
    )

    run_id = f"embeddings-bakeoff-{uuid.uuid4().hex[:12]}"
    eligible = [m for m in measurements if m.get("meets_minimums")]

    # Failing-minimum path: write the row honestly, but DO NOT lock and
    # exit non-zero so an ``&&`` chain catches it.
    if not eligible:
        leader = max(measurements, key=lambda m: m["score"])
        extra_summary.update(
            {
                "leader_score": leader["score"],
                "leader_candidate_id": leader["candidate_id"],
                "winner_locked": False,
                "component": "text_embeddings",
                "minimum_check": "FAILED",
            }
        )
        if args.dry_run:
            sys.stdout.write(
                json.dumps(
                    {
                        "would_not_lock": True,
                        "run_id": run_id,
                        "extra_summary": extra_summary,
                    },
                    indent=2,
                )
                + "\n"
            )
            return 1
        with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
            eval_runs.insert_run(
                conn,
                run_id=run_id,
                run_date=date.today(),
                code_path="embeddings_bakeoff",
                chunking_strategy="fixed_window",
                embedding_model_id=leader["candidate_id"],
                candidate_set_yaml=cfg,
                summary=extra_summary,
            )
        threshold = extra_summary.get("embeddings_min_threshold", 0.75)
        sys.stdout.write(
            f"FAILED: no candidate met embeddings minimum "
            f"(best score {leader['score']:.3f} < {threshold:.2f}). "
            f"row written: {run_id}; winner NOT locked.\n"
            f"Next step: amend ADR 004 (v3.2) before promoting Voyage / Gemini.\n"
        )
        return 1

    winner = _select_winner(measurements, tie_break_pp)
    extra_summary.update(
        {
            "leader_score": max(m["score"] for m in eligible),
            "winner_score": winner["score"],
        }
    )

    if args.dry_run:
        sys.stdout.write(
            json.dumps(
                {
                    "would_lock": winner["candidate_id"],
                    "run_id": run_id,
                    "extra_summary": extra_summary,
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        eval_runs.insert_run(
            conn,
            run_id=run_id,
            run_date=date.today(),
            code_path="embeddings_bakeoff",
            chunking_strategy="fixed_window",
            embedding_model_id=winner["candidate_id"],
            candidate_set_yaml=cfg,
            summary={"component": "text_embeddings"},
        )
        eval_runs.lock_bakeoff_winner(
            conn,
            component="text_embeddings",
            candidate_id=winner["candidate_id"],
            run_id=run_id,
            extra_summary=extra_summary,
        )
    sys.stdout.write(
        f"locked text_embeddings winner: {winner['candidate_id']} ({run_id})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
