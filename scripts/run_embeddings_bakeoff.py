"""Embeddings bakeoff — phase-2 wiring, scaffolded in phase 1 (step 35a).

This script applies the ADR 004 selection rule (cheapest candidate
within ``selection_rule.tie_break_pp`` of the leader AND meeting the
``candidates.text_embeddings.minimums``) and locks the winner via
``db.repos.eval_runs.lock_bakeoff_winner('text_embeddings', ...)``.

Phase 1 only ships the scaffold so the contract is testable. Phase 2
will replace the measurements file with a real per-channel
TimestampRecall@5 sweep over the dev corpus.

CLI:

    python -m scripts.run_embeddings_bakeoff
        [--measurements-json PATH]   # default: synthetic stub
        [--dry-run]                  # print but don't write DB

Measurements file shape: a JSON array of objects with keys
``candidate_id`` (yaml id), ``score`` (TimestampRecall@5 0..1),
``price_per_mtok_usd`` (≥0), ``meets_minimums`` (bool).
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurements-json", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    tie_break_pp = float(cfg["selection_rule"]["tie_break_pp"])

    if args.measurements_json is None:
        # Synthetic stub — bge-m3 wins on cost ties (phase-1 default).
        measurements = [
            {
                "candidate_id": "bge-m3-all-channels",
                "score": 0.80,
                "price_per_mtok_usd": 0.0,
                "meets_minimums": True,
            }
        ]
    else:
        measurements = json.loads(args.measurements_json.read_text(encoding="utf-8"))

    winner = _select_winner(measurements, tie_break_pp)
    run_id = f"embeddings-bakeoff-{uuid.uuid4().hex[:12]}"

    extra_summary = {
        "measurements": measurements,
        "leader_score": max(
            m["score"] for m in measurements if m.get("meets_minimums")
        ),
        "winner_score": winner["score"],
        "tie_break_pp": tie_break_pp,
    }

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
    sys.stdout.write(f"locked text_embeddings winner: {winner['candidate_id']} ({run_id})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
