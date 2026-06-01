"""Generator bakeoff (#2) — phase-2 faithfulness + abstention runner.

PROVISIONAL by construction this session: the faithfulness judge is NOT
yet validated against a human-labelled ``boundary_audit.jsonl`` (Cohen's
kappa cannot be computed without human labels). Per CLAUDE.md hard rule 4
and ``docs/eval-methodology.md`` Judge Versioning, a generator winner is
only a *claim* once the judge clears ``cohens_kappa_vs_human_min: 0.60``.
So this runner NEVER locks a winner (``lock_bakeoff_winner`` is not
called) and never amends ADR 004. It writes honest, reproducible rows and
applies the ADR-004 selection rule descriptively, labelled provisional.

Design:

* Generators scored: ``gemma-4-31b`` (google-gemma) and
  ``qwen3-235b-a22b-instruct`` (alibaba-qwen). The judge is a CONSTANT
  ``deepseek-v3.2`` (deepseek) so the comparison holds the judge fixed
  (anti-contamination: no per-run judge swap). A constant judge forces
  excluding the generator of its own family, so ``deepseek-v3.2`` is NOT
  scored as a generator under this judge — documented in the report.
* Each candidate: measure faithfulness over ``dev_gold`` (single_clip) and
  abstention over corpus ``negative.jsonl``. One ``eval_runs`` row
  (``code_path='minimal_generation'``) + one ``eval_results`` row per
  dev_gold and per negative example, in a SINGLE transaction per candidate
  (same atomicity contract as the embeddings bakeoff).
* Selection rule (ADR 004 v3.1): cheapest candidate within
  ``tie_break_pp`` of the leader meeting ALL minimums. Score axis is
  ``claims_grounded_rate`` (the faithfulness proxy for ``claims_supported``;
  see ``measure_generation`` docstring).

CLI:

    python -m scripts.run_generator_bakeoff [--dry-run] [--limit-negatives N]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from datetime import date
from pathlib import Path

import psycopg
import yaml

from db.conn import dsn as resolve_dsn
from db.repos import eval_runs
from eval.runners import measure_generation as mg

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eval" / "config" / "model_candidates.yaml"

# Constant cross-family judge. deepseek is cross-family to both scored
# generators AND to the curation family (claude / anthropic).
JUDGE_ID = "deepseek-v3.2"
# Generators scored under the constant judge (deepseek excluded — same
# family as the judge).
GENERATOR_IDS = ("gemma-4-31b", "qwen3-235b-a22b-instruct")


def _yaml_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_minimums(gen_summary: dict, neg_summary: dict, mins: dict) -> tuple[bool, dict]:
    """Apply the per-component generator minimums. Returns
    ``(all_met, per_minimum_pass)``. A ``None`` measured value (e.g. no
    judged examples) counts as a miss."""

    def _ge(val, threshold) -> bool:
        return val is not None and float(val) >= float(threshold)

    def _le(val, threshold) -> bool:
        return val is not None and float(val) <= float(threshold)

    # Missing/None p95 must FAIL the latency minimum — a generator we could
    # not time is not one we can certify under 2s. Do NOT coerce None to 0
    # (that would silently pass an untimed candidate).
    p95_ms = gen_summary.get("p95_latency_ms")
    p95_sec = p95_ms / 1000.0 if p95_ms is not None else None

    checks = {
        "claims_supported_min": _ge(
            gen_summary.get("claims_grounded_rate"), mins["claims_supported_min"]
        ),
        "citation_accuracy_min": _ge(
            gen_summary.get("citation_accuracy"), mins["citation_accuracy_min"]
        ),
        "json_parse_success_min": _ge(
            gen_summary.get("json_parse_success"), mins["json_parse_success_min"]
        ),
        "corpus_negative_refusal_min": _ge(
            neg_summary.get("corpus_negative_refusal"), mins["corpus_negative_refusal_min"]
        ),
        "p95_latency_sec_max": _le(p95_sec, mins["p95_latency_sec_max"]),
    }
    return all(checks.values()), checks


def select_generator_winner(measurements: list[dict], tie_break_pp: float) -> dict | None:
    """ADR-004 selection: cheapest candidate within ``tie_break_pp`` of the
    leader, among those meeting all minimums. Returns ``None`` if none
    qualify. Score axis is ``claims_grounded_rate``."""
    eligible = [m for m in measurements if m.get("meets_minimums") and m.get("score") is not None]
    if not eligible:
        return None
    leader_score = max(m["score"] for m in eligible)
    band = leader_score - (tie_break_pp / 100.0)
    within = [m for m in eligible if m["score"] >= band]
    within.sort(key=lambda m: (m["price_output_per_mtok_usd"], -m["score"], m["candidate_id"]))
    return within[0]


def _candidate_price(cfg: dict, candidate_id: str) -> float:
    for opt in cfg["candidates"]["generator"]["options"]:
        if opt["id"] == candidate_id:
            return float(opt.get("price_output_per_mtok_usd") or 0.0)
    raise KeyError(candidate_id)


def persist_candidate_run(
    conn: psycopg.Connection,
    *,
    run_id: str,
    candidate_id: str,
    judge_id: str,
    judge_hash: str,
    cfg: dict,
    gen_summary: dict,
    gen_per_example: dict[str, dict],
    neg_summary: dict,
    neg_per_example: dict[str, dict],
    minimum_checks: dict,
    meets_minimums: bool,
) -> None:
    """Write one ``eval_runs`` row + per-example ``eval_results`` (dev_gold
    + negatives) for one candidate, in the caller's transaction. No lock —
    this bakeoff is provisional until the judge is kappa-validated."""
    summary = {
        "component": "generator",
        "code": "generator_bakeoff",
        "provisional": True,
        "provisional_reason": "judge not kappa-validated (boundary_audit.jsonl empty)",
        "candidate_id": candidate_id,
        "judge_id": judge_id,
        "judge_prompt_sha256": judge_hash,
        "minimum_checks": minimum_checks,
        "meets_minimums": meets_minimums,
        **gen_summary,
        **neg_summary,
    }
    eval_runs.insert_run(
        conn,
        run_id=run_id,
        run_date=date.today(),
        code_path="minimal_generation",
        chunking_strategy="fixed_window",
        embedding_model_id="bge-m3-all-channels",
        generator_model_id=candidate_id,
        judge_model_id=judge_id,
        judge_prompt_hash=judge_hash,
        candidate_set_yaml=cfg,
        summary=summary,
    )
    for example_id, rec in gen_per_example.items():
        eval_runs.insert_result(
            conn,
            run_id=run_id,
            example_id=example_id,
            system_output={"set": "dev_gold", **rec},
            metrics={
                "parse_ok": rec.get("parse_ok"),
                "abstain": rec.get("abstain"),
                "latency_ms": rec.get("latency_ms"),
                "claims_grounded_frac": rec.get("claims_grounded_frac"),
                "citation_accuracy_frac": rec.get("citation_accuracy_frac"),
            },
        )
    for example_id, rec in neg_per_example.items():
        eval_runs.insert_result(
            conn,
            run_id=run_id,
            example_id=example_id,
            system_output={"set": "negative", **rec},
            metrics={
                "abstain": rec.get("abstain"),
                "parse_ok": rec.get("parse_ok"),
                "latency_ms": rec.get("latency_ms"),
            },
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="measure + print, no DB writes")
    ap.add_argument(
        "--limit-negatives",
        type=int,
        default=None,
        help="cap negatives scored (cost control during iteration)",
    )
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    tie_break_pp = float(cfg["selection_rule"]["tie_break_pp"])
    mins = cfg["candidates"]["generator"]["minimums"]
    cfg_sha = _yaml_sha256(_CONFIG_PATH)

    judge_prompt = mg.load_judge_prompt()
    judge_hash = mg.judge_prompt_sha256(judge_prompt)

    dev_gold = mg.load_dev_gold_single_clip()
    negatives = mg.load_corpus_negatives()
    if args.limit_negatives is not None:
        negatives = negatives[: args.limit_negatives]

    sys.stdout.write(
        f"generator bakeoff (PROVISIONAL): judge={JUDGE_ID} "
        f"judge_sha={judge_hash[:12]} dev_gold={len(dev_gold)} negatives={len(negatives)}\n"
        f"generators={list(GENERATOR_IDS)} (deepseek-v3.2 excluded: shares judge family)\n"
        f"candidate_set_yaml_sha256={cfg_sha[:12]}\n\n"
    )

    measurements: list[dict] = []
    runs: list[dict] = []

    # Measure each candidate against the live retrieval surface + providers.
    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        for candidate_id in GENERATOR_IDS:
            gen_summary, gen_per_example = mg.measure_generator_over_dev_gold(
                conn,
                generator_id=candidate_id,
                judge_id=JUDGE_ID,
                judge_system=judge_prompt,
                examples=dev_gold,
            )
            neg_summary, neg_per_example = mg.measure_abstention_over_negatives(
                conn, generator_id=candidate_id, negatives=negatives
            )
            meets, checks = check_minimums(gen_summary, neg_summary, mins)
            measurements.append(
                {
                    "candidate_id": candidate_id,
                    "score": gen_summary.get("claims_grounded_rate"),
                    "price_output_per_mtok_usd": _candidate_price(cfg, candidate_id),
                    "meets_minimums": meets,
                }
            )
            runs.append(
                {
                    "candidate_id": candidate_id,
                    "gen_summary": gen_summary,
                    "gen_per_example": gen_per_example,
                    "neg_summary": neg_summary,
                    "neg_per_example": neg_per_example,
                    "checks": checks,
                    "meets": meets,
                }
            )
            sys.stdout.write(
                f"  {candidate_id}: claims_grounded={gen_summary.get('claims_grounded_rate')} "
                f"citation_acc={gen_summary.get('citation_accuracy')} "
                f"parse={gen_summary.get('json_parse_success')} "
                f"neg_refusal={neg_summary.get('corpus_negative_refusal')} "
                f"p95_ms={gen_summary.get('p95_latency_ms')} meets_min={meets}\n"
            )

    winner = select_generator_winner(measurements, tie_break_pp)
    sys.stdout.write(
        "\nselection (provisional, ADR-004 descriptive): "
        + (
            f"{winner['candidate_id']} (cheapest within {tie_break_pp}pp meeting minimums)"
            if winner
            else "NO winner — no candidate meets all minimums (see minimum_checks per run)"
        )
        + "\n"
    )

    if args.dry_run:
        sys.stdout.write("--dry-run: no DB writes\n")
        return 0

    # One transaction per candidate: insert_run + all per-example results
    # commit together; a mid-loop failure rolls back that candidate's run
    # with zero partial state. No lock is ever written (provisional).
    written: list[str] = []
    for r in runs:
        run_id = f"generator-bakeoff-{uuid.uuid4().hex[:12]}"
        with psycopg.connect(resolve_dsn()) as conn:
            persist_candidate_run(
                conn,
                run_id=run_id,
                candidate_id=r["candidate_id"],
                judge_id=JUDGE_ID,
                judge_hash=judge_hash,
                cfg=cfg,
                gen_summary=r["gen_summary"],
                gen_per_example=r["gen_per_example"],
                neg_summary=r["neg_summary"],
                neg_per_example=r["neg_per_example"],
                minimum_checks=r["checks"],
                meets_minimums=r["meets"],
            )
        written.append(run_id)
        sys.stdout.write(f"  wrote {r['candidate_id']} → {run_id}\n")

    sys.stdout.write(
        f"\nDONE (provisional). {len(written)} candidate run(s) written; NO winner locked "
        f"(judge unvalidated). run_ids: {written}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
