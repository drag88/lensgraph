"""Chunking ablation (PROVISIONAL) — fixed_window vs transcript_segment.

Retrieval/substrate-focused: chunks + embeds the 3 talks under
``transcript_segment`` (coexisting with the existing ``fixed_window`` rows via
the chunks natural key), then compares the two strategies on the SAME dev_gold
queries and the SAME retrieval channels.

PROVISIONAL by construction:
  * No winner / selection. The boundary judge is not cold-kappa-validated
    (only a provisional edge-kappa exists), so edge/standalone quality is NOT
    scored here — this runner measures retrieval substrate only (TR@5, chunk
    counts, durations, embed alignment).
  * Writes one ``eval_runs`` row per strategy (``code_path='chunking_ablation'``)
    with per-example ``eval_results``; never locks a winner.

Phases (``--measure-only`` skips phase 1):
  1. Re-chunk + re-embed ``transcript_segment`` (delete-first for idempotency,
     reuse the strategy-aware ``chunk_handler`` + drain ``ingest_embed_text``).
  2. Measure both strategies and persist + report.

CLI: ``python -m scripts.run_chunking_ablation [--measure-only]``
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date
from pathlib import Path

import psycopg
import yaml

from db.conn import dsn as resolve_dsn
from db.repos import eval_runs
from eval.runners import measure_embeddings as me
from retrieve import bm25, dense, multivec, rrf, sparse

VIDS = ("W_CYk2ogcDI", "aie_sg_2026_d2_arize_alyx", "nXafozNIk3c")
BASELINE = "fixed_window"
CANDIDATE = "transcript_segment"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eval" / "config" / "model_candidates.yaml"
_REPORT_DIR = Path(__file__).resolve().parent.parent / "eval" / "reports"


# ---- phase 1: chunk + embed the candidate strategy -----------------------


def rechunk_candidate(conn: psycopg.Connection) -> None:
    """Delete any existing ``transcript_segment`` rows (idempotent re-run) and
    re-chunk via the strategy-aware handler, which fans out embed_text jobs."""
    from ingest.handlers import chunk_handler

    for vid in VIDS:
        # Stale per-chunk embed_text status rows for this strategy's chunk_ids.
        conn.execute(
            """DELETE FROM ingest_step_status WHERE step='embed_text' AND entity_id IN
               (SELECT chunk_id FROM chunks WHERE video_id=%s AND chunking_strategy=%s)""",
            (vid, CANDIDATE),
        )
        deleted = conn.execute(
            "DELETE FROM chunks WHERE video_id=%s AND chunking_strategy=%s",
            (vid, CANDIDATE),
        ).rowcount
        chunk_handler(
            conn, {"video_id": vid, "step": "chunk", "entity_id": 0, "strategy": CANDIDATE}
        )
        n = conn.execute(
            "SELECT count(*) FROM chunks WHERE video_id=%s AND chunking_strategy=%s",
            (vid, CANDIDATE),
        ).fetchone()[0]
        sys.stdout.write(f"  {vid}: deleted {deleted} old {CANDIDATE}, now {n}\n")


# ---- measurement ---------------------------------------------------------


def _channel_fns(strategy: str) -> dict:
    """Strategy-bound channel callables shaped for measure_embeddings.tr_at_k."""

    def _dense(conn, q):
        return dense.retrieve(conn, q, top_k=5, chunking_strategy=strategy)

    def _sparse(conn, q):
        return sparse.retrieve(conn, q, top_k=5, chunking_strategy=strategy)

    def _multivec(conn, q):
        return multivec.retrieve(conn, q, top_k=5, chunking_strategy=strategy)

    def _rrf_4ch(conn, q):
        b = bm25.retrieve(conn, q, top_k=30, chunking_strategy=strategy)
        d = dense.retrieve(conn, q, top_k=30, chunking_strategy=strategy)
        s = sparse.retrieve(conn, q, top_k=30, chunking_strategy=strategy)
        m = multivec.retrieve(conn, q, top_k=30, chunking_strategy=strategy)
        return rrf.fuse({"bm25": b, "dense": d, "sparse": s, "multivec": m}, top_k=5)

    return {"dense": _dense, "sparse": _sparse, "multivec": _multivec, "rrf_4ch": _rrf_4ch}


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(round(0.95 * len(ordered) + 0.5)) - 1))
    return ordered[rank]


def chunk_stats(conn: psycopg.Connection, strategy: str) -> dict:
    """Per-talk chunk count + avg/p95 duration, plus channel-alignment counts."""
    per_talk = {}
    total = 0
    for vid in VIDS:
        durs = [
            float(e) - float(s)
            for s, e in conn.execute(
                "SELECT start_sec, end_sec FROM chunks WHERE video_id=%s AND chunking_strategy=%s",
                (vid, strategy),
            ).fetchall()
        ]
        total += len(durs)
        per_talk[vid] = {
            "n": len(durs),
            "avg_dur": round(sum(durs) / len(durs), 1) if durs else 0.0,
            "p95_dur": round(_p95(durs), 1),
        }
    # embed alignment (distinct chunk_ids per channel == chunk count)
    align = {}
    for label, sql in (
        ("dense", "dense_embeds"),
        ("sparse", "sparse_embeds"),
        ("token", "chunk_token_embeds"),
    ):
        align[label] = conn.execute(
            f"SELECT count(DISTINCT t.chunk_id) FROM {sql} t "
            f"JOIN chunks c ON c.chunk_id=t.chunk_id WHERE c.chunking_strategy=%s",
            (strategy,),
        ).fetchone()[0]
    tsv = conn.execute(
        "SELECT count(*) FROM chunks WHERE chunking_strategy=%s AND tsv IS NOT NULL", (strategy,)
    ).fetchone()[0]
    return {
        "n_total": total,
        "per_talk": per_talk,
        "embed_alignment": {**align, "tsv": tsv, "chunks": total},
        "aligned": align["dense"] == align["sparse"] == align["token"] == tsv == total,
    }


def measure_strategy(conn: psycopg.Connection, strategy: str, examples) -> tuple[dict, dict]:
    """Per-channel TR@5 + per-example detail for one strategy."""
    fns = _channel_fns(strategy)
    per_channel = {}
    per_example_pass = {ex.example_id: {} for ex in examples}
    per_example_topk = {ex.example_id: {} for ex in examples}
    for name, fn in fns.items():
        score, passes, top_ks = me.tr_at_k(conn, examples, fn, k=5)
        per_channel[name] = score
        for ex, p, tk in zip(examples, passes, top_ks, strict=True):
            per_example_pass[ex.example_id][name] = bool(p)
            per_example_topk[ex.example_id][name] = tk
    summary = {
        "per_channel_tr_at_5": per_channel,
        "vector_best_score": max(per_channel[c] for c in ("dense", "sparse", "multivec")),
        "n_examples": len(examples),
        **chunk_stats(conn, strategy),
    }
    detail = {
        ex.example_id: {
            "gold_span": {
                "video_id": ex.video_id,
                "start_sec": ex.start_sec,
                "end_sec": ex.end_sec,
            },
            "pass_at_5": per_example_pass[ex.example_id],
            "top_k_per_channel": per_example_topk[ex.example_id],
        }
        for ex in examples
    }
    return summary, detail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure-only", action="store_true", help="skip phase 1 (chunk+embed)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    examples = me.load_dev_gold_single_clip()

    if not args.measure_only:
        sys.stdout.write(f"phase 1: re-chunk + re-embed {CANDIDATE}\n")
        with psycopg.connect(resolve_dsn()) as conn:
            rechunk_candidate(conn)
            conn.commit()
        from scripts.drain_ingest_queue import drain

        n = drain("ingest_embed_text")
        sys.stdout.write(f"  embedded {n} {CANDIDATE} chunks\n")

    sys.stdout.write(f"\nphase 2: measure (dev_gold n={len(examples)})\n")
    run_ids = {}
    with psycopg.connect(resolve_dsn(), autocommit=True) as rconn:
        results = {s: measure_strategy(rconn, s, examples) for s in (BASELINE, CANDIDATE)}

    # persist one eval_runs row per strategy (no winner, provisional)
    for strategy, (summary, detail) in results.items():
        run_id = f"chunking-ablation-{strategy[:4]}-{uuid.uuid4().hex[:8]}"
        run_ids[strategy] = run_id
        with psycopg.connect(resolve_dsn()) as conn:
            eval_runs.insert_run(
                conn,
                run_id=run_id,
                run_date=date.today(),
                code_path="chunking_ablation",
                chunking_strategy=strategy,
                embedding_model_id="bge-m3-all-channels",
                candidate_set_yaml=cfg,
                summary={"component": "chunking", "provisional": True, **summary},
            )
            for ex_id, d in detail.items():
                eval_runs.insert_result(
                    conn,
                    run_id=run_id,
                    example_id=ex_id,
                    system_output={
                        "strategy": strategy,
                        "gold_span": d["gold_span"],
                        "top_k_per_channel": d["top_k_per_channel"],
                    },
                    metrics={"strategy": strategy, "pass_at_5_per_channel": d["pass_at_5"]},
                )

    _report(results, run_ids)
    return 0


def _report(results: dict, run_ids: dict) -> None:
    def fmt(s):
        sm = results[s][0]
        ch = sm["per_channel_tr_at_5"]
        return ch, sm

    sys.stdout.write("\n=== CHUNKING ABLATION (PROVISIONAL — no winner) ===\n")
    sys.stdout.write(
        f"{'strategy':20} {'chunks':>7} {'dense':>6} {'sparse':>6} {'multivec':>9} {'rrf_4ch':>8} {'aligned':>8}\n"
    )
    for s in (BASELINE, CANDIDATE):
        ch, sm = fmt(s)
        sys.stdout.write(
            f"{s:20} {sm['n_total']:>7} {ch['dense']:>6.2f} {ch['sparse']:>6.2f} "
            f"{ch['multivec']:>9.2f} {ch['rrf_4ch']:>8.2f} {str(sm['aligned']):>8}\n"
        )
    sys.stdout.write("\nper-talk chunk count / avg / p95 duration:\n")
    for s in (BASELINE, CANDIDATE):
        sm = results[s][0]
        parts = [
            f"{v}: n={d['n']} avg={d['avg_dur']}s p95={d['p95_dur']}s"
            for v, d in sm["per_talk"].items()
        ]
        sys.stdout.write(f"  {s}: " + " | ".join(parts) + "\n")
    sys.stdout.write(f"\nrun_ids: {run_ids}\n")
    sys.stdout.write(
        "PROVISIONAL: retrieval substrate only; boundary edge/standalone NOT scored "
        "(judge not cold-kappa-validated). No winner.\n"
    )

    # write a provisional methodology report
    out = _REPORT_DIR / f"{date.today().isoformat()}_chunking_ablation"
    out.mkdir(parents=True, exist_ok=True)
    import json

    (out / "summary.json").write_text(
        json.dumps(
            {
                "provisional": True,
                "run_ids": run_ids,
                "strategies": {s: results[s][0] for s in results},
            },
            indent=2,
        )
    )
    sys.stdout.write(f"wrote {out}/summary.json\n")


if __name__ == "__main__":
    raise SystemExit(main())
