"""Step-35 bakeoff-prep sweep — runs ``minimal_generation.generate_for_bakeoff``
across every dev_gold example with one explicit generator candidate.

Design step 35 (phase-1 week-5):
    "One-off run of minimal_generation.generate_for_bakeoff on all 11 dev_gold
    examples with ONE generator candidate (arbitrarily selected — NOT 'the
    winner') produces per-example latency, parse_ok, citations into eval_runs
    (code_path='minimal_generation'). UNLOCKS phase 2."

The candidate id passed via ``--generator`` is reproducibly-named in the
artifact via the sweep_id; phase-2's bakeoff runner picks the actual
candidate set per ADR 004 v3.1 and is the one that locks a winner.

Retrieval is held constant across examples — 5 channels → RRF (top 30) →
BGE-reranker (top 8). The generator sees only the post-rerank chunks; no
gold labels are ever exposed.

Cost: ~$0.001-0.002 per example × 11 examples ≈ $0.01-0.02 against
DeepInfra. Latency: ~30-90s per generation call + ~5-15s retrieval; warm
total ~8-12 minutes on this box.

CLI:
    uv run python -m scripts.run_bakeoff_prep_sweep \\
        --generator qwen3-235b-a22b-instruct \\
        --corpus ai_engineering_v0
        [--limit N]                  # stop after N examples (smoke run)
        [--sweep-id <slug>]          # default: ts-prefixed slug
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from db.conn import dsn as resolve_dsn
from eval.runners import minimal_generation as mg
from eval.runners import providers
from retrieve import bm25, dense, multivec, rerank, rrf, sparse, visual

CORPUS_DIR_TEMPLATE = "eval/corpora/{corpus}"
# Phase-0 gate counts single_clip + synthesis non-negatives across the
# whole corpus (10 dev_gold.jsonl + 1 synthesis.jsonl = 11 today). Design
# step 35 says "all 11 dev_gold examples" — the phrasing means the
# phase-0-counted set, not literally one file. Sweep both.
_GOLD_FILES = ("dev_gold.jsonl", "synthesis.jsonl")


def _load_dev_gold(corpus: str) -> list[dict]:
    """Read all non-negative committed gold examples for ``corpus``. We
    skip negatives because they carry no generation signal (CLAUDE.md
    phase-0-gate semantics)."""
    base = Path(CORPUS_DIR_TEMPLATE.format(corpus=corpus))
    examples: list[dict] = []
    for fname in _GOLD_FILES:
        path = base / fname
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            examples.append(json.loads(line))
    return [ex for ex in examples if ex.get("question_type") != "negative"]


def _retrieve_chunks(conn, query: str) -> list[mg.RetrievedChunkLite]:
    """Run the 5-channel retrieve + RRF + BGE-rerank and return the top-8
    as RetrievedChunkLite. Mirrors generate/nodes/retrieve + rerank
    semantics but without LangGraph wrapping (and without the synthesis
    sub-query fan-out — sweep uses the raw question for parity across
    examples)."""
    channels = {
        "bm25": bm25.retrieve(conn, query, top_k=30),
        "dense": dense.retrieve(conn, query, top_k=30),
        "sparse": sparse.retrieve(conn, query, top_k=30),
        "multivec": multivec.retrieve(conn, query, top_k=30),
        "visual": visual.retrieve(conn, query, top_k=30),
    }
    fused = rrf.fuse(channels, top_k=30)
    reranked = rerank.rerank(conn, query, fused, top_k=8)
    return [
        mg.RetrievedChunkLite(
            chunk_id=r.chunk_id,
            video_id=r.video_id,
            start_sec=r.start_sec,
            end_sec=r.end_sec,
            text=r.text,
        )
        for r in reranked
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--generator", required=True, help="yaml candidate id, e.g. qwen3-235b-a22b-instruct")
    p.add_argument("--corpus", default="ai_engineering_v0")
    p.add_argument("--limit", type=int, default=None, help="stop after N examples")
    p.add_argument("--sweep-id", default=None, help="sweep slug for eval_runs.run_id prefix")
    args = p.parse_args(argv)

    sweep_id = args.sweep_id or datetime.now(UTC).strftime("sweep-%Y%m%dT%H%M%S")
    examples = _load_dev_gold(args.corpus)
    if args.limit:
        examples = examples[: args.limit]

    print(f"sweep_id={sweep_id} generator={args.generator} examples={len(examples)}", flush=True)

    written = 0
    skipped: list[tuple[str, str]] = []
    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        for i, ex in enumerate(examples, 1):
            ex_id = ex["id"]
            query = ex["question"]
            run_id = f"{sweep_id}-{ex_id}"
            t0 = time.perf_counter()
            try:
                chunks = _retrieve_chunks(conn, query)
            except Exception as e:
                skipped.append((ex_id, f"retrieve_error: {e}"))
                print(f"  [{i}/{len(examples)}] {ex_id} RETRIEVE_ERROR: {e}", flush=True)
                continue
            if not chunks:
                skipped.append((ex_id, "no_chunks_retrieved"))
                print(f"  [{i}/{len(examples)}] {ex_id} SKIPPED no_chunks", flush=True)
                continue
            try:
                out = mg.generate_for_bakeoff(
                    conn=conn,
                    example_id=ex_id,
                    query=query,
                    chunks=chunks,
                    candidate_id=args.generator,
                    run_id=run_id,
                )
            except providers.ProviderError as e:
                skipped.append((ex_id, f"provider_error: {e}"))
                print(f"  [{i}/{len(examples)}] {ex_id} PROVIDER_ERROR: {e}", flush=True)
                continue
            elapsed = int((time.perf_counter() - t0) * 1000)
            print(
                f"  [{i}/{len(examples)}] {ex_id} "
                f"parse_ok={out['parse_ok']} abstain={out['parsed']['abstain']} "
                f"latency_ms={out['latency_ms']} total_ms={elapsed} "
                f"claims={len(out['parsed']['claims'])} citations={len(out['parsed']['citations'])}",
                flush=True,
            )
            written += 1

    print(f"\nDONE. wrote {written}/{len(examples)} eval_runs rows. skipped={len(skipped)}", flush=True)
    if skipped:
        print("skipped detail:", flush=True)
        for ex_id, reason in skipped:
            print(f"  {ex_id}: {reason}", flush=True)
    return 0 if written > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
