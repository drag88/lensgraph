"""Bakeoff-prep readiness report.

`python -m scripts.bakeoff_prep [--corpus <name>]` walks dev_gold +
synthesis JSONL files for a corpus, collects every referenced video_id
(top-level and per-span), and reports per-video readiness across the
ingest pipeline.

A channel is reported ready iff its rowcount equals chunks_n. The prior
boolean formulation reported "ready" for any non-empty channel and hid
partial coverage (e.g. 47/50 dense embeds rendered as ✓).

Read-only — never writes to the DB. Returns 0 unconditionally so callers
can treat it as a status report, not a gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

from db.conn import dsn as resolve_dsn
from ingest import readiness

REPO_ROOT = Path(__file__).resolve().parent.parent


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _fmt_total(n: int) -> str:
    return f"{_mark(n > 0)} ({n})"


def _fmt_ratio(n: int, denom: int) -> str:
    ok = denom > 0 and n == denom
    return f"{_mark(ok)} ({n}/{denom})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts.bakeoff_prep",
        description="Report ingest readiness per dev_gold video.",
    )
    parser.add_argument(
        "--corpus",
        default="ai_engineering_v0",
        help="corpus directory under eval/corpora/ (default: ai_engineering_v0)",
    )
    ns = parser.parse_args(argv or [])

    corpus_dir = REPO_ROOT / "eval" / "corpora" / ns.corpus

    with psycopg.connect(resolve_dsn()) as conn:
        reports = readiness.for_corpus(conn, corpus_dir)

    print(f"bakeoff-prep: corpus={ns.corpus} videos={len(reports)}")
    header = (
        f"{'video_id':<34} "
        f"{'talks':<5}  "
        f"{'chunks':<10}  "
        f"{'dense':<12}  "
        f"{'sparse':<12}  "
        f"{'tokens':<12}"
    )
    print(header)
    for r in reports:
        print(
            f"{r.video_id:<34} "
            f"{_mark(r.talks_present):<5}  "
            f"{_fmt_total(r.chunks_n):<10}  "
            f"{_fmt_ratio(r.dense_n, r.chunks_n):<12}  "
            f"{_fmt_ratio(r.sparse_n, r.chunks_n):<12}  "
            f"{_fmt_ratio(r.tokens_n, r.chunks_n):<12}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
