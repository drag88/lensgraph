"""Bakeoff-prep readiness report.

`python -m scripts.bakeoff_prep` walks the dev_gold + synthesis JSONL files
for a corpus, collects every referenced video_id (top-level and per-span),
and reports per-video readiness across the ingest pipeline:

  - talks row present
  - chunks count > 0
  - dense_embeds count > 0
  - sparse_embeds count > 0

Read-only — never writes to the DB. Returns 0 unconditionally so callers
can treat it as a status report, not a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg

from db.conn import dsn as resolve_dsn

REPO_ROOT = Path(__file__).resolve().parent.parent


def _collect_video_ids(corpus_dir: Path) -> list[str]:
    """Read dev_gold.jsonl + synthesis.jsonl; return sorted unique video_ids
    from top-level `video_id` and any `gold_spans[*].video_id`."""
    video_ids: set[str] = set()
    for name in ("dev_gold.jsonl", "synthesis.jsonl"):
        path = corpus_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                top = row.get("video_id")
                if top:
                    video_ids.add(top)
                for span in row.get("gold_spans", []) or []:
                    sv = span.get("video_id")
                    if sv:
                        video_ids.add(sv)
    return sorted(video_ids)


def _probe(conn: psycopg.Connection, video_id: str) -> tuple[bool, int, int, int]:
    row = conn.execute("SELECT 1 FROM talks WHERE video_id = %s", (video_id,)).fetchone()
    has_talk = row is not None
    chunks_n = conn.execute(
        "SELECT count(*) FROM chunks WHERE video_id = %s", (video_id,)
    ).fetchone()[0]
    dense_n = conn.execute(
        "SELECT count(*) FROM dense_embeds d "
        "JOIN chunks c ON c.chunk_id = d.chunk_id "
        "WHERE c.video_id = %s",
        (video_id,),
    ).fetchone()[0]
    sparse_n = conn.execute(
        "SELECT count(*) FROM sparse_embeds s "
        "JOIN chunks c ON c.chunk_id = s.chunk_id "
        "WHERE c.video_id = %s",
        (video_id,),
    ).fetchone()[0]
    return has_talk, int(chunks_n), int(dense_n), int(sparse_n)


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


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
    video_ids = _collect_video_ids(corpus_dir)

    print(f"bakeoff-prep: corpus={ns.corpus} videos={len(video_ids)}")
    print(f"{'video_id':<32} {'talks':>5}  {'chunks':>6}  {'dense':>5}  {'sparse':>6}")

    with psycopg.connect(resolve_dsn()) as conn:
        for vid in video_ids:
            has_talk, chunks_n, dense_n, sparse_n = _probe(conn, vid)
            print(
                f"{vid:<32} "
                f"{_mark(has_talk):>5}  "
                f"{_mark(chunks_n > 0):>6}  "
                f"{_mark(dense_n > 0):>5}  "
                f"{_mark(sparse_n > 0):>6}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
