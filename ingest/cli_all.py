"""Corpus-wide ingest entry point.

`python -m ingest.cli_all --corpus ai_engineering_v0`

For each talk in the corpus's talks.yaml:
  1. fetch_reconcile (idempotent — talks row + ingest_step_status + chunk
     enqueue, no embeddings yet).

After all talks processed:
  2. Drain ingest_chunk queue via process_one(chunk_handler) until empty.
  3. Drain ingest_embed_text queue via process_one(embed_text_handler)
     until empty. The first embed_text call loads the BGE-M3 model
     (~10s warm, ~78s cold). Subsequent calls per chunk: ~1s on MPS.
  4. Drain ingest_frames queue via process_one(frames_handler) until empty.
     Requires pre-staged .mp4 files under ``videos/<corpus>/<id>.mp4`` —
     handler raises FileNotFoundError per video otherwise (the message
     rolls back to the queue and the rest of the corpus continues).
     embed_frames messages enqueued here drain in slice 2.

One-shot driver — runs to completion and exits. The long-running multi-queue
worker arrives in a later slice.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psycopg
import yaml

from db.conn import dsn as resolve_dsn
from db.repos.talks import Talk
from ingest import readiness
from ingest.fetch import LocalFsFetcher, paths_from_talks_yaml
from ingest.handlers import chunk_handler, embed_text_handler, frames_handler
from ingest.pipeline import fetch_reconcile
from queues import pgmq_client, workers

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class _Args:
    corpus: str


def _parse_args(argv: list[str]) -> _Args:
    parser = argparse.ArgumentParser(
        prog="ingest.cli_all",
        description="Ingest every talk in a corpus end-to-end via local transcripts.",
    )
    parser.add_argument(
        "--corpus",
        default="ai_engineering_v0",
        help="corpus directory under eval/corpora/ (default: ai_engineering_v0)",
    )
    ns = parser.parse_args(argv)
    return _Args(corpus=ns.corpus)


def _load_talks(corpus_dir: Path) -> list[dict]:
    talks_yaml = corpus_dir / "talks.yaml"
    with talks_yaml.open(encoding="utf-8") as f:
        return list(yaml.safe_load(f))


def _build_talk(entry: dict) -> Talk:
    return Talk(
        video_id=entry["video_id"],
        title=entry["title"],
        speaker=entry["speaker"],
        url=entry["url"],
        license=entry["license"],
        captions_source=entry["captions_source"],
        transcript_path=entry["transcript_path"],
        transcript_sha256=entry["transcript_sha256"],
        duration_sec=int(entry["duration_sec"]),
        format_tags=list(entry["format_tags"]),
        accessed_at=datetime.fromisoformat(entry["accessed_at"].replace("Z", "+00:00")),
        source_video_id=entry.get("source_video_id"),
        source_start_sec=entry.get("source_start_sec"),
        source_end_sec=entry.get("source_end_sec"),
        notes=entry.get("notes"),
        ingested_at=None,
    )


def _drain(
    conn: psycopg.Connection,
    queue: str,
    handler: Callable[[psycopg.Connection, dict], None],
) -> int:
    """Pull messages off `queue` and run them through `handler` one at a time
    inside the caller's autocommit connection. Each process_one call opens
    its own transaction so a handler crash redelivers only that message.
    Returns the count drained."""
    drained = 0
    while True:
        with conn.transaction():
            processed = workers.process_one(conn, queue, handler)
        if not processed:
            return drained
        drained += 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or [])

    corpus_dir = REPO_ROOT / "eval" / "corpora" / args.corpus
    talks = _load_talks(corpus_dir)

    fetcher = LocalFsFetcher(
        paths=paths_from_talks_yaml(corpus_dir, repo_root=REPO_ROOT),
    )

    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        pgmq_client.ensure_queues(conn)

        per_video_fetch: dict[str, str] = {}
        for entry in talks:
            talk = _build_talk(entry)
            try:
                with conn.transaction():
                    fetch_reconcile(conn, talk, fetcher)
                per_video_fetch[talk.video_id] = "ok"
                print(f"fetch_reconcile ok: {talk.video_id}")
            except Exception as exc:  # noqa: BLE001 — report-and-continue is the design
                per_video_fetch[talk.video_id] = f"fetch_error: {exc!r}"
                print(f"fetch_reconcile FAIL: {talk.video_id}: {exc!r}")

        chunk_drained = _drain(conn, "ingest_chunk", chunk_handler)
        print(f"drained {chunk_drained} from ingest_chunk")

        embed_drained = _drain(conn, "ingest_embed_text", embed_text_handler)
        print(f"drained {embed_drained} from ingest_embed_text")

        # Frames drain depends on pre-staged .mp4 files under
        # ``videos/<corpus>/``. Without those, frames_handler raises
        # FileNotFoundError and _drain would retry-loop forever (the
        # message becomes visible again on every rollback). Skip cleanly
        # so operators can stage videos out of band without blocking the
        # text-only ingest.
        videos_root = REPO_ROOT / "videos" / args.corpus
        has_videos = videos_root.exists() and any(videos_root.iterdir())
        if has_videos:
            frames_drained = _drain(conn, "ingest_frames", frames_handler)
            print(f"drained {frames_drained} from ingest_frames")
        else:
            frames_drained = 0
            print(
                f"skipping ingest_frames drain — no files under "
                f"{videos_root.relative_to(REPO_ROOT)}; stage .mp4 files "
                "there and rerun to populate frames"
            )

        print(
            f"summary: corpus={args.corpus} videos={len(talks)} "
            f"chunk_msgs={chunk_drained} embed_msgs={embed_drained} "
            f"frames_msgs={frames_drained}"
        )
        for vid, status in per_video_fetch.items():
            print(f"  {vid}: {status}")

        fetch_failed = any(status != "ok" for status in per_video_fetch.values())

        reports = readiness.for_corpus(conn, corpus_dir)
        incomplete = [r for r in reports if not r.complete]
        for r in incomplete:
            print(
                f"INCOMPLETE: {r.video_id} "
                f"talks={'y' if r.talks_present else 'n'} "
                f"chunks={r.chunks_n} "
                f"dense={r.dense_n}/{r.chunks_n} "
                f"sparse={r.sparse_n}/{r.chunks_n} "
                f"tokens={r.tokens_n}/{r.chunks_n}"
            )

    if fetch_failed:
        return 1
    if incomplete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
