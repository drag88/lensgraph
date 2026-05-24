"""Ingest CLI — drive the fetch-stage reconciler for a single video.

`python -m ingest.cli VIDEO_ID [--corpus ai_engineering_v0]` looks the video
up in the corpus's talks.yaml, builds a LocalFsFetcher backed by talks.yaml
transcript paths, ensures PGMQ queues exist, and runs `fetch_reconcile`
inside one transaction. A summary of the reconciler's decisions is printed
on success; any failure rolls back, leaving no partial talks row.

This is the offline ingestion entry point. yt-dlp-backed ingestion lands
when YtDlpFetcher is wired up; for now LocalFsFetcher reads the pre-staged
VTTs referenced by talks.yaml.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psycopg
import yaml

from db.conn import dsn as resolve_dsn
from db.repos.talks import Talk
from ingest.fetch import LocalFsFetcher, paths_from_talks_yaml
from ingest.pipeline import fetch_reconcile
from queues import pgmq_client

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class _Args:
    video_id: str
    corpus: str


def _parse_args(argv: list[str]) -> _Args:
    parser = argparse.ArgumentParser(
        prog="ingest.cli",
        description="Ingest a single video by id using local transcripts from talks.yaml.",
    )
    parser.add_argument("video_id", help="video_id from eval/corpora/<corpus>/talks.yaml")
    parser.add_argument(
        "--corpus",
        default="ai_engineering_v0",
        help="corpus directory under eval/corpora/ (default: ai_engineering_v0)",
    )
    ns = parser.parse_args(argv)
    return _Args(video_id=ns.video_id, corpus=ns.corpus)


def _load_talk_entry(corpus_dir: Path, video_id: str) -> dict:
    talks_yaml = corpus_dir / "talks.yaml"
    with talks_yaml.open(encoding="utf-8") as f:
        talks = yaml.safe_load(f)
    for entry in talks:
        if entry["video_id"] == video_id:
            return entry
    raise KeyError(
        f"video_id {video_id!r} not found in {talks_yaml} "
        f"(corpus={corpus_dir.name})"
    )


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or [])

    corpus_dir = REPO_ROOT / "eval" / "corpora" / args.corpus
    entry = _load_talk_entry(corpus_dir, args.video_id)
    talk = _build_talk(entry)

    fetcher = LocalFsFetcher(
        paths=paths_from_talks_yaml(corpus_dir, repo_root=REPO_ROOT),
        captions_source=entry["captions_source"],
    )

    with psycopg.connect(resolve_dsn(), autocommit=True) as conn:
        pgmq_client.ensure_queues(conn)
        with conn.transaction():
            result = fetch_reconcile(conn, talk, fetcher)

    print(f"ingested {talk.video_id}:")
    print(f"  asr_skipped:    {result.asr_skipped}")
    print(f"  frames_skipped: {result.frames_skipped}")
    print(f"  chunk_msg_id:   {result.chunk_msg_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
