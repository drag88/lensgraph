"""Per-video ingest readiness report.

A video is "complete" when:
  - talks row exists
  - chunks_n > 0
  - dense_n  == chunks_n
  - sparse_n == chunks_n
  - tokens_n == chunks_n   (count of DISTINCT chunk_ids in chunk_token_embeds)

Partial coverage (dense_n < chunks_n etc.) is the failure mode this
module exists to surface — the prior boolean-only formulation reported
"ready" when a channel was 1-row-non-empty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import psycopg


@dataclass(frozen=True)
class VideoReadiness:
    video_id: str
    talks_present: bool
    chunks_n: int
    dense_n: int
    sparse_n: int
    tokens_n: int

    @property
    def complete(self) -> bool:
        return (
            self.talks_present
            and self.chunks_n > 0
            and self.dense_n == self.chunks_n
            and self.sparse_n == self.chunks_n
            and self.tokens_n == self.chunks_n
        )


def for_video(conn: psycopg.Connection, video_id: str) -> VideoReadiness:
    """Single SQL query joining all four counts. Returns 0s if no rows."""
    row = conn.execute(
        """
        SELECT
          EXISTS(SELECT 1 FROM talks WHERE video_id = %(v)s)::bool AS talks_present,
          (SELECT count(*) FROM chunks WHERE video_id = %(v)s) AS chunks_n,
          (SELECT count(*) FROM dense_embeds d
             JOIN chunks c ON c.chunk_id = d.chunk_id
             WHERE c.video_id = %(v)s) AS dense_n,
          (SELECT count(*) FROM sparse_embeds s
             JOIN chunks c ON c.chunk_id = s.chunk_id
             WHERE c.video_id = %(v)s) AS sparse_n,
          (SELECT count(DISTINCT cte.chunk_id) FROM chunk_token_embeds cte
             JOIN chunks c ON c.chunk_id = cte.chunk_id
             WHERE c.video_id = %(v)s) AS tokens_n
        """,
        {"v": video_id},
    ).fetchone()
    return VideoReadiness(
        video_id=video_id,
        talks_present=bool(row[0]),
        chunks_n=int(row[1]),
        dense_n=int(row[2]),
        sparse_n=int(row[3]),
        tokens_n=int(row[4]),
    )


def for_corpus(conn: psycopg.Connection, corpus_dir: Path) -> list[VideoReadiness]:
    """Read dev_gold + synthesis to collect video_ids, then build a
    VideoReadiness per video. Sort by video_id for deterministic output."""
    vids = collect_video_ids(corpus_dir)
    return [for_video(conn, v) for v in sorted(vids)]


def collect_video_ids(corpus_dir: Path) -> set[str]:
    """Union of video_ids referenced in dev_gold.jsonl and synthesis.jsonl.
    Reads top-level video_id AND gold_spans[*].video_id."""
    vids: set[str] = set()
    for filename in ("dev_gold.jsonl", "synthesis.jsonl"):
        path = corpus_dir / filename
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "video_id" in ex and ex["video_id"]:
                vids.add(ex["video_id"])
            for span in ex.get("gold_spans", []) or []:
                if "video_id" in span and span["video_id"]:
                    vids.add(span["video_id"])
    return vids
