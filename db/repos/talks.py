"""Talk repository — round-trips talks.yaml rows.

Talk mirrors the shape of eval/schemas/talk.schema.json. Repo functions are
stateless: callers pass an open psycopg connection so transactional
boundaries stay in caller hands (ingest pipeline, reconcilers, tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg

_COLUMNS = (
    "video_id, source_video_id, source_start_sec, source_end_sec, "
    "title, speaker, url, duration_sec, format_tags, license, "
    "captions_source, transcript_path, transcript_sha256, accessed_at, "
    "notes, ingested_at"
)


@dataclass(frozen=True)
class Talk:
    video_id: str
    title: str
    speaker: str
    url: str
    duration_sec: int
    format_tags: list[str]
    license: str
    captions_source: str
    transcript_path: str
    transcript_sha256: str
    accessed_at: datetime
    source_video_id: str | None = None
    source_start_sec: float | None = None
    source_end_sec: float | None = None
    notes: str | None = None
    ingested_at: datetime | None = None


def upsert(conn: psycopg.Connection, talk: Talk) -> None:
    """Insert or replace a talk by video_id. All non-key columns are overwritten."""
    conn.execute(
        f"""
        INSERT INTO talks ({_COLUMNS})
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO UPDATE SET
          source_video_id   = EXCLUDED.source_video_id,
          source_start_sec  = EXCLUDED.source_start_sec,
          source_end_sec    = EXCLUDED.source_end_sec,
          title             = EXCLUDED.title,
          speaker           = EXCLUDED.speaker,
          url               = EXCLUDED.url,
          duration_sec      = EXCLUDED.duration_sec,
          format_tags       = EXCLUDED.format_tags,
          license           = EXCLUDED.license,
          captions_source   = EXCLUDED.captions_source,
          transcript_path   = EXCLUDED.transcript_path,
          transcript_sha256 = EXCLUDED.transcript_sha256,
          accessed_at       = EXCLUDED.accessed_at,
          notes             = EXCLUDED.notes,
          ingested_at       = EXCLUDED.ingested_at
        """,
        (
            talk.video_id,
            talk.source_video_id,
            talk.source_start_sec,
            talk.source_end_sec,
            talk.title,
            talk.speaker,
            talk.url,
            talk.duration_sec,
            talk.format_tags,
            talk.license,
            talk.captions_source,
            talk.transcript_path,
            talk.transcript_sha256,
            talk.accessed_at,
            talk.notes,
            talk.ingested_at,
        ),
    )


def get(conn: psycopg.Connection, video_id: str) -> Talk | None:
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM talks WHERE video_id = %s",
        (video_id,),
    ).fetchone()
    if row is None:
        return None
    return Talk(
        video_id=row[0],
        source_video_id=row[1],
        source_start_sec=row[2],
        source_end_sec=row[3],
        title=row[4],
        speaker=row[5],
        url=row[6],
        duration_sec=row[7],
        format_tags=list(row[8]),
        license=row[9],
        captions_source=row[10],
        transcript_path=row[11],
        transcript_sha256=row[12],
        accessed_at=row[13],
        notes=row[14],
        ingested_at=row[15],
    )
