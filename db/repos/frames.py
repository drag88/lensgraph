"""Frames repository — persists `FrameSample` rows to the `frames` table.

Idempotent on `UNIQUE (video_id, frame_sec)`. On conflict, `image_path`
and `sha256` are refreshed (handy when frames are re-sampled at the same
cadence onto a different on-disk layout) but `pooled_embedding` is NEVER
touched — that column is owned by the `embed_frames` step (slice 2 of
the week-5 mission stack).

Repo functions are stateless: callers pass an open psycopg connection so
transactional boundaries stay in caller hands (frames_handler, tests).
"""

from __future__ import annotations

import psycopg

from ingest.frames import FrameSample


def upsert(conn: psycopg.Connection, samples: list[FrameSample]) -> list[int]:
    """Bulk upsert. Returns `frame_id` list aligned to input order.

    Empty input short-circuits with no DB call. `pooled_embedding` is
    deliberately omitted from the column list — the column defaults to
    NULL on insert, and the `DO UPDATE` clause leaves any previously
    populated embedding alone.
    """
    if not samples:
        return []
    ids: list[int] = []
    for s in samples:
        row = conn.execute(
            """
            INSERT INTO frames (video_id, frame_sec, image_path, sha256)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (video_id, frame_sec) DO UPDATE SET
                image_path = EXCLUDED.image_path,
                sha256     = EXCLUDED.sha256
            RETURNING frame_id
            """,
            (s.video_id, s.frame_sec, s.image_path, s.sha256),
        ).fetchone()
        ids.append(int(row[0]))
    return ids
