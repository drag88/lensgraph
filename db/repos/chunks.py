"""Chunks repository — persists Chunk dataclasses to the chunks table.

Repo functions are stateless: callers pass an open psycopg connection so
transactional boundaries stay in caller hands (chunk handler, reconcilers,
tests).

Idempotency: the natural key is (video_id, chunking_strategy, start_sec,
end_sec). ON CONFLICT DO UPDATE on the non-key columns lets RETURNING bring
the existing chunk_id back — so re-running upsert against the same input
returns the same ids without inserting duplicates.
"""

from __future__ import annotations

import psycopg

from chunking.types import Chunk


def upsert(conn: psycopg.Connection, chunks: list[Chunk]) -> list[int]:
    """Bulk upsert. Returns chunk_id list aligned to input order.

    UNIQUE (video_id, chunking_strategy, start_sec, end_sec) is the natural
    key — ON CONFLICT triggers DO UPDATE on the non-key columns so RETURNING
    yields the existing chunk_id back. Idempotent on re-run.
    """
    if not chunks:
        return []
    ids: list[int] = []
    for ch in chunks:
        row = conn.execute(
            """
            INSERT INTO chunks (
                video_id, chunking_strategy, start_sec, end_sec,
                text, token_count, frame_secs
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id, chunking_strategy, start_sec, end_sec)
            DO UPDATE SET
                text        = EXCLUDED.text,
                token_count = EXCLUDED.token_count,
                frame_secs  = EXCLUDED.frame_secs
            RETURNING chunk_id
            """,
            (
                ch.video_id,
                ch.chunking_strategy,
                ch.start_sec,
                ch.end_sec,
                ch.text,
                ch.token_count,
                ch.frame_secs,
            ),
        ).fetchone()
        ids.append(int(row[0]))
    return ids


def get_text(
    conn: psycopg.Connection,
    chunk_id: int,
    video_id: str,
) -> str | None:
    """Return the text for chunk_id (scoped to video_id), or None if missing.

    video_id is required because the embed handler dequeues a payload that
    carries both — scoping the lookup catches accidental cross-video reads.
    """
    row = conn.execute(
        "SELECT text FROM chunks WHERE chunk_id = %s AND video_id = %s",
        (chunk_id, video_id),
    ).fetchone()
    if row is None:
        return None
    return row[0]
