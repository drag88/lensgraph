"""Visual retrieval — stage 1: pooled prefilter.

Encodes the text query with ColQwen2.5's text encoder, then runs cosine
HNSW search against frames.pooled_embedding. Returns the top-k frames
ordered by cosine similarity. Stage 2 (per-patch MaxSim over the
prefiltered frames) lands in step 24-25 once embed/colqwen.encode_image_patches
exists.

frames with NULL pooled_embedding are excluded by the WHERE clause —
the frames_pooled_hnsw partial index in 0003_frames.sql already enforces
this on the storage side.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from embed.colqwen import encode_text_query


@dataclass(frozen=True)
class FrameResult:
    """One frame returned by visual retrieval."""

    frame_id: int
    video_id: str
    frame_sec: float
    image_path: str
    score: float
    rank: int


def retrieve_frames(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 30,
) -> list[FrameResult]:
    """Pooled-prefilter retrieve: cosine HNSW over frames.pooled_embedding."""
    qvec = encode_text_query(query)
    register_vector(conn)
    rows = conn.execute(
        """
        SELECT frame_id, video_id, frame_sec, image_path,
               1 - (pooled_embedding <=> %s) AS score
        FROM frames
        WHERE pooled_embedding IS NOT NULL
        ORDER BY pooled_embedding <=> %s
        LIMIT %s
        """,
        (qvec, qvec, top_k),
    ).fetchall()
    return [
        FrameResult(
            frame_id=r[0],
            video_id=r[1],
            frame_sec=r[2],
            image_path=r[3],
            score=float(r[4]),
            rank=i + 1,
        )
        for i, r in enumerate(rows)
    ]
