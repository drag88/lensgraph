"""Dense retrieval via BGE-M3 1024-d embeddings + pgvector cosine HNSW.

Encodes the query, then runs an ORDER BY embedding <=> qvec LIMIT k.
score = 1 - cosine_distance.

Note: `register_vector(conn)` is called before parameter binding so the
numpy query vector adapts cleanly to pgvector's `vector` type. The call is
idempotent on a given connection (see db/repos/embeds.py).
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from embed.bge_m3 import encode
from retrieve.types import ChannelResult


def retrieve(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 30,
) -> list[ChannelResult]:
    """Encode the query as a BGE-M3 dense vector, return top-k chunks by cosine similarity."""
    register_vector(conn)

    qvec = encode([query]).dense[0]

    rows = conn.execute(
        """
        SELECT c.chunk_id, c.video_id, c.start_sec, c.end_sec, c.text,
               1 - (d.embedding <=> %s) AS score
        FROM dense_embeds d JOIN chunks c USING (chunk_id)
        ORDER BY d.embedding <=> %s
        LIMIT %s
        """,
        (qvec, qvec, top_k),
    ).fetchall()

    return [
        ChannelResult(
            chunk_id=row[0],
            video_id=row[1],
            start_sec=float(row[2]),
            end_sec=float(row[3]),
            text=row[4],
            score=float(row[5]),
            rank=i + 1,
        )
        for i, row in enumerate(rows)
    ]
