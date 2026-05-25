"""BM25-ish retrieval via Postgres FTS.

Uses ts_rank(tsv, websearch_to_tsquery('english', query)) ordered DESC.
The chunks.tsv column is GENERATED ALWAYS — populated automatically by the
chunker writes, no separate index step needed.

The chunks_tsv_idx GIN index keeps this fast even at v1 corpus scale.

`websearch_to_tsquery` is chosen over `to_tsquery` because it is far more
forgiving of natural-language queries: free-text phrases, stop words,
quoted substrings, and operators like OR / - are all accepted without
caller-side escaping. `to_tsquery` requires syntactically valid input and
raises on any whitespace, which makes it unfit for direct query strings.
"""

from __future__ import annotations

import psycopg

from retrieve.types import ChannelResult


def retrieve(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 30,
) -> list[ChannelResult]:
    """Run a Postgres FTS query against chunks.tsv, return top-k by ts_rank."""
    rows = conn.execute(
        """
        SELECT chunk_id, video_id, start_sec, end_sec, text,
               ts_rank(tsv, websearch_to_tsquery('english', %s)) AS score
        FROM chunks
        WHERE tsv @@ websearch_to_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, query, top_k),
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
