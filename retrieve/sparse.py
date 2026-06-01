"""Sparse retrieval via BGE-M3 lexical_weights + pgvector sparsevec.

Inner-product distance (<#>) is the operator. score = -(distance) so
higher = more relevant.

pgvector-python's SparseVector accepts 0-based keys and emits 1-based
wire format internally — see db/repos/embeds.py module docstring for the
contract. We pass the BGE-M3 lexical_weights dict (already 0-based)
through unmodified.
"""

from __future__ import annotations

import psycopg
from pgvector import SparseVector
from pgvector.psycopg import register_vector

from embed.bge_m3 import VOCAB_SIZE, encode
from retrieve.types import ChannelResult


def retrieve(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 30,
    chunking_strategy: str = "fixed_window",
) -> list[ChannelResult]:
    """Encode the query as a BGE-M3 sparse vector, return top-k chunks by inner product.

    ``chunking_strategy`` filters to one strategy's chunks (see ``dense.retrieve``).
    Defaults to ``fixed_window`` so production is unaffected by ablation rows.
    """
    register_vector(conn)

    raw_sparse = encode([query]).sparse[0]
    sv = SparseVector(raw_sparse, VOCAB_SIZE)

    rows = conn.execute(
        """
        SELECT c.chunk_id, c.video_id, c.start_sec, c.end_sec, c.text,
               -(s.embedding <#> %s) AS score
        FROM sparse_embeds s JOIN chunks c USING (chunk_id)
        WHERE c.chunking_strategy = %s
        ORDER BY s.embedding <#> %s
        LIMIT %s
        """,
        (sv, chunking_strategy, sv, top_k),
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
