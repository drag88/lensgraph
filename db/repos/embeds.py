"""Embedding repositories — dense, sparse, multi-vector channels.

Three concerns, one file because they all wrap a single chunk_id and share
the pgvector adapter setup.

*** Sparsevec indexing contract — empirically verified ***

pgvector's wire format for sparsevec is 1-based (the text representation
shows indices 1..dim, and `embedding::text` returns the same), BUT the
pgvector-python `SparseVector` class handles that conversion internally:
it takes 0-based dict keys and emits 1-based wire format. So callers (us)
pass `dict[int, float]` keyed by 0-based BGE-M3 token ids directly — no
shift in this wrapper. Passing key=0 yields wire `{1:v}/dim`; passing
key=VOCAB_SIZE-1 (=250001) yields wire `{250002:v}/250002`.

A previous version of this wrapper added a +1 shift here, which
double-shifted and crashed with "sparsevec index out of bounds" for
boundary tokens. The fix is to trust pgvector-python and pass through.

The sparse channel is consumed only by inner-product retrieval, never
decoded back to token ids — so we don't worry about the 0/1 distinction
beyond getting the storage right.
"""

from __future__ import annotations

import numpy as np
import psycopg
from pgvector import SparseVector
from pgvector.psycopg import register_vector

from embed.bge_m3 import VOCAB_SIZE


def _ensure_registered(conn: psycopg.Connection) -> None:
    """Register pgvector adapters on this connection. Idempotent —
    register_vector is safe to call multiple times on the same connection."""
    register_vector(conn)


def upsert_dense(
    conn: psycopg.Connection,
    chunk_id: int,
    vector: np.ndarray,
) -> None:
    """INSERT or replace the dense_embeds row for chunk_id.

    vector shape: (DENSE_DIM,) i.e. (1024,) for BGE-M3.
    """
    _ensure_registered(conn)
    conn.execute(
        """
        INSERT INTO dense_embeds(chunk_id, embedding) VALUES (%s, %s)
        ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """,
        (chunk_id, vector),
    )


def upsert_sparse(
    conn: psycopg.Connection,
    chunk_id: int,
    sparse: dict[int, float],
) -> None:
    """INSERT or replace the sparse_embeds row.

    `sparse` is a dict keyed by 0-based BGE-M3 token ids. pgvector-python's
    SparseVector accepts 0-based keys and produces 1-based wire format
    automatically — see module docstring. Empty `sparse` writes the
    all-zero vector — the chunk row still exists, just with no lexical mass.
    """
    _ensure_registered(conn)
    sv = SparseVector(sparse, VOCAB_SIZE)
    conn.execute(
        """
        INSERT INTO sparse_embeds(chunk_id, embedding) VALUES (%s, %s)
        ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """,
        (chunk_id, sv),
    )


def replace_token_embeds(
    conn: psycopg.Connection,
    chunk_id: int,
    vectors: np.ndarray,
) -> None:
    """DELETE existing rows for chunk_id, INSERT fresh per-token vectors.

    vectors shape: (T, DENSE_DIM). Per-token storage; consumed by MaxSim in
    retrieve/multivec. Replace-style (not upsert) because token count can
    change across runs and (chunk_id, position) gaps are confusing — an
    empty `vectors` leaves the chunk with zero token rows.
    """
    _ensure_registered(conn)
    conn.execute("DELETE FROM chunk_token_embeds WHERE chunk_id = %s", (chunk_id,))
    if vectors.size == 0:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunk_token_embeds(chunk_id, position, embedding) VALUES (%s, %s, %s)",
            [(chunk_id, i, vectors[i]) for i in range(vectors.shape[0])],
        )
