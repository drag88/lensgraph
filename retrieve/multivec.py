"""Multi-vector retrieval via BGE-M3 per-token embeddings + MaxSim.

Prefilter via dense + sparse top-K_pref each (default K_pref = 100). Union
the candidate chunk_ids, fetch their token rows from chunk_token_embeds in
one batched SELECT, compute MaxSim = sum_t_q max_t_c (q_t @ c_t) per
chunk, take top-k by MaxSim.

Per design §4: "MaxSim only over dense+sparse top-100 candidate union —
never the full corpus." HNSW over individual token vectors is not useful,
so the prefilter is the only way to keep MaxSim tractable.

Encoding the query as a per-token matrix is a single bge_m3.encode call;
we accept the 3x re-encode cost (dense / sparse / multi each call encode
internally via the dense and sparse retrieve modules) because the encode
is dwarfed by HNSW + token-fetch latency. A future fast path could share
one BgeM3Output across all three channels by accepting an encoded query
as a kwarg.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from embed.bge_m3 import encode
from retrieve import dense, sparse
from retrieve.types import ChannelResult


def retrieve(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 30,
    prefilter_k: int = 100,
    chunking_strategy: str = "fixed_window",
) -> list[ChannelResult]:
    """Two-stage retrieval: dense+sparse prefilter, then MaxSim over candidate token vectors.

    ``chunking_strategy`` is threaded to the dense+sparse prefilter, so the
    candidate set — and therefore the MaxSim output — is confined to one
    strategy's chunks. Defaults to ``fixed_window``.
    """
    qmulti = encode([query]).multi[0]  # (T_q, 1024)

    # 1. Prefilter via the other two text channels (strategy-confined).
    d_results = dense.retrieve(conn, query, top_k=prefilter_k, chunking_strategy=chunking_strategy)
    s_results = sparse.retrieve(conn, query, top_k=prefilter_k, chunking_strategy=chunking_strategy)
    candidate_ids = list({r.chunk_id for r in d_results} | {r.chunk_id for r in s_results})
    if not candidate_ids:
        return []

    # 2. Fetch token embeds for candidates in ONE batched SELECT.
    register_vector(conn)
    rows = conn.execute(
        """
        SELECT chunk_id, position, embedding
        FROM chunk_token_embeds
        WHERE chunk_id = ANY(%s)
        ORDER BY chunk_id, position
        """,
        (candidate_ids,),
    ).fetchall()

    chunk_tokens: dict[int, list[np.ndarray]] = defaultdict(list)
    for r in rows:
        chunk_tokens[r[0]].append(r[2])

    # 3. MaxSim per chunk. float32 cast keeps the matmul on a single dtype path
    # — pgvector returns float arrays but dtype is not guaranteed across versions,
    # and a silent float64 upcast would double memory + halve throughput.
    qmulti32 = qmulti.astype(np.float32)
    scored: list[tuple[int, float]] = []
    for chunk_id, tokens in chunk_tokens.items():
        if not tokens:
            continue
        ctok = np.stack(tokens).astype(np.float32)
        sims = qmulti32 @ ctok.T  # (T_q, T_c)
        maxsim = float(sims.max(axis=1).sum())
        scored.append((chunk_id, maxsim))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    top_ids = [c for c, _ in top]
    score_map = dict(top)

    # 4. Hydrate chunk metadata for the top-k.
    chunk_rows = conn.execute(
        "SELECT chunk_id, video_id, start_sec, end_sec, text FROM chunks WHERE chunk_id = ANY(%s)",
        (top_ids,),
    ).fetchall()
    meta = {r[0]: r for r in chunk_rows}

    return [
        ChannelResult(
            chunk_id=cid,
            video_id=meta[cid][1],
            start_sec=float(meta[cid][2]),
            end_sec=float(meta[cid][3]),
            text=meta[cid][4],
            score=score_map[cid],
            rank=i + 1,
        )
        for i, cid in enumerate(top_ids)
    ]
