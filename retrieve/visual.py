"""Visual retrieval — true 5th channel (pooled prefilter + per-patch MaxSim).

Two public entry points:

* ``retrieve_frames(conn, query, *, top_k)`` — stage-1 only. Returns the top-k
  frames by pooled-vector cosine. Kept for the slice-0 tests and for the
  trace viewer (which surfaces "which frame matched"); production callers
  use ``retrieve()`` instead.

* ``retrieve(conn, query, *, top_k, prefilter_k)`` — channel-level entry
  per design §4. Three stages:

    (a) coarse: pgvector HNSW over ``frames.pooled_embedding`` using the
        ColQwen text-encoder mean-pooled query embedding (same encoder
        the pooled column was built with). Returns top ``prefilter_k``
        frames (default 200).
    (b) map frames → containing chunks via the chunks table:
        ``cf.frame_sec ∈ [chunks.start_sec, chunks.end_sec)``. One SQL JOIN.
    (c) MaxSim refine over the candidate chunks' frame_patches against
        the per-query-token patches from ``encode_text_query_patches``.
        Returns top ``top_k`` chunks.

Stage (c) implementation choice — *one MaxSim per chunk over the
concatenated patches of all the chunk's prefiltered frames*. The alternative
(sum of per-frame MaxSim within a chunk) overweights chunks with more
frames in the prefilter, which is a sampling artifact, not signal. Stacking
all the chunk's prefiltered patches into a single (sum_P, 128) matrix and
running MaxSim against the query patches once is both simpler and
size-invariant on the chunk axis. The query side is fixed at ``T_q`` tokens
so the comparison is symmetric for the eval write-up.

Edge cases:
  * Empty pooled prefilter → ``[]`` (no model load triggered downstream).
  * Frames with NULL pooled_embedding are excluded by the partial HNSW
    index from migration 0003; the WHERE clause defends in depth.
  * A chunk whose prefiltered frames have NO rows in ``frame_patches``
    contributes nothing to stage (c) and falls out — this is the
    "patches not yet computed" case slice-1 deliberately left behind
    (frame_patches population lands with the other agent's worker).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from embed.colqwen import encode_text_query, encode_text_query_patches
from retrieve.types import ChannelResult


@dataclass(frozen=True)
class FrameResult:
    """One frame returned by stage-1 visual retrieval (`retrieve_frames`)."""

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
    """Stage-1 only: pooled-prefilter cosine HNSW over frames.pooled_embedding."""
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


def retrieve(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 30,
    prefilter_k: int = 200,
) -> list[ChannelResult]:
    """Channel-level visual retrieve: pooled HNSW prefilter, then MaxSim
    refine over frame_patches.

    Returns top_k chunks sorted by MaxSim score (descending), 1-indexed
    `rank`. Empty pooled prefilter → []. Chunks whose prefiltered frames
    have no patches in `frame_patches` are silently dropped (documented
    in the module docstring).
    """
    register_vector(conn)

    # --- Stage (a): pooled-vector HNSW prefilter ---
    pooled_q = encode_text_query(query)
    pre_rows = conn.execute(
        """
        SELECT frame_id, video_id, frame_sec
        FROM frames
        WHERE pooled_embedding IS NOT NULL
        ORDER BY pooled_embedding <=> %s
        LIMIT %s
        """,
        (pooled_q, prefilter_k),
    ).fetchall()
    if not pre_rows:
        return []

    candidate_frame_ids = [r[0] for r in pre_rows]

    # --- Stage (b): map frames → containing chunks (one JOIN) ---
    # frame_sec ∈ [chunks.start_sec, chunks.end_sec). Half-open interval so
    # a frame sampled exactly at the boundary belongs to the later chunk —
    # matches the convention `chunking.fixed_window` uses when populating
    # chunks.frame_secs.
    map_rows = conn.execute(
        """
        WITH cand_frames AS (
            SELECT frame_id, video_id, frame_sec
            FROM frames
            WHERE frame_id = ANY(%s)
        )
        SELECT c.chunk_id, c.video_id, c.start_sec, c.end_sec, c.text, cf.frame_id
        FROM cand_frames cf
        JOIN chunks c
          ON c.video_id = cf.video_id
         AND cf.frame_sec >= c.start_sec
         AND cf.frame_sec <  c.end_sec
        """,
        (candidate_frame_ids,),
    ).fetchall()
    if not map_rows:
        return []

    # chunk_id → list[frame_id] (one chunk can match multiple candidate frames)
    chunk_frames: dict[int, list[int]] = defaultdict(list)
    chunk_meta: dict[int, tuple[str, float, float, str]] = {}
    for chunk_id, video_id, start_sec, end_sec, text, frame_id in map_rows:
        chunk_frames[chunk_id].append(frame_id)
        chunk_meta.setdefault(
            chunk_id, (video_id, float(start_sec), float(end_sec), text)
        )

    # --- Stage (c): batched fetch of frame_patches for ALL candidate frames ---
    all_candidate_frame_ids = sorted({fid for fids in chunk_frames.values() for fid in fids})
    patch_rows = conn.execute(
        """
        SELECT frame_id, patch_index, embedding
        FROM frame_patches
        WHERE frame_id = ANY(%s)
        ORDER BY frame_id, patch_index
        """,
        (all_candidate_frame_ids,),
    ).fetchall()
    if not patch_rows:
        return []

    frame_patches: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame_id, _patch_idx, embedding in patch_rows:
        frame_patches[frame_id].append(embedding)

    # --- MaxSim per chunk over the union of its prefiltered frames' patches ---
    qpatches = encode_text_query_patches(query).astype(np.float32)  # (T_q, 128)
    if qpatches.size == 0:
        return []

    scored: list[tuple[int, float]] = []
    for chunk_id, frame_ids in chunk_frames.items():
        patches: list[np.ndarray] = []
        for fid in frame_ids:
            patches.extend(frame_patches.get(fid, ()))
        if not patches:
            continue
        cpatches = np.stack(patches).astype(np.float32)  # (sum_P, 128)
        sims = qpatches @ cpatches.T  # (T_q, sum_P)
        maxsim = float(sims.max(axis=1).sum())
        scored.append((chunk_id, maxsim))

    if not scored:
        return []

    scored.sort(key=lambda x: (-x[1], x[0]))
    top = scored[:top_k]

    return [
        ChannelResult(
            chunk_id=cid,
            video_id=chunk_meta[cid][0],
            start_sec=chunk_meta[cid][1],
            end_sec=chunk_meta[cid][2],
            text=chunk_meta[cid][3],
            score=score,
            rank=i + 1,
        )
        for i, (cid, score) in enumerate(top)
    ]
