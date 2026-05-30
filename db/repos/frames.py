"""Frames repository — persists `FrameSample` rows to the `frames` table
plus ColQwen-derived pooled + per-patch vectors.

Idempotent on `UNIQUE (video_id, frame_sec)`. On conflict, `image_path`
and `sha256` are refreshed (handy when frames are re-sampled at the same
cadence onto a different on-disk layout) but `pooled_embedding` is NEVER
touched by ``upsert`` — that column is owned by ``update_pooled``, called
out-of-band from ``embed_frames_handler`` (slice 2).

``replace_patches`` mirrors the shape contract of
``embeds_repo.replace_token_embeds``: DELETE existing rows for the
frame_id, then INSERT one row per patch with (frame_id, patch_index,
embedding). Idempotent on re-run; never produces gaps in patch_index.

Repo functions are stateless: callers pass an open psycopg connection so
transactional boundaries stay in caller hands (frames_handler,
embed_frames_handler, tests).
"""

from __future__ import annotations

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from embed.colqwen import POOLED_DIM
from ingest.frames import FrameSample


def _ensure_registered(conn: psycopg.Connection) -> None:
    """Register pgvector adapters on this connection. Idempotent —
    register_vector is safe to call multiple times on the same connection."""
    register_vector(conn)


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


def get(conn: psycopg.Connection, frame_id: int, video_id: str) -> tuple[str, float] | None:
    """Return (image_path, frame_sec) for frame_id scoped to video_id, or
    None if missing.

    video_id is required because the embed handler dequeues a payload that
    carries both — scoping the lookup catches accidental cross-video reads
    (mirrors chunks_repo.get_text's contract).
    """
    row = conn.execute(
        "SELECT image_path, frame_sec FROM frames WHERE frame_id = %s AND video_id = %s",
        (frame_id, video_id),
    ).fetchone()
    if row is None:
        return None
    return (row[0], float(row[1]))


def update_pooled(
    conn: psycopg.Connection,
    frame_id: int,
    embedding: np.ndarray,
) -> None:
    """Set frames.pooled_embedding for an existing frame_id.

    embedding shape: (POOLED_DIM,) i.e. (128,) for ColQwen2.5. A missing
    frame_id is a no-op at the SQL level (UPDATE matches zero rows) — the
    handler is expected to have verified the frame exists via ``get``
    before calling.
    """
    if embedding.ndim != 1 or embedding.shape[0] != POOLED_DIM:
        raise ValueError(f"update_pooled expects ({POOLED_DIM},); got shape {embedding.shape}")
    _ensure_registered(conn)
    conn.execute(
        "UPDATE frames SET pooled_embedding = %s WHERE frame_id = %s",
        (embedding, frame_id),
    )


def clear_embeddings(conn: psycopg.Connection, frame_id: int) -> None:
    """Drop any ColQwen-derived vectors for a frame: NULL ``pooled_embedding``
    and DELETE its ``frame_patches`` rows.

    Used by ``embed_frames_handler`` when a re-embed yields NaN/inf patches
    or pooled vector: the handler skips writing new rows, but a previous good
    embedding may still be on the row. Leaving it would make the frame keep
    serving a stale pooled vector from the HNSW prefilter while the report
    claims the frame carries no embedding. Clearing makes the skip honest —
    the frame is genuinely absent from retrieval (partial HNSW index from
    migration 0003 omits NULL pooled rows). A missing frame_id is a no-op.
    """
    conn.execute("UPDATE frames SET pooled_embedding = NULL WHERE frame_id = %s", (frame_id,))
    conn.execute("DELETE FROM frame_patches WHERE frame_id = %s", (frame_id,))


def replace_patches(
    conn: psycopg.Connection,
    frame_id: int,
    patches: np.ndarray,
) -> None:
    """DELETE existing frame_patches rows for frame_id, INSERT fresh per-patch
    vectors.

    patches shape: (P, POOLED_DIM). P must be >= 1 — a zero-patch frame
    has no visual signal to store, and the handler must filter that case
    before calling (mirrors embed_text's contract: empty multi vectors
    leave the chunk with zero token rows, but the handler still calls;
    here we're stricter because a zero-patch frame is structurally
    impossible from ColQwen).

    Replace-style (not upsert): patch count can vary across re-encodes
    (different image, different aspect ratio), and (frame_id, patch_index)
    gaps would be confusing. Full replace keeps the table consistent on
    every re-run.
    """
    if patches.ndim != 2 or patches.shape[1] != POOLED_DIM:
        raise ValueError(f"replace_patches expects (P, {POOLED_DIM}); got shape {patches.shape}")
    if patches.shape[0] < 1:
        raise ValueError(
            "replace_patches requires at least 1 patch row; "
            "filter zero-patch frames in the handler before calling"
        )
    _ensure_registered(conn)
    conn.execute("DELETE FROM frame_patches WHERE frame_id = %s", (frame_id,))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO frame_patches(frame_id, patch_index, embedding) VALUES (%s, %s, %s)",
            [(frame_id, i, patches[i]) for i in range(patches.shape[0])],
        )
