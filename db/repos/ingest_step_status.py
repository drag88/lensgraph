"""Per-step / per-entity ingest status tracking.

Tracks the state machine for the PGMQ-driven ingest pipeline. Each row is
keyed by (video_id, step, entity_id). entity_id=0 for whole-video steps
(fetch, asr, chunk, frame_sample); chunk_id / frame_id otherwise (fan-out:
embed_text per chunk, embed_frames per frame).

State transitions:
  upsert            (idempotent) ensure a row exists; overwrites status only
  claim_for_update  pending -> in_progress; increments attempts; SKIP LOCKED
                    so concurrent workers see disjoint rows
  mark_completed    in_progress -> completed; stamps completed_at
  mark_failed       in_progress -> failed; stamps completed_at + error_payload

Attempts, started_at, and error_payload persist across resets so the row
preserves history for reconcilers + post-mortem inspection.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def upsert(
    conn: psycopg.Connection,
    video_id: str,
    step: str,
    *,
    entity_id: int = 0,
    status: str = "pending",
) -> None:
    """Insert a status row; on conflict overwrite `status` only."""
    conn.execute(
        """
        INSERT INTO ingest_step_status(video_id, step, entity_id, status, updated_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (video_id, step, entity_id) DO UPDATE
        SET status     = EXCLUDED.status,
            updated_at = now()
        """,
        (video_id, step, entity_id, status),
    )


def claim_for_update(
    conn: psycopg.Connection,
    step: str,
    *,
    batch_size: int = 1,
) -> list[tuple[str, int]]:
    """Atomically transition up to `batch_size` pending rows to in_progress.

    Returns the (video_id, entity_id) pairs claimed. Concurrent callers see
    disjoint rows via SELECT FOR UPDATE SKIP LOCKED on the inner pick.
    """
    rows = conn.execute(
        """
        UPDATE ingest_step_status AS s
        SET status     = 'in_progress',
            attempts   = s.attempts + 1,
            started_at = now(),
            updated_at = now()
        FROM (
            SELECT video_id, step, entity_id
            FROM ingest_step_status
            WHERE step = %s AND status = 'pending'
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        ) AS claim
        WHERE s.video_id = claim.video_id
          AND s.step      = claim.step
          AND s.entity_id = claim.entity_id
        RETURNING s.video_id, s.entity_id
        """,
        (step, batch_size),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def mark_completed(
    conn: psycopg.Connection,
    video_id: str,
    step: str,
    *,
    entity_id: int = 0,
) -> None:
    conn.execute(
        """
        UPDATE ingest_step_status
        SET status       = 'completed',
            completed_at = now(),
            updated_at   = now()
        WHERE video_id = %s AND step = %s AND entity_id = %s
        """,
        (video_id, step, entity_id),
    )


def mark_failed(
    conn: psycopg.Connection,
    video_id: str,
    step: str,
    error_payload: dict[str, Any],
    *,
    entity_id: int = 0,
) -> None:
    """Transition to failed and persist a jsonb error payload."""
    conn.execute(
        """
        UPDATE ingest_step_status
        SET status        = 'failed',
            completed_at  = now(),
            error_payload = %s,
            updated_at    = now()
        WHERE video_id = %s AND step = %s AND entity_id = %s
        """,
        (Jsonb(error_payload), video_id, step, entity_id),
    )
