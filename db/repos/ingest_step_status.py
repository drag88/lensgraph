"""Per-step / per-entity ingest status tracking.

State machine for the PGMQ-driven ingest pipeline. Rows are keyed by
(video_id, step, entity_id). entity_id=0 for whole-video steps (fetch, asr,
chunk, frame_sample); chunk_id / frame_id for fan-out (embed_text per chunk,
embed_frames per frame).

Job-identity discipline: claim_for_update, mark_completed, and mark_failed
all operate on the EXACT (video_id, step, entity_id) tuple from the PGMQ
payload — never on "some pending row of step X". A worker dequeued for
video B cannot accidentally transition video A's state. This is the
contract design §3 requires: a dequeued job must lock the row identified
by its own payload, period.

Transitions (one row only):
  claim_for_update   FOR UPDATE locks the row; if completed/skipped, returns
                     that state untouched (worker acks + exits); else
                     transitions to in_progress, increments attempts, stamps
                     started_at. MUST be called inside a transaction.
                     Raises IngestStepNotFoundError if no row exists.
  mark_completed     in_progress -> completed; returns True iff exactly one
                     row transitioned. False means the row was not in
                     in_progress (contract violation — caller decides).
  mark_failed        in_progress -> failed; persists jsonb error_payload;
                     same True/False semantics.

Why not 'pending only' on claim? PGMQ may redeliver after a visibility
timeout, and the prior worker may have crashed mid-step leaving status
in_progress. Re-claim is correct: bump attempts, reset started_at, proceed.
The work itself must be idempotent — that's the pipeline-level contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


class IngestStepNotFoundError(KeyError):
    """No ingest_step_status row exists for the given (video_id, step, entity_id)."""


@dataclass(frozen=True)
class ClaimResult:
    """Outcome of claim_for_update.

    status:
      - 'in_progress'  — row just transitioned; worker should proceed with work
      - 'completed'    — already done; worker should ack the PGMQ message and exit
      - 'skipped'      — explicitly skipped (e.g. asr step on a video with caps);
                         worker should ack and exit
    attempts: post-transition value (incremented if we claimed; unchanged otherwise)
    """

    status: str
    attempts: int


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
    video_id: str,
    step: str,
    *,
    entity_id: int = 0,
) -> ClaimResult:
    """Lock the exact (video_id, step, entity_id) row and transition if needed.

    See module docstring for state-transition semantics. MUST be called inside
    a transaction so the FOR UPDATE lock is held until commit/rollback. Raises
    IngestStepNotFoundError if no row exists for the key.
    """
    row = conn.execute(
        """
        SELECT status, attempts
        FROM ingest_step_status
        WHERE video_id = %s AND step = %s AND entity_id = %s
        FOR UPDATE
        """,
        (video_id, step, entity_id),
    ).fetchone()
    if row is None:
        raise IngestStepNotFoundError(
            f"no ingest_step_status row for video_id={video_id!r} "
            f"step={step!r} entity_id={entity_id}"
        )
    status, attempts = row
    if status in {"completed", "skipped"}:
        return ClaimResult(status=status, attempts=attempts)
    conn.execute(
        """
        UPDATE ingest_step_status
        SET status     = 'in_progress',
            attempts   = attempts + 1,
            started_at = now(),
            updated_at = now()
        WHERE video_id = %s AND step = %s AND entity_id = %s
        """,
        (video_id, step, entity_id),
    )
    return ClaimResult(status="in_progress", attempts=attempts + 1)


def mark_completed(
    conn: psycopg.Connection,
    video_id: str,
    step: str,
    *,
    entity_id: int = 0,
) -> bool:
    """Transition in_progress -> completed. Returns True iff a row transitioned."""
    row = conn.execute(
        """
        UPDATE ingest_step_status
        SET status       = 'completed',
            completed_at = now(),
            updated_at   = now()
        WHERE video_id = %s AND step = %s AND entity_id = %s
          AND status = 'in_progress'
        RETURNING 1
        """,
        (video_id, step, entity_id),
    ).fetchone()
    return row is not None


def mark_failed(
    conn: psycopg.Connection,
    video_id: str,
    step: str,
    error_payload: dict[str, Any],
    *,
    entity_id: int = 0,
) -> bool:
    """Transition in_progress -> failed; persist jsonb error payload.

    Returns True iff a row transitioned. False indicates the row was not in
    in_progress when called (contract violation; caller decides whether to
    raise, log, or ignore).
    """
    row = conn.execute(
        """
        UPDATE ingest_step_status
        SET status        = 'failed',
            completed_at  = now(),
            error_payload = %s,
            updated_at    = now()
        WHERE video_id = %s AND step = %s AND entity_id = %s
          AND status = 'in_progress'
        RETURNING 1
        """,
        (Jsonb(error_payload), video_id, step, entity_id),
    ).fetchone()
    return row is not None
