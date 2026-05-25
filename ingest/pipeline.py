"""Fetch-stage reconciler.

Orchestrates the fetch path:
  1. Run the Fetcher to put the transcript on disk + get provenance.
  2. Upsert the talks row (transcript_path, transcript_sha256,
     captions_source, ingested_at all derived from the fetcher).
  3. Mark fetch=completed.
  4. Probe transcript quality — pass → asr=skipped; fail → asr=pending +
     enqueue ingest_asr.
  5. Inspect format_tags — none of {slides_heavy, code_heavy, whiteboard,
     live_demo, diagram} → frame_sample=skipped; otherwise pending +
     enqueue ingest_frames.
  6. chunk=pending; enqueue ingest_chunk.

Caller wraps the call in `with conn.transaction():`. Every DB write and
queue send commits atomically — a crash mid-reconcile leaves no partial
state. A re-run upserts the same shape; ON CONFLICT clauses on talks and
ingest_step_status make the reconciler idempotent at the row level.
Queue sends are NOT idempotent (each re-run enqueues another chunk
message); the consuming worker uses claim_for_update to no-op on a
redelivery (see queues/workers.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from db.repos import ingest_step_status as iss
from db.repos.talks import Talk
from db.repos.talks import upsert as upsert_talk
from ingest.fetch import Fetcher
from ingest.quality import probe
from queues import pgmq_client

# Tags that imply visual content worth frame-sampling. Anything outside this
# set is text-only and skips the frames step. Public so cli_all can mirror
# this decision when computing which .mp4s the frames-drain gate requires.
VISUAL_TAGS = frozenset({"slides_heavy", "code_heavy", "whiteboard", "live_demo", "diagram"})


@dataclass(frozen=True)
class ReconcileResult:
    """What the reconciler decided. Used by tests + (eventually) logging."""

    asr_skipped: bool
    frames_skipped: bool
    chunk_msg_id: int


def fetch_reconcile(
    conn: psycopg.Connection,
    talk: Talk,
    fetcher: Fetcher,
) -> ReconcileResult:
    """Fetch transcript, persist talks row, set status rows, enqueue chunk
    (and asr/frames if not skipped). Must be called inside a transaction.
    """
    fetched = fetcher.fetch(talk.video_id)

    persisted = Talk(
        video_id=talk.video_id,
        title=talk.title,
        speaker=talk.speaker,
        url=talk.url,
        duration_sec=talk.duration_sec,
        format_tags=talk.format_tags,
        license=talk.license,
        captions_source=fetched.captions_source,
        transcript_path=str(fetched.transcript_path),
        transcript_sha256=fetched.transcript_sha256,
        accessed_at=talk.accessed_at,
        source_video_id=talk.source_video_id,
        source_start_sec=talk.source_start_sec,
        source_end_sec=talk.source_end_sec,
        notes=talk.notes,
        ingested_at=datetime.now(UTC),
    )
    upsert_talk(conn, persisted)

    # fetch just happened (we're the producer, not a consumer transitioning
    # through claim/mark). Direct upsert to completed is correct here.
    iss.upsert(conn, talk.video_id, step="fetch", status="completed")

    quality = probe(fetched.transcript_path, talk.duration_sec)
    asr_skipped = quality.passes
    iss.upsert(
        conn,
        talk.video_id,
        step="asr",
        status="skipped" if asr_skipped else "pending",
    )
    if not asr_skipped:
        pgmq_client.send(conn, "ingest_asr", {"video_id": talk.video_id, "step": "asr"})

    needs_frames = bool(VISUAL_TAGS.intersection(talk.format_tags))
    frames_skipped = not needs_frames
    iss.upsert(
        conn,
        talk.video_id,
        step="frame_sample",
        status="pending" if needs_frames else "skipped",
    )
    if needs_frames:
        pgmq_client.send(
            conn,
            "ingest_frames",
            {"video_id": talk.video_id, "step": "frame_sample"},
        )

    iss.upsert(conn, talk.video_id, step="chunk", status="pending")
    chunk_msg_id = pgmq_client.send(
        conn, "ingest_chunk", {"video_id": talk.video_id, "step": "chunk"}
    )

    return ReconcileResult(
        asr_skipped=asr_skipped,
        frames_skipped=frames_skipped,
        chunk_msg_id=chunk_msg_id,
    )
