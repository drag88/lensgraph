"""PGMQ worker handlers for ingest pipeline steps.

Each handler:
  - takes (conn, payload)
  - reads/mutates state under the caller's transaction (process_one wraps
    both DB writes and queue ack atomically — see queues/workers.py)
  - raises on contract violations (missing talks row, missing chunk row,
    etc.) — the surrounding transaction rolls back and PGMQ redelivers
  - is idempotent: re-running for the same payload produces the same
    final state, never duplicates.

The chunk handler fans out one ingest_step_status row + one
ingest_embed_text message per persisted chunk. The status row is upserted
BEFORE the message is enqueued so the embed_text handler's
claim_for_update always finds its row (otherwise IngestStepNotFoundError).

The frames handler mirrors that shape: one status row + one
ingest_embed_frames message per persisted frame_id. Pooled / per-patch
embeddings are written by the (still-deferred) embed_frames_handler in
slice 2 — frames_handler only sets the substrate up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg

from chunking import get_chunker
from chunking.fixed_window import STRATEGY_NAME
from db.repos import chunks as chunks_repo
from db.repos import embeds as embeds_repo
from db.repos import frames as frames_repo
from db.repos import ingest_step_status as iss
from db.repos import talks as talks_repo
from embed import bge_m3
from ingest import frames as frames_mod
from queues import pgmq_client


def chunk_handler(conn: psycopg.Connection, payload: dict[str, Any]) -> None:
    """ingest_chunk: read transcript, chunk, persist, fan out embed_text jobs.

    Payload: {video_id, step: "chunk", entity_id=0 (whole-video step)}.
    """
    video_id = payload["video_id"]
    talk = talks_repo.get(conn, video_id)
    if talk is None:
        raise ValueError(f"no talks row for video_id={video_id!r}")

    transcript_text = Path(talk.transcript_path).read_text(encoding="utf-8")
    chunker = get_chunker(STRATEGY_NAME)
    chunks = chunker(transcript_text, frames=[], video_id=video_id)
    if not chunks:
        # Empty transcript — nothing to fan out; handler ack-and-exit.
        return

    chunk_ids = chunks_repo.upsert(conn, chunks)

    for chunk_id in chunk_ids:
        iss.upsert(conn, video_id, step="embed_text", entity_id=chunk_id, status="pending")

    pgmq_client.send_batch(
        conn,
        "ingest_embed_text",
        [{"video_id": video_id, "step": "embed_text", "entity_id": cid} for cid in chunk_ids],
    )


def embed_text_handler(conn: psycopg.Connection, payload: dict[str, Any]) -> None:
    """ingest_embed_text: encode one chunk's text, write all 3 embed tables.

    Payload: {video_id, step: "embed_text", entity_id: chunk_id}.
    Loads BGE-M3 lazily via embed.bge_m3 (singleton, amortised across calls).
    """
    video_id = payload["video_id"]
    chunk_id = int(payload["entity_id"])

    text = chunks_repo.get_text(conn, chunk_id=chunk_id, video_id=video_id)
    if text is None:
        raise ValueError(f"no chunk row for chunk_id={chunk_id}, video_id={video_id!r}")

    out = bge_m3.encode([text])
    embeds_repo.upsert_dense(conn, chunk_id, out.dense[0])
    embeds_repo.upsert_sparse(conn, chunk_id, out.sparse[0])
    embeds_repo.replace_token_embeds(conn, chunk_id, out.multi[0])


def frames_handler(conn: psycopg.Connection, payload: dict[str, Any]) -> None:
    """ingest_frames: sample local video frames, persist, fan out embed_frames.

    Payload: {video_id, step: "frame_sample", entity_id=0 (whole-video)}.

    Chapter-slice rule: when ``talk.source_video_id`` is set, the physical
    video is the parent (``source_video_id``); we sample the parent's
    ``[source_start_sec, source_end_sec)`` window and let ``frames.sample``
    align ``frame_sec`` to the talk's zero. Regular videos sample
    ``[0, duration_sec)`` of their own file.

    Out of scope for slice 1: pooled / per-patch ColQwen embeddings — the
    embed_frames status rows + ingest_embed_frames messages enqueued here
    feed the slice-2 ``embed_frames_handler`` (not yet implemented).
    """
    video_id = payload["video_id"]
    talk = talks_repo.get(conn, video_id)
    if talk is None:
        raise ValueError(f"no talks row for video_id={video_id!r}")

    video_path = frames_mod.default_video_path_for_talk(talk)
    if not video_path.exists():
        raise FileNotFoundError(
            f"local video not found at {video_path} for video_id={video_id!r}; "
            "frames_handler requires a pre-staged .mp4 under "
            "videos/<corpus>/<source_video_id or video_id>.mp4"
        )

    if talk.source_video_id is not None:
        start = float(talk.source_start_sec or 0.0)
        duration = float((talk.source_end_sec or 0.0) - (talk.source_start_sec or 0.0))
    else:
        start = 0.0
        duration = float(talk.duration_sec)

    samples = frames_mod.sample(
        video_path,
        video_id=video_id,
        every_sec=10.0,
        start_sec=start,
        duration_sec=duration,
    )
    if not samples:
        return

    frame_ids = frames_repo.upsert(conn, samples)

    for fid in frame_ids:
        iss.upsert(conn, video_id, step="embed_frames", entity_id=fid, status="pending")

    pgmq_client.send_batch(
        conn,
        "ingest_embed_frames",
        [{"video_id": video_id, "step": "embed_frames", "entity_id": fid} for fid in frame_ids],
    )
