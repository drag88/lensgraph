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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg

from chunking import get_chunker
from chunking.fixed_window import STRATEGY_NAME
from db.repos import chunks as chunks_repo
from db.repos import embeds as embeds_repo
from db.repos import ingest_step_status as iss
from db.repos import talks as talks_repo
from embed import bge_m3
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
        iss.upsert(
            conn, video_id, step="embed_text", entity_id=chunk_id, status="pending"
        )

    pgmq_client.send_batch(
        conn,
        "ingest_embed_text",
        [
            {"video_id": video_id, "step": "embed_text", "entity_id": cid}
            for cid in chunk_ids
        ],
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
        raise ValueError(
            f"no chunk row for chunk_id={chunk_id}, video_id={video_id!r}"
        )

    out = bge_m3.encode([text])
    embeds_repo.upsert_dense(conn, chunk_id, out.dense[0])
    embeds_repo.upsert_sparse(conn, chunk_id, out.sparse[0])
    embeds_repo.replace_token_embeds(conn, chunk_id, out.multi[0])
