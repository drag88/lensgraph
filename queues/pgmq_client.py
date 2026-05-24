"""Thin psycopg wrapper around PGMQ SQL functions.

PGMQ is functional — `pgmq.send(queue, jsonb)`, `pgmq.read(queue, vt, qty)`,
`pgmq.archive(queue, msg_id)`, etc. We wrap them in typed helpers so the
rest of the codebase doesn't carry raw `SELECT pgmq.*(...)` calls.

Queues are created at startup (NOT in migrations) because the PGMQ API is
itself functional — `pgmq.create()` is a procedure, not DDL. See design §2
(comment block) and §7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

# Phase-1 queue topology per design §3. Each queue has a sibling `_dlq`
# created alongside; DLQs are NOT auto-created by pgmq.
QUEUES: tuple[str, ...] = (
    "ingest_fetch",
    "ingest_asr",
    "ingest_frames",
    "ingest_chunk",
    "ingest_embed_text",
    "ingest_embed_frames",
)


@dataclass(frozen=True)
class Message:
    """One row from pgmq.read(). Payload is the deserialized jsonb body."""

    msg_id: int
    read_ct: int
    message: dict[str, Any]


def ensure_queues(
    conn: psycopg.Connection,
    queues: tuple[str, ...] = QUEUES,
) -> None:
    """Idempotently create each queue and its `_dlq` sibling. pgmq.create
    uses CREATE TABLE IF NOT EXISTS internally, so it is safe to call on
    every worker boot."""
    for q in queues:
        conn.execute("SELECT pgmq.create(%s)", (q,))
        conn.execute("SELECT pgmq.create(%s)", (f"{q}_dlq",))


def send(conn: psycopg.Connection, queue: str, payload: dict[str, Any]) -> int:
    """Enqueue a jsonb payload. Returns the assigned msg_id."""
    row = conn.execute(
        "SELECT pgmq.send(%s, %s)", (queue, Jsonb(payload))
    ).fetchone()
    return row[0]


def read(
    conn: psycopg.Connection,
    queue: str,
    *,
    vt: int = 30,
    qty: int = 1,
) -> list[Message]:
    """Dequeue up to `qty` messages with a `vt`-second visibility timeout.

    Transactional consume: pgmq.read() is plain SQL; rolling back the
    surrounding transaction undoes the read (message becomes visible again
    immediately). The standard worker pattern wraps read + handler + archive
    in one transaction so a crash mid-handler re-delivers the message.
    """
    rows = conn.execute(
        "SELECT msg_id, read_ct, message FROM pgmq.read(%s, %s, %s)",
        (queue, vt, qty),
    ).fetchall()
    return [Message(msg_id=r[0], read_ct=r[1], message=r[2]) for r in rows]


def archive(conn: psycopg.Connection, queue: str, msg_id: int) -> bool:
    """Move a message to the archive table (ack). Returns True if archived."""
    row = conn.execute("SELECT pgmq.archive(%s, %s)", (queue, msg_id)).fetchone()
    return bool(row[0])


def delete(conn: psycopg.Connection, queue: str, msg_id: int) -> bool:
    """Drop a message outright (no archive copy). Returns True if deleted."""
    row = conn.execute("SELECT pgmq.delete(%s, %s)", (queue, msg_id)).fetchone()
    return bool(row[0])


def purge(conn: psycopg.Connection, queue: str) -> int:
    """Delete all messages from the queue. Returns the count purged."""
    row = conn.execute("SELECT pgmq.purge_queue(%s)", (queue,)).fetchone()
    return int(row[0])
