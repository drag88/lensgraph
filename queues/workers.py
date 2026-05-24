"""Worker skeleton.

Phase-1 step 6 ships the testable unit only — `process_one()`. The full
multi-queue worker loop + `python -m queues.workers --queues ...` CLI lands
in step 7+ when real handlers exist (ingest fetch, chunk, embed).

`process_one()` is the atomic contract:
  - dequeue one message
  - claim the matching ingest_step_status row by EXACT (video_id, step,
    entity_id) from the payload
  - if the row is already completed/skipped, ack and exit (work was done)
  - otherwise run the handler, mark completed, archive the message
  - all four DB operations live in the CALLER'S transaction — wrap in
    `with conn.transaction():` so a handler crash rolls back the read
    (message redelivers) and the claim (attempts not artificially bumped).

Payload contract: every message body is a JSON object with at minimum
`video_id` (str) and `step` (str). `entity_id` (int) is optional and
defaults to 0 (whole-video steps).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg

from db.repos import ingest_step_status as iss
from queues import pgmq_client

Handler = Callable[[psycopg.Connection, dict[str, Any]], None]


def process_one(
    conn: psycopg.Connection,
    queue: str,
    handler: Handler,
    *,
    vt: int = 30,
) -> bool:
    """Process at most one message. Returns True if a message was processed,
    False if the queue was empty.

    Caller is responsible for wrapping the call in a transaction. The four
    DB writes (claim, handler-side effects, mark_completed, archive) must
    commit atomically with the read so a handler crash re-delivers the job.
    """
    messages = pgmq_client.read(conn, queue, vt=vt, qty=1)
    if not messages:
        return False
    msg = messages[0]
    payload = msg.message
    video_id = payload["video_id"]
    step = payload["step"]
    entity_id = int(payload.get("entity_id", 0))

    claim = iss.claim_for_update(conn, video_id, step, entity_id=entity_id)
    if claim.status in ("completed", "skipped"):
        # Already done. Ack the redelivery / duplicate and exit.
        pgmq_client.archive(conn, queue, msg.msg_id)
        return True

    handler(conn, payload)
    iss.mark_completed(conn, video_id, step, entity_id=entity_id)
    pgmq_client.archive(conn, queue, msg.msg_id)
    return True
