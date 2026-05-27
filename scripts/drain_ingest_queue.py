"""One-off operational driver: drain a single PGMQ queue using a named handler.

This is the missing piece between ``ingest.cli`` (which enqueues
``ingest_frames`` + ``ingest_chunk`` jobs after ``fetch_reconcile``) and the
production worker loop that lands in a later phase. Used to take the
visual eval gate from STUB → real numbers by:

  1. ``python -m scripts.drain_ingest_queue ingest_frames``
     drains the frame_sample queue → populates ``frames`` table +
     enqueues N ``ingest_embed_frames`` jobs (one per frame_id).
  2. ``python -m scripts.drain_ingest_queue ingest_embed_frames``
     drains the embed_frames queue → populates
     ``frames.pooled_embedding`` + ``frame_patches`` via ColQwen.

Each ``process_one`` call wraps in its own transaction. A handler failure
rolls back AND redelivers via PGMQ visibility timeout, so a partial drain
is safe to resume.

Lives under ``scripts/`` (not ``eval/runners/``) because it is operational
glue for the existing ingest pipeline, not eval logic.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import psycopg

from db.conn import dsn as resolve_dsn
from ingest.handlers import (
    chunk_handler,
    embed_frames_handler,
    embed_text_handler,
    frames_handler,
)
from queues import workers as workers_mod

_HANDLERS: dict[str, Callable] = {
    "ingest_chunk": chunk_handler,
    "ingest_embed_text": embed_text_handler,
    "ingest_frames": frames_handler,
    "ingest_embed_frames": embed_frames_handler,
}


def drain(queue: str, *, max_messages: int | None = None, vt: int = 600) -> int:
    """Process every visible message on ``queue`` until the queue is empty.

    Returns the count of messages processed. ``vt`` is the per-message
    visibility timeout in seconds — bumped from the worker default (30s)
    because ColQwen patch encoding routinely exceeds 30s per frame on a
    cold model, which would cause redeliveries mid-handler.
    """
    handler = _HANDLERS[queue]
    n = 0
    with psycopg.connect(resolve_dsn()) as conn:
        # Connection-level autocommit OFF; each iteration opens its own
        # transaction via ``conn.transaction()`` so a handler failure rolls
        # back ONLY that message.
        while True:
            if max_messages is not None and n >= max_messages:
                break
            try:
                with conn.transaction():
                    processed = workers_mod.process_one(conn, queue, handler, vt=vt)
                if not processed:
                    break
                n += 1
                if n % 10 == 0:
                    sys.stdout.write(f"  ... processed {n} messages from {queue}\n")
                    sys.stdout.flush()
            except Exception as exc:  # noqa: BLE001 — log + continue
                sys.stderr.write(f"handler error on {queue} msg #{n + 1}: {exc!r}\n")
                # The exception aborts the current `with conn.transaction()`,
                # which rolls back. The message redelivers after `vt`. Stop
                # the drain so we don't busy-loop on a poison message.
                raise
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("queue", choices=sorted(_HANDLERS.keys()))
    ap.add_argument("--max", type=int, default=None, help="cap messages processed")
    ap.add_argument(
        "--vt",
        type=int,
        default=600,
        help="per-message visibility timeout in seconds (default 600 — ColQwen-safe)",
    )
    args = ap.parse_args(argv)
    n = drain(args.queue, max_messages=args.max, vt=args.vt)
    sys.stdout.write(f"drained {n} messages from {args.queue}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
