"""Repository for the LangGraph observability tables.

Two tables (see ``db/migrations/0005_traces.sql``):

  * ``traces``       — one row per ``answer()`` call. Inserted at graph
                       start with ``status='running'``; updated at end.
  * ``trace_spans``  — one row per node execution. Batch-inserted at
                       graph completion to avoid hot-path overhead
                       (design §5: "one ``INSERT ... VALUES (...)`` per
                       trace").

Per design §5 + slice-3 brief contract #6: span ``input``/``output``
jsonb columns store ``chunk_ids: list[int]``, never chunk text — the
phase-4 trace viewer joins ``chunks`` on read.

The batched span flush uses the same ``list[str]`` + ``::jsonb[]`` cast
idiom that ``queues/pgmq_client.py::send_batch`` uses, because psycopg
adapts ``list[Jsonb]`` as a Postgres TEXT-array literal of JSON strings
(not ``jsonb[]``) and that fails the parameter type-check.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def start_trace(
    conn: psycopg.Connection,
    *,
    trace_id: str,
    query: str,
    corpus_id: str,
    generator_model_id: str | None = None,
) -> None:
    """Insert the trace row with ``status='running'``.

    ``started_at`` defaults to ``now()`` server-side. ``ended_at``,
    ``final_answer``, ``final_citations``, ``iterations``, ``latency_ms``
    are filled in by ``end_trace``.
    """
    conn.execute(
        """
        INSERT INTO traces (trace_id, query, corpus_id, generator_model_id, status)
        VALUES (%s, %s, %s, %s, 'running')
        """,
        (trace_id, query, corpus_id, generator_model_id),
    )


def end_trace(
    conn: psycopg.Connection,
    *,
    trace_id: str,
    status: str,
    final_answer: str | None,
    final_citations: list[dict] | None,
    iterations: int,
    latency_ms: int,
) -> None:
    """Mark the trace as completed/errored/aborted; fill terminal fields.

    ``status`` should be one of ``'completed'``, ``'error'``,
    ``'aborted'`` — the column has no CHECK constraint at v0, the
    caller owns the vocabulary. ``final_citations`` is stored as
    ``jsonb`` (``Jsonb`` wrapper handles the cast).
    """
    citations_jsonb = Jsonb(final_citations) if final_citations is not None else None
    conn.execute(
        """
        UPDATE traces
           SET ended_at = now(),
               status = %s,
               final_answer = %s,
               final_citations = %s,
               iterations = %s,
               latency_ms = %s
         WHERE trace_id = %s
        """,
        (
            status,
            final_answer,
            citations_jsonb,
            iterations,
            latency_ms,
            trace_id,
        ),
    )


def _coerce_iso(value: Any) -> str:
    """Return an ISO-8601 string for a ``datetime`` or pass through a
    string unchanged. We accept both because span emission constructs
    ``datetime`` objects in-memory but tests can hand-craft span dicts
    with pre-formatted timestamps."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def flush_spans(conn: psycopg.Connection, spans: list[dict]) -> None:
    """Batch-insert spans in a single round-trip.

    Span shape (per ``generate/trace.py::span``):
        {
            "span_id": str,
            "trace_id": str,
            "parent_span_id": str | None,
            "node_name": str,
            "iteration": int,
            "started_at": datetime | str,
            "ended_at": datetime | str,
            "input": dict,
            "output": dict,
            "metadata": dict,
            "status": str,                # 'ok' | 'error'
        }

    Empty input short-circuits with no DB call — same idiom as
    ``pgmq_client.send_batch``.

    The ``jsonb`` columns are passed as a TEXT array of JSON strings
    cast to ``jsonb[]`` server-side; ``list[Jsonb]`` would adapt to
    ``text[]`` and the parameter type-check would fail.
    """
    if not spans:
        return

    span_ids = [s["span_id"] for s in spans]
    trace_ids = [s["trace_id"] for s in spans]
    parent_ids = [s.get("parent_span_id") for s in spans]
    node_names = [s["node_name"] for s in spans]
    iterations = [int(s.get("iteration", 0)) for s in spans]
    started = [_coerce_iso(s["started_at"]) for s in spans]
    ended = [_coerce_iso(s.get("ended_at")) for s in spans]
    inputs = [json.dumps(s.get("input") or {}) for s in spans]
    outputs = [json.dumps(s.get("output") or {}) for s in spans]
    metadatas = [json.dumps(s.get("metadata") or {}) for s in spans]
    statuses = [s.get("status", "ok") for s in spans]

    conn.execute(
        """
        INSERT INTO trace_spans (
            span_id, trace_id, parent_span_id, node_name, iteration,
            started_at, ended_at, input, output, metadata, status
        )
        SELECT * FROM unnest(
            %s::text[],
            %s::text[],
            %s::text[],
            %s::text[],
            %s::int[],
            %s::timestamptz[],
            %s::timestamptz[],
            %s::jsonb[],
            %s::jsonb[],
            %s::jsonb[],
            %s::text[]
        )
        """,
        (
            span_ids,
            trace_ids,
            parent_ids,
            node_names,
            iterations,
            started,
            ended,
            inputs,
            outputs,
            metadatas,
            statuses,
        ),
    )
