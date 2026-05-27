"""Span emission helpers + state snapshotter for the LangGraph loop.

Three primitives:

  * :func:`new_uuid7` — RFC 9562 UUIDv7 (time-ordered) as a hex string.
    Used for ``trace_id`` and ``span_id``. Tiny inline implementation
    so we don't take a dep on a uuid7 library.

  * :func:`serialize_state_snapshot` — turn an ``AgentState`` slice into
    a jsonb-safe dict. Bulky ``retrieved``/``reranked`` lists collapse
    to ``*_chunk_ids: list[int]`` per design §5 (the trace viewer joins
    ``chunks`` on read). Pydantic models dump via ``model_dump``.

  * :func:`span` — a context manager that wraps one node execution.
    Records start time, captures the input snapshot, yields a mutable
    dict the caller fills with ``output_snapshot`` and ``metadata``,
    then assembles a span dict and appends to ``buffer``. Exceptions
    are recorded with ``status='error'`` and re-raised.
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from generate.state import AgentState


def new_uuid7() -> str:
    """Return a UUIDv7 as a 36-char hyphenated lowercase hex string.

    RFC 9562 layout:
      - 48 bits: ms since Unix epoch
      - 4 bits: version (0111)
      - 12 bits: random
      - 2 bits: variant (10)
      - 62 bits: random

    Time-ordered so traces sort naturally by start time. We hand-roll
    rather than take a dep — the spec is short, and ``secrets.token_bytes``
    + bit-twiddling is ~10 lines.
    """
    unix_ms = int(time.time_ns() // 1_000_000) & 0xFFFFFFFFFFFF  # 48 bits
    rand_a = int.from_bytes(secrets.token_bytes(2), "big") & 0x0FFF  # 12 bits
    rand_b = int.from_bytes(secrets.token_bytes(8), "big") & 0x3FFFFFFFFFFFFFFF  # 62 bits

    # Assemble the 128-bit integer.
    n = (unix_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0x2 << 62) | rand_b
    hex_ = f"{n:032x}"
    return f"{hex_[0:8]}-{hex_[8:12]}-{hex_[12:16]}-{hex_[16:20]}-{hex_[20:32]}"


def _jsonable(value: Any) -> Any:
    """Recursively coerce a value into a JSON-safe shape.

    ``BaseModel`` → ``model_dump`` (jsonable_encoder semantics). Lists
    and dicts are walked. Primitives pass through. Datetimes become
    ISO strings. Unknown types fall back to ``str(value)`` so the trace
    is never blocked by a stray object.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def serialize_state_snapshot(state: AgentState) -> dict:
    """Snapshot an ``AgentState`` slice into a jsonb-safe dict.

    Bulky ``retrieved`` / ``reranked`` lists collapse to
    ``retrieved_chunk_ids`` / ``reranked_chunk_ids`` per design §5.
    Pydantic models dump via ``model_dump``. Primitives pass through.

    Snapshot is a SHALLOW copy of the state keys present at call time —
    we don't deep-copy chunk text or token embeds, which is the whole
    point of collapsing to ids.
    """
    out: dict = {}
    for k, v in state.items():
        if k in ("retrieved", "reranked"):
            # v is list[RetrievedChunk]; we only persist the ids.
            out[f"{k}_chunk_ids"] = [int(c.chunk_id) for c in v]
            continue
        out[k] = _jsonable(v)
    return out


@contextmanager
def span(
    buffer: list[dict],
    *,
    trace_id: str,
    node_name: str,
    iteration: int,
    input_snapshot: dict,
    parent_span_id: str | None = None,
) -> Iterator[dict]:
    """Wrap one node execution; emit one span dict into ``buffer``.

    Usage::

        carrier = {}                     # mutable yielded dict
        with span(buf, trace_id=..., node_name="plan",
                  iteration=0, input_snapshot=...) as carrier:
            do_node_work()
            carrier["output_snapshot"] = serialize_state_snapshot(state)
            carrier["metadata"] = {"latency_ms": 12}

    On normal exit the span is appended with ``status='ok'``. On
    exception, the span is appended with ``status='error'``,
    ``metadata['error'] = str(exc)``, the exception is re-raised, and
    the partial output_snapshot (if the caller set one) is preserved.

    ``input_snapshot`` is captured eagerly (before node runs) — the
    caller is responsible for serialising it via
    ``serialize_state_snapshot`` BEFORE entering the context, so we
    record the pre-node state rather than the post-node state.
    """
    span_id = new_uuid7()
    carrier: dict = {"output_snapshot": None, "metadata": None}
    started_at = datetime.now(UTC)
    perf_start = time.perf_counter()
    raised: BaseException | None = None
    try:
        yield carrier
    except BaseException as exc:
        raised = exc
        raise
    finally:
        ended_at = datetime.now(UTC)
        latency_ms = int((time.perf_counter() - perf_start) * 1000)
        metadata = dict(carrier.get("metadata") or {})
        metadata.setdefault("latency_ms", latency_ms)
        status = "ok"
        if raised is not None:
            status = "error"
            metadata["error"] = f"{type(raised).__name__}: {raised}"
        buffer.append(
            {
                "span_id": span_id,
                "trace_id": trace_id,
                "parent_span_id": parent_span_id,
                "node_name": node_name,
                "iteration": int(iteration),
                "started_at": started_at,
                "ended_at": ended_at,
                "input": carrier.get("input_snapshot") or input_snapshot,
                "output": carrier.get("output_snapshot") or {},
                "metadata": metadata,
                "status": status,
            }
        )


def verify_threshold_from_env(default: float = 0.6) -> float:
    """Resolve the Verify confidence threshold from
    ``LENSGRAPH_VERIFY_CONFIDENCE_THRESHOLD`` (design §5).

    Invalid float → fall back to ``default`` silently; this is a tuning
    knob, not a correctness gate."""
    raw = os.environ.get("LENSGRAPH_VERIFY_CONFIDENCE_THRESHOLD")
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default
