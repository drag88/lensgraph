"""Real per-channel TimestampRecall@5 sweep for the embeddings bakeoff.

Loads single_clip examples from ``dev_gold.jsonl``, runs each text
retrieval channel (``dense``, ``sparse``, ``multivec``, ``rrf_4ch``)
against the live retrieval surface, and returns per-channel TR@5 scores
ready to drop into ``eval_runs.summary``.

TR@5 (from ``docs/eval-methodology.md`` + ADR 004 v3.1): for an example
with gold span ``[s, e]`` on video ``V``, a channel ``passes@5`` if any of
the top-5 retrieved chunks satisfies
``chunk.video_id == V`` AND open-interval overlap
``chunk.start_sec < e AND chunk.end_sec > s``. ``TR@5 = passes / total``.

Per ADR 004 v3.1, only ``dev_gold`` (single_clip subset) informs
selection. ``synthesis.jsonl`` is reported separately and is not blended
into this average.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg

from retrieve import bm25, dense, multivec, rrf, sparse
from retrieve.types import ChannelResult, FusedResult

_DEV_GOLD_PATH = (
    Path(__file__).resolve().parent.parent
    / "corpora"
    / "ai_engineering_v0"
    / "dev_gold.jsonl"
)


@dataclass(frozen=True)
class GoldQuery:
    example_id: str
    question: str
    video_id: str
    start_sec: float
    end_sec: float


def load_dev_gold_single_clip(path: Path = _DEV_GOLD_PATH) -> list[GoldQuery]:
    """Load single_clip examples from ``dev_gold.jsonl``.

    Skips non-single_clip rows (synthesis examples carry multi-video gold
    spans and per ``docs/eval-methodology.md`` are reported separately,
    not blended into TR@5). Negatives have empty gold_spans by schema and
    cannot score a recall metric — they would be skipped here too, but
    ``dev_gold.jsonl`` does not contain any.
    """
    queries: list[GoldQuery] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("question_type") != "single_clip":
                continue
            span = ex["gold_spans"][0]
            queries.append(
                GoldQuery(
                    example_id=ex["id"],
                    question=ex["question"],
                    video_id=ex["video_id"],
                    start_sec=float(span["start_sec"]),
                    end_sec=float(span["end_sec"]),
                )
            )
    return queries


def _passes_at_k(
    results: Sequence[ChannelResult | FusedResult],
    gold: GoldQuery,
    *,
    k: int,
) -> bool:
    for r in results[:k]:
        if (
            r.video_id == gold.video_id
            and r.end_sec > gold.start_sec
            and r.start_sec < gold.end_sec
        ):
            return True
    return False


ChannelFn = Callable[
    [psycopg.Connection, str],
    list[ChannelResult] | list[FusedResult],
]


def tr_at_k(
    conn: psycopg.Connection,
    examples: Sequence[GoldQuery],
    channel_fn: ChannelFn,
    *,
    k: int = 5,
) -> tuple[float, list[bool]]:
    """Aggregate TR@k for ``examples`` against ``channel_fn``.

    Returns ``(score, per_example_passes)`` where ``score = passes / total``.
    The channel function is responsible for returning at least ``k`` ranked
    results (caller controls top_k on the underlying retrieve module).
    """
    if not examples:
        return 0.0, []
    passes = [_passes_at_k(channel_fn(conn, ex.question), ex, k=k) for ex in examples]
    return sum(passes) / len(examples), passes


def _dense_fn(conn: psycopg.Connection, q: str) -> list[ChannelResult]:
    return dense.retrieve(conn, q, top_k=5)


def _sparse_fn(conn: psycopg.Connection, q: str) -> list[ChannelResult]:
    return sparse.retrieve(conn, q, top_k=5)


def _multivec_fn(conn: psycopg.Connection, q: str) -> list[ChannelResult]:
    return multivec.retrieve(conn, q, top_k=5)


def _rrf_4ch_fn(conn: psycopg.Connection, q: str) -> list[FusedResult]:
    # 30 per channel is the production-default fan-in (design §4); the fused
    # output is truncated to 5 for the TR@5 measurement.
    b = bm25.retrieve(conn, q, top_k=30)
    d = dense.retrieve(conn, q, top_k=30)
    s = sparse.retrieve(conn, q, top_k=30)
    m = multivec.retrieve(conn, q, top_k=30)
    return rrf.fuse({"bm25": b, "dense": d, "sparse": s, "multivec": m}, top_k=5)


CHANNEL_FNS: dict[str, ChannelFn] = {
    "dense": _dense_fn,
    "sparse": _sparse_fn,
    "multivec": _multivec_fn,
    "rrf_4ch": _rrf_4ch_fn,
}

# Vector-only channels per ADR 004 v3.1 minimum
# (``timestamp_recall_at_5_vector_only_min``): RRF is reported but blends
# BM25 so it does not count toward the vector-only minimum.
VECTOR_ONLY_CHANNELS = ("dense", "sparse", "multivec")


def measure_all_channels(
    conn: psycopg.Connection,
    examples: Sequence[GoldQuery],
    *,
    k: int = 5,
) -> dict:
    """Run TR@k across every channel in ``CHANNEL_FNS``.

    Returns a dict suitable for embedding in ``eval_runs.summary``:

      * ``per_channel_tr_at_5`` — ``{channel: float}``
      * ``per_example`` — ``{example_id: {channel: bool}}``
      * ``n_examples`` — total scored
      * ``vector_best_score`` — max over VECTOR_ONLY_CHANNELS (drives
        the selection-rule minimum check)
    """
    per_channel: dict[str, float] = {}
    per_example: dict[str, dict[str, bool]] = {ex.example_id: {} for ex in examples}
    for name, fn in CHANNEL_FNS.items():
        score, passes = tr_at_k(conn, examples, fn, k=k)
        per_channel[name] = score
        for ex, p in zip(examples, passes, strict=True):
            per_example[ex.example_id][name] = bool(p)
    return {
        "per_channel_tr_at_5": per_channel,
        "per_example": per_example,
        "n_examples": len(examples),
        "vector_best_score": max(per_channel[c] for c in VECTOR_ONLY_CHANNELS),
    }
