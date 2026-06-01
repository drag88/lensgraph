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
    Path(__file__).resolve().parent.parent / "corpora" / "ai_engineering_v0" / "dev_gold.jsonl"
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


def _fully_contains(r: ChannelResult | FusedResult, gold: GoldQuery) -> bool:
    return (
        r.video_id == gold.video_id and r.start_sec <= gold.start_sec and r.end_sec >= gold.end_sec
    )


def _iou_vs_gold(r: ChannelResult | FusedResult, gold: GoldQuery) -> float:
    if r.video_id != gold.video_id:
        return 0.0
    inter = max(0.0, min(r.end_sec, gold.end_sec) - max(r.start_sec, gold.start_sec))
    union = (r.end_sec - r.start_sec) + (gold.end_sec - gold.start_sec) - inter
    return inter / union if union > 0 else 0.0


def gold_span_contained_at_k(
    results: Sequence[ChannelResult | FusedResult], gold: GoldQuery, *, k: int
) -> bool:
    """True if ANY of the top-k chunks FULLY contains the gold span
    (``chunk.start <= gold.start`` AND ``chunk.end >= gold.end``).

    Penalizes over-segmentation: chunks smaller than the gold span cannot
    contain it, so a too-fine strategy scores low here. Like TR@k, it favors
    larger chunks — read it alongside ``iou_at_1`` (which penalizes them).
    """
    return any(_fully_contains(r, gold) for r in results[:k])


def iou_at_1(results: Sequence[ChannelResult | FusedResult], gold: GoldQuery) -> float:
    """Intersection-over-union of the rank-1 chunk vs the gold span.

    Penalizes chunks that are too large (big union) or off-center. This is the
    metric where coarse strategies (e.g. a 150s transcript_segment chunk around
    a 20s span) lose. 0.0 if no results or wrong video.
    """
    return _iou_vs_gold(results[0], gold) if results else 0.0


ChannelFn = Callable[
    [psycopg.Connection, str],
    list[ChannelResult] | list[FusedResult],
]


def _result_to_dict(r: ChannelResult | FusedResult) -> dict:
    return {
        "chunk_id": r.chunk_id,
        "video_id": r.video_id,
        "start_sec": r.start_sec,
        "end_sec": r.end_sec,
        "rank": r.rank,
        "score": float(r.score),
    }


def tr_at_k(
    conn: psycopg.Connection,
    examples: Sequence[GoldQuery],
    channel_fn: ChannelFn,
    *,
    k: int = 5,
) -> tuple[float, list[bool], list[list[dict]]]:
    """Aggregate TR@k for ``examples`` against ``channel_fn``.

    Returns ``(score, per_example_passes, per_example_top_k)`` where
    ``score = passes / total`` and ``per_example_top_k`` is the list of
    top-k retrieved chunks (as JSON-serialisable dicts) for each input,
    aligned by index. The channel function is responsible for returning
    at least ``k`` ranked results (caller controls top_k on the
    underlying retrieve module).
    """
    if not examples:
        return 0.0, [], []
    passes: list[bool] = []
    top_ks: list[list[dict]] = []
    for ex in examples:
        results = channel_fn(conn, ex.question)
        top_ks.append([_result_to_dict(r) for r in results[:k]])
        passes.append(_passes_at_k(results, ex, k=k))
    return sum(passes) / len(examples), passes, top_ks


def channel_recall(
    conn: psycopg.Connection,
    examples: Sequence[GoldQuery],
    channel_fn: ChannelFn,
    *,
    k: int = 5,
) -> dict:
    """One pass over ``examples`` returning TR@k, GoldSpanContained@k, and
    IoU@1 for ``channel_fn``, plus per-example detail. One retrieval call per
    example (vs three if the metrics were computed separately)."""
    if not examples:
        return {
            "tr_at_k": 0.0,
            "gold_span_contained_at_k": 0.0,
            "iou_at_1": 0.0,
            "per_example_pass": [],
            "per_example_contained": [],
            "per_example_iou": [],
            "top_ks": [],
        }
    tr: list[bool] = []
    contained: list[bool] = []
    ious: list[float] = []
    top_ks: list[list[dict]] = []
    for ex in examples:
        results = channel_fn(conn, ex.question)
        top_ks.append([_result_to_dict(r) for r in results[:k]])
        tr.append(_passes_at_k(results, ex, k=k))
        contained.append(gold_span_contained_at_k(results, ex, k=k))
        ious.append(iou_at_1(results, ex))
    n = len(examples)
    return {
        "tr_at_k": sum(tr) / n,
        "gold_span_contained_at_k": sum(contained) / n,
        "iou_at_1": sum(ious) / n,
        "per_example_pass": tr,
        "per_example_contained": contained,
        "per_example_iou": ious,
        "top_ks": top_ks,
    }


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
) -> tuple[dict, dict[str, dict]]:
    """Run TR@k across every channel in ``CHANNEL_FNS``.

    Returns ``(summary, per_example_detail)``:

      * ``summary`` is the lightweight aggregate destined for
        ``eval_runs.summary``:

          - ``per_channel_tr_at_5`` — ``{channel: float}``
          - ``per_example`` — ``{example_id: {channel: bool}}``
          - ``n_examples`` — total scored
          - ``vector_best_score`` — max over VECTOR_ONLY_CHANNELS
            (drives the selection-rule minimum check)

      * ``per_example_detail`` carries the heavier per-example data the
        script unpacks into one ``eval_results`` row per example:

          - ``gold_span`` — the example's video_id + start/end seconds
          - ``pass_at_5`` — per-channel boolean (same as in summary)
          - ``top_k_per_channel`` — per-channel list of the top-k chunks
            actually returned (chunk_id, video_id, span, rank, score)
    """
    per_channel: dict[str, float] = {}
    per_example_pass: dict[str, dict[str, bool]] = {ex.example_id: {} for ex in examples}
    per_example_top_k: dict[str, dict[str, list[dict]]] = {ex.example_id: {} for ex in examples}
    for name, fn in CHANNEL_FNS.items():
        score, passes, top_ks = tr_at_k(conn, examples, fn, k=k)
        per_channel[name] = score
        for ex, p, tk in zip(examples, passes, top_ks, strict=True):
            per_example_pass[ex.example_id][name] = bool(p)
            per_example_top_k[ex.example_id][name] = tk

    summary = {
        "per_channel_tr_at_5": per_channel,
        "per_example": per_example_pass,
        "n_examples": len(examples),
        "vector_best_score": max(per_channel[c] for c in VECTOR_ONLY_CHANNELS),
    }
    per_example_detail = {
        ex.example_id: {
            "gold_span": {
                "video_id": ex.video_id,
                "start_sec": ex.start_sec,
                "end_sec": ex.end_sec,
            },
            "pass_at_5": per_example_pass[ex.example_id],
            "top_k_per_channel": per_example_top_k[ex.example_id],
        }
        for ex in examples
    }
    return summary, per_example_detail
