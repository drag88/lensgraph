"""Visual retrieval eval metrics — separate from bakeoff #1 (text embeddings).

Bakeoff #1 (text embeddings) measured the four text channels (dense,
sparse, multivec, rrf_4ch) and is locked. Visual retrieval — ColQwen
patches via ``retrieve.visual`` — is a 5th channel that is NOT covered
by the embeddings minimum in ADR 004 v3.1. This module provides the
parallel eval gate for visual retrieval against a dedicated
``visual_gold.jsonl`` corpus.

Two primary metrics + one optional comparison:

* ``VisualFrameRecall@k`` — frame-level recall using
  ``retrieve.visual.retrieve_frames``. A frame passes if its
  ``video_id`` matches AND its ``frame_sec`` falls within
  ``[gold.start_sec - tolerance_sec, gold.end_sec + tolerance_sec]``.
  Default tolerance is ``FRAME_SAMPLE_EVERY_SEC = 10.0`` (the cadence
  ``ingest.frames.sample`` uses), so a gold span that lands between two
  sampled frames still gets credit.

* ``VisualChunkTR@k`` — chunk-level recall using
  ``retrieve.visual.retrieve``. Same open-interval overlap convention as
  the text TR@5 from bakeoff #1: pass if any returned chunk has
  ``video_id == gold.video_id AND chunk.start_sec < gold.end_sec AND
  chunk.end_sec > gold.start_sec``.

* ``VisualLift@k`` (optional, computed when both run) — for each
  visual_gold example, compare a 5-channel RRF (BM25 + dense + sparse
  + multivec + visual) against the 4-channel RRF used in bakeoff #1
  (BM25 + dense + sparse + multivec). Lift =
  ``(with_visual_pass_count - without_visual_pass_count) / n``. The
  framing is: does adding the visual channel actually help on examples
  where visual signal is hypothesized to matter?

Methodology guardrails (from CLAUDE.md hard rules + experiment-tracking):

* Only ``dev_gold`` informs selection. The visual eval reads
  ``visual_gold.jsonl`` (a dev-only file in the same corpus dir; the
  validator enforces ``verified: true`` exactly as it does for the four
  text-eval files).
* ``test_gold.jsonl`` is locked. It is NOT consulted by this module
  under any flag.
* If ``visual_gold.jsonl`` is empty (the scaffolded default), the
  callable returns an empty result set; callers (the runner script,
  tests) surface this as "skip with clear message" rather than scoring
  a degenerate 0.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg

from retrieve import bm25, dense, multivec, rrf, sparse, visual
from retrieve.types import ChannelResult, FusedResult
from retrieve.visual import FrameResult

_VISUAL_GOLD_PATH = (
    Path(__file__).resolve().parent.parent
    / "corpora"
    / "ai_engineering_v0"
    / "visual_gold.jsonl"
)

# Frame sampling cadence used by ``ingest.frames.sample`` (every_sec=10.0
# default). A gold span at second 175 with frames at 170 and 180 should
# still get frame-level credit, so the per-example overlap window is
# expanded by ``FRAME_SAMPLE_EVERY_SEC`` on each side. Document the
# default loudly; do not silently tune.
FRAME_SAMPLE_EVERY_SEC = 10.0

# A visual_gold entry MUST carry at least one of these modality tags.
# Transcript-only / audio-only examples belong in dev_gold, not here —
# scoring them via the visual channel would be measurement-mode confusion.
VISUAL_MODALITIES = frozenset({"slide", "screen_code", "diagram", "whiteboard"})


@dataclass(frozen=True)
class VisualGoldQuery:
    example_id: str
    question: str
    video_id: str
    start_sec: float
    end_sec: float


def load_visual_gold(path: Path = _VISUAL_GOLD_PATH) -> list[VisualGoldQuery]:
    """Load single_clip rows from ``visual_gold.jsonl``.

    Empty file → empty list (the runner reports "0 visual gold examples,
    skipping" rather than zeroing the metric). Synthesis rows are not
    blended into visual metrics — they are reported separately if/when
    visual synthesis examples land (none today).

    Raises ``ValueError`` if any single_clip row's ``modality`` does not
    intersect ``VISUAL_MODALITIES``. The validator enforces the same
    rule on committed visual_gold.jsonl, but the loader re-checks at
    runtime because tests and tmp-fixture paths bypass the validator.
    Raising loudly here means a misclassified entry surfaces before
    measurement, never after writing partial eval_results."""
    if not path.exists():
        return []
    queries: list[VisualGoldQuery] = []
    modality_errors: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        ex = json.loads(line)
        if ex.get("question_type") != "single_clip":
            continue
        modality = ex.get("modality") or []
        if not (set(modality) & VISUAL_MODALITIES):
            modality_errors.append(
                f"id={ex.get('id', '?')}: modality={modality} lacks any of "
                f"{sorted(VISUAL_MODALITIES)} (visual_gold is for visual-bearing examples only)"
            )
            continue
        span = ex["gold_spans"][0]
        queries.append(
            VisualGoldQuery(
                example_id=ex["id"],
                question=ex["question"],
                video_id=ex["video_id"],
                start_sec=float(span["start_sec"]),
                end_sec=float(span["end_sec"]),
            )
        )
    if modality_errors:
        raise ValueError(
            f"{path}: {len(modality_errors)} visual_gold entry/entries lack "
            f"a visual modality tag:\n  - " + "\n  - ".join(modality_errors)
        )
    return queries


# --- Metric primitives (pure; no DB calls) ---------------------------------


def frame_passes_at_k(
    frames: Sequence[FrameResult],
    gold: VisualGoldQuery,
    *,
    k: int,
    tolerance_sec: float = FRAME_SAMPLE_EVERY_SEC,
) -> bool:
    """Return True if any of the top-k frames falls within tolerance of
    the gold span on the matching video."""
    lo = gold.start_sec - tolerance_sec
    hi = gold.end_sec + tolerance_sec
    for f in frames[:k]:
        if f.video_id == gold.video_id and lo <= f.frame_sec <= hi:
            return True
    return False


def chunk_passes_at_k(
    results: Sequence[ChannelResult | FusedResult],
    gold: VisualGoldQuery,
    *,
    k: int,
) -> bool:
    """Open-interval overlap on (gold.start, gold.end) with the chunk
    span, matching the text TR@5 convention."""
    for r in results[:k]:
        if (
            r.video_id == gold.video_id
            and r.end_sec > gold.start_sec
            and r.start_sec < gold.end_sec
        ):
            return True
    return False


# --- Per-channel measurement loops (DB-bound) ------------------------------


FrameFn = Callable[[psycopg.Connection, str], Sequence[FrameResult]]
ChunkFn = Callable[[psycopg.Connection, str], Sequence[ChannelResult | FusedResult]]


def _frame_to_dict(f: FrameResult) -> dict:
    return {
        "frame_id": f.frame_id,
        "video_id": f.video_id,
        "frame_sec": f.frame_sec,
        "image_path": f.image_path,
        "rank": f.rank,
        "score": float(f.score),
    }


def _chunk_to_dict(r: ChannelResult | FusedResult) -> dict:
    return {
        "chunk_id": r.chunk_id,
        "video_id": r.video_id,
        "start_sec": r.start_sec,
        "end_sec": r.end_sec,
        "rank": r.rank,
        "score": float(r.score),
    }


def visual_frame_recall_at_k(
    conn: psycopg.Connection,
    examples: Sequence[VisualGoldQuery],
    *,
    k: int = 5,
    tolerance_sec: float = FRAME_SAMPLE_EVERY_SEC,
    frame_fn: FrameFn | None = None,
) -> tuple[float, list[bool], list[list[dict]]]:
    """Aggregate VisualFrameRecall@k over ``examples``.

    Returns ``(score, per_example_passes, per_example_top_k_frames)``.
    The frame function defaults to ``retrieve.visual.retrieve_frames`` but
    can be injected for tests."""
    fn = frame_fn or (lambda c, q: visual.retrieve_frames(c, q, top_k=k))
    if not examples:
        return 0.0, [], []
    passes: list[bool] = []
    top_ks: list[list[dict]] = []
    for ex in examples:
        frames = list(fn(conn, ex.question))
        top_ks.append([_frame_to_dict(f) for f in frames[:k]])
        passes.append(frame_passes_at_k(frames, ex, k=k, tolerance_sec=tolerance_sec))
    return sum(passes) / len(examples), passes, top_ks


def visual_chunk_tr_at_k(
    conn: psycopg.Connection,
    examples: Sequence[VisualGoldQuery],
    *,
    k: int = 5,
    chunk_fn: ChunkFn | None = None,
) -> tuple[float, list[bool], list[list[dict]]]:
    """Aggregate VisualChunkTR@k over ``examples`` using the visual
    channel's chunk-level entry point.

    Returns ``(score, per_example_passes, per_example_top_k_chunks)``.
    Defaults to ``retrieve.visual.retrieve``; injectable for tests."""
    fn = chunk_fn or (lambda c, q: visual.retrieve(c, q, top_k=k))
    if not examples:
        return 0.0, [], []
    passes: list[bool] = []
    top_ks: list[list[dict]] = []
    for ex in examples:
        results = list(fn(conn, ex.question))
        top_ks.append([_chunk_to_dict(r) for r in results[:k]])
        passes.append(chunk_passes_at_k(results, ex, k=k))
    return sum(passes) / len(examples), passes, top_ks


def _rrf_with_visual(conn: psycopg.Connection, q: str, *, k: int) -> list[FusedResult]:
    b = bm25.retrieve(conn, q, top_k=30)
    d = dense.retrieve(conn, q, top_k=30)
    s = sparse.retrieve(conn, q, top_k=30)
    m = multivec.retrieve(conn, q, top_k=30)
    v = visual.retrieve(conn, q, top_k=30)
    return rrf.fuse(
        {"bm25": b, "dense": d, "sparse": s, "multivec": m, "visual": v},
        top_k=k,
    )


def _rrf_text_only(conn: psycopg.Connection, q: str, *, k: int) -> list[FusedResult]:
    # Identical to the rrf_4ch path measured in bakeoff #1, kept here to
    # keep the comparison apples-to-apples without cross-importing.
    b = bm25.retrieve(conn, q, top_k=30)
    d = dense.retrieve(conn, q, top_k=30)
    s = sparse.retrieve(conn, q, top_k=30)
    m = multivec.retrieve(conn, q, top_k=30)
    return rrf.fuse(
        {"bm25": b, "dense": d, "sparse": s, "multivec": m},
        top_k=k,
    )


def visual_lift_at_k(
    conn: psycopg.Connection,
    examples: Sequence[VisualGoldQuery],
    *,
    k: int = 5,
) -> dict:
    """Compare 5-channel RRF (text + visual) against 4-channel RRF
    (text only) on the same visual gold set.

    Returns a dict with::

        {
          "with_visual_pass_rate": float,
          "without_visual_pass_rate": float,
          "lift_pp": float,              # (with - without) * 100, percentage points
          "per_example": {
            example_id: {"with_visual": bool, "without_visual": bool}
          }
        }

    Empty ``examples`` returns zeros with empty per_example. Interpretation
    is the runner's job — small dev sets make even +/-10pp swings noisy.
    """
    if not examples:
        return {
            "with_visual_pass_rate": 0.0,
            "without_visual_pass_rate": 0.0,
            "lift_pp": 0.0,
            "per_example": {},
        }
    per_example: dict[str, dict[str, bool]] = {}
    with_pass = without_pass = 0
    for ex in examples:
        with_pass_ex = chunk_passes_at_k(_rrf_with_visual(conn, ex.question, k=k), ex, k=k)
        without_pass_ex = chunk_passes_at_k(_rrf_text_only(conn, ex.question, k=k), ex, k=k)
        per_example[ex.example_id] = {
            "with_visual": bool(with_pass_ex),
            "without_visual": bool(without_pass_ex),
        }
        with_pass += int(with_pass_ex)
        without_pass += int(without_pass_ex)
    n = len(examples)
    with_rate = with_pass / n
    without_rate = without_pass / n
    return {
        "with_visual_pass_rate": with_rate,
        "without_visual_pass_rate": without_rate,
        "lift_pp": round((with_rate - without_rate) * 100, 2),
        "per_example": per_example,
    }


def measure_visual(
    conn: psycopg.Connection,
    examples: Sequence[VisualGoldQuery],
    *,
    k: int = 5,
    tolerance_sec: float = FRAME_SAMPLE_EVERY_SEC,
    include_lift: bool = True,
) -> tuple[dict, dict[str, dict]]:
    """Run all visual metrics and return ``(summary, per_example_detail)``
    in the same shape ``eval.runners.measure_embeddings.measure_all_channels``
    uses, so the runner script can reuse the bakeoff #1 transactional
    write pattern.

    ``include_lift`` runs an extra 2-channel comparison per example
    (5-channel vs 4-channel RRF). It is optional because lift requires
    BOTH the visual stack AND the text stack to be populated for the
    same examples — and on tiny dev sets the result is mostly anecdotal.
    """
    frame_score, frame_passes, frame_topks = visual_frame_recall_at_k(
        conn, examples, k=k, tolerance_sec=tolerance_sec
    )
    chunk_score, chunk_passes, chunk_topks = visual_chunk_tr_at_k(
        conn, examples, k=k
    )

    summary: dict = {
        "visual_frame_recall_at_k": frame_score,
        "visual_chunk_tr_at_k": chunk_score,
        "k": k,
        "frame_tolerance_sec": tolerance_sec,
        "n_examples": len(examples),
        "per_example": {
            ex.example_id: {
                "frame_pass": bool(fp),
                "chunk_pass": bool(cp),
            }
            for ex, fp, cp in zip(examples, frame_passes, chunk_passes, strict=True)
        },
    }
    per_example_detail: dict[str, dict] = {
        ex.example_id: {
            "gold_span": {
                "video_id": ex.video_id,
                "start_sec": ex.start_sec,
                "end_sec": ex.end_sec,
            },
            "frame_pass_at_k": bool(fp),
            "chunk_pass_at_k": bool(cp),
            "top_k_frames": ftk,
            "top_k_chunks": ctk,
        }
        for ex, fp, cp, ftk, ctk in zip(
            examples, frame_passes, chunk_passes, frame_topks, chunk_topks, strict=True
        )
    }

    if include_lift and examples:
        lift = visual_lift_at_k(conn, examples, k=k)
        summary["visual_lift"] = {
            "with_visual_pass_rate": lift["with_visual_pass_rate"],
            "without_visual_pass_rate": lift["without_visual_pass_rate"],
            "lift_pp": lift["lift_pp"],
        }
        for ex_id, p in lift["per_example"].items():
            per_example_detail[ex_id]["lift"] = p

    return summary, per_example_detail
