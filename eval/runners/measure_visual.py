"""Visual retrieval eval metrics — separate from bakeoff #1 (text embeddings).

Bakeoff #1 (text embeddings) measured the four text channels (dense,
sparse, multivec, rrf_4ch) and is locked. Visual retrieval — ColQwen
patches via ``retrieve.visual`` — is a 5th channel that is NOT covered
by the embeddings minimum in ADR 004 v3.1. This module provides the
eval gate for visual retrieval against a dedicated
``visual_gold.jsonl`` corpus.

Four metrics + one comparison:

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

* ``AnswerTermHit@k`` — TEXT-SIDE diagnostic. For rows whose curator
  filled in ``visual_evidence`` (a list of answer-bearing strings
  confirmed against the gold FRAME), the row passes iff some top-k chunk
  both span-overlaps the gold AND its TRANSCRIPT TEXT contains the
  curator-supplied terms. This is NOT a visual-modality answer-grounding
  metric. ColQwen frames are not OCR'd or judged. The metric only checks
  whether the SURFACED TRANSCRIPT CHUNK (regardless of which retriever
  surfaced it) quotes what the curator saw on the slide.

* ``VisualAnswerGrounding@k`` — FRAME-SIDE answer-grounding metric. For
  rows whose curator filled in ``visual_evidence``, call
  ``retrieve.visual.retrieve_frames`` to get the top-k frames by the
  visual channel, OCR each retrieved frame's PNG, and check whether at
  least one frame both (a) overlaps the gold span on the right video
  (open-interval ± ``FRAME_SAMPLE_EVERY_SEC`` tolerance, same as
  ``VisualFrameRecall@k``) AND (b) has OCR text satisfying the same
  conjunction-of-disjunctions rule as ``chunk_text_contains_evidence``.
  This is the only metric in the module that actually validates a
  visual model surfaced an answer-bearing frame. Reported as a
  STANDALONE rate, not a lift: the 4-channel text RRF retrieves chunks
  not frames, so a "without-visual" baseline is not meaningful here.
  If the rate is high, the visual channel grounds answers; if low, it
  does not. The text-side ``AnswerTermHit@k`` is the paired number for
  context — both are computed per row when evidence is provided.

  OCR backend: ``pytesseract`` against locally-installed Tesseract
  (``brew install tesseract``). PSM 3 (fully automatic page
  segmentation, default) + English language model. Documented in the
  methodology MDX. Tesseract is a deterministic tool, not an LLM, so
  the constant lives next to the primitive rather than under the
  ``model_candidates.yaml`` ADR-004 selection rule.

* ``VisualLift@k`` — for each visual_gold example, compare a 5-channel
  RRF (BM25 + dense + sparse + multivec + visual) against the 4-channel
  RRF used in bakeoff #1 (BM25 + dense + sparse + multivec). Lift is
  reported on both ``VisualChunkTR@k`` and ``AnswerTermHit@k`` so the
  caller can see whether adding the visual channel surfaces more
  span-overlapping chunks AND/OR more answer-term-bearing transcript
  chunks. Lift on either tier does not imply the visual model is
  validating against the slide.

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
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import psycopg

from retrieve import bm25, dense, multivec, rrf, sparse, visual
from retrieve.types import ChannelResult, FusedResult
from retrieve.visual import FrameResult

# OCR configuration for VisualAnswerGrounding@k. Tesseract is a tool
# (deterministic, no model selection ADR), so the constants live here
# next to the primitive. PSM 3 is the default fully-automatic page
# segmentation mode — empirically the best Tesseract behaviour on the
# 360p slide frames currently in the substrate (PSM 6 collapses
# multi-column slide layouts). ENG is sufficient for the current corpus
# (English-language conference talks). Document any changes in the
# methodology MDX so the kappa/grounding numbers stay reproducible.
_TESSERACT_LANG = "eng"
_TESSERACT_PSM = 3

# Cap the LRU cache by image_path so a single eval run never re-OCRs
# the same frame across the with-visual + without-visual stacks or the
# pooled-then-MaxSim path. 4096 is well above the 529-frame substrate
# today and gives headroom for v1 (~5x).
_OCR_CACHE_SIZE = 4096

_VISUAL_GOLD_PATH = (
    Path(__file__).resolve().parent.parent / "corpora" / "ai_engineering_v0" / "visual_gold.jsonl"
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


@dataclass(frozen=True)
class VisualEvidenceItem:
    """One curator-supplied required-term entry from a visual_gold row.

    Provenance: the terms come from the gold FRAME (e.g. axis labels,
    code identifiers visible on a slide). The metric that consumes them
    (``AnswerTermHit@k``) checks them against the TRANSCRIPT TEXT of
    retrieved chunks — it does NOT validate them against any visual
    model output, OCR, or human-confirmed frame ID. The field name
    ``visual_evidence`` describes where the curator looked; the check
    runs on text.

    The entry passes against a chunk if at least one of ``required_any``
    appears as a substring (after normalization) in the chunk's text.
    Multiple items on the same row are conjoined: every item must pass.
    """

    visual_element: str
    required_any: tuple[str, ...]
    normalize: str = "lowercase_collapse_ws"


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


def load_visual_evidence(
    path: Path = _VISUAL_GOLD_PATH,
) -> dict[str, list[VisualEvidenceItem]]:
    """Load curator-supplied ``visual_evidence`` entries keyed by example_id.

    Rows without a ``visual_evidence`` field are skipped (the new metric
    treats them as not-evaluable — they do not contribute to the
    numerator or denominator of AnswerTermHit@k). Rows with malformed
    entries raise ValueError so a typo never silently makes a row pass.
    """
    if not path.exists():
        return {}
    out: dict[str, list[VisualEvidenceItem]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        ex = json.loads(line)
        items_raw = ex.get("visual_evidence")
        if not items_raw:
            continue
        items: list[VisualEvidenceItem] = []
        for entry in items_raw:
            required_any = tuple(entry["required_any"])
            if not required_any:
                raise ValueError(
                    f"{path}: id={ex.get('id', '?')} has empty required_any "
                    "in a visual_evidence entry"
                )
            items.append(
                VisualEvidenceItem(
                    visual_element=str(entry["visual_element"]),
                    required_any=required_any,
                    normalize=str(entry.get("normalize", "lowercase_collapse_ws")),
                )
            )
        out[ex["id"]] = items
    return out


# --- Metric primitives (pure; no DB calls) ---------------------------------


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str, mode: str) -> str:
    """Normalize ``text`` according to ``mode``.

    ``lowercase_collapse_ws`` lowercases and collapses runs of whitespace
    to a single space, then strips. ``exact`` returns as-is. Unknown
    modes raise ValueError so a typo in a fixture or curator file fails
    loudly rather than silently bypassing the metric.
    """
    if mode == "exact":
        return text
    if mode == "lowercase_collapse_ws":
        return _WS_RE.sub(" ", text.lower()).strip()
    raise ValueError(f"unknown normalize mode: {mode!r}")


def chunk_text_contains_evidence(
    chunk_text: str,
    evidence: Sequence[VisualEvidenceItem],
) -> bool:
    """True iff EVERY ``VisualEvidenceItem`` in ``evidence`` has at least
    one term in ``required_any`` (normalized per the item's mode) as a
    substring of the chunk text (normalized the same way).

    Empty ``evidence`` returns True (vacuous). Callers gate the
    "no evidence on this row" case at the aggregator level — see
    ``answer_term_hit_at_k``.
    """
    for item in evidence:
        normalized_chunk = normalize_text(chunk_text, item.normalize)
        normalized_terms = [normalize_text(t, item.normalize) for t in item.required_any]
        if not any(t in normalized_chunk for t in normalized_terms if t):
            return False
    return True


def answer_term_matches_at_k(
    results: Sequence[ChannelResult | FusedResult],
    gold: VisualGoldQuery,
    evidence: Sequence[VisualEvidenceItem],
    *,
    k: int,
) -> bool:
    """True iff some top-k chunk both overlaps the gold span on the right
    video AND its TRANSCRIPT TEXT satisfies ``chunk_text_contains_evidence``.

    The check runs on chunk transcript text, not on any frame, OCR
    output, or visual judge. The metric distinguishes topically-similar
    chunks from chunks whose text actually quotes the curator's
    answer-bearing strings. Empty ``evidence`` returns False — the
    metric is undefined for rows without curator-supplied terms, so
    callers must skip them.
    """
    if not evidence:
        return False
    for r in results[:k]:
        if (
            r.video_id == gold.video_id
            and r.end_sec > gold.start_sec
            and r.start_sec < gold.end_sec
            and chunk_text_contains_evidence(r.text, evidence)
        ):
            return True
    return False


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


# --- VisualAnswerGrounding@k primitives (frame OCR, no DB) ------------------


@lru_cache(maxsize=_OCR_CACHE_SIZE)
def frame_ocr_text(image_path: str) -> str:
    """OCR a frame PNG and return the concatenated detected text.

    Pure function on the file at ``image_path``. Cached by path so a
    single eval run does not re-OCR the same frame across visual_eval's
    pooled-prefilter path and the chunked-MaxSim path. Tests inject
    their own ``ocr_fn`` to avoid touching real images.

    Errors (missing file, Tesseract not installed) are NOT swallowed —
    the metric is undefined for frames that cannot be OCR'd, so the
    aggregator catches and surfaces them rather than scoring a false
    pass / fail."""
    # Import is inline so the rest of the module doesn't pay the
    # pytesseract import cost when callers stick to the span-overlap or
    # text-side metrics. Tesseract itself is invoked by pytesseract as a
    # subprocess on each call.
    #
    # Pass the PATH, not a PIL Image. pytesseract 5.5.x raises a
    # UnicodeDecodeError on PIL-Image input on this stack (it mis-reads
    # Tesseract's stderr when round-tripping the image through a temp
    # file); handing Tesseract the path lets Leptonica load the PNG
    # directly, which is also closer to the `tesseract <file>` CLI.
    import pytesseract

    return pytesseract.image_to_string(
        image_path, lang=_TESSERACT_LANG, config=f"--psm {_TESSERACT_PSM}"
    )


OcrFn = Callable[[str], str]


def frame_ocr_matches_at_k(
    frames: Sequence[FrameResult],
    gold: VisualGoldQuery,
    evidence: Sequence[VisualEvidenceItem],
    *,
    k: int,
    ocr_fn: OcrFn,
    tolerance_sec: float = FRAME_SAMPLE_EVERY_SEC,
) -> bool:
    """True iff some top-k frame whose ``video_id == gold.video_id`` AND
    ``frame_sec`` falls inside the gold span (± ``tolerance_sec``) ALSO
    has OCR text satisfying ``chunk_text_contains_evidence``.

    Combines the existing ``frame_passes_at_k`` window with a per-frame
    OCR-text check. The OCR text is treated identically to chunk text
    by ``chunk_text_contains_evidence``: conjunction across items,
    disjunction within ``required_any``, normalization per item.

    Empty ``evidence`` returns False — the metric is undefined for rows
    without curator-supplied terms; callers gate this at the aggregator
    level (see ``visual_answer_grounding_at_k``).
    """
    if not evidence:
        return False
    lo = gold.start_sec - tolerance_sec
    hi = gold.end_sec + tolerance_sec
    for f in frames[:k]:
        if f.video_id != gold.video_id:
            continue
        if not (lo <= f.frame_sec <= hi):
            continue
        ocr_text = ocr_fn(f.image_path)
        if chunk_text_contains_evidence(ocr_text, evidence):
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
    # ``text`` is included because the answer-term metric and the
    # methodology MDX both need to inspect the surfaced chunk text (the
    # span-overlap test alone cannot distinguish a topical chunk from
    # an answer-bearing one — see ``answer_term_matches_at_k``).
    return {
        "chunk_id": r.chunk_id,
        "video_id": r.video_id,
        "start_sec": r.start_sec,
        "end_sec": r.end_sec,
        "text": r.text,
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


def answer_term_hit_at_k(
    conn: psycopg.Connection,
    examples: Sequence[VisualGoldQuery],
    evidence_by_example: dict[str, Sequence[VisualEvidenceItem]],
    *,
    k: int = 5,
    chunk_fn: ChunkFn | None = None,
) -> tuple[float, list[bool | None], list[list[dict]]]:
    """Aggregate AnswerTermHit@k over ``examples`` using the visual
    channel's chunk-level entry point.

    Text-side diagnostic: validates whether the SURFACED CHUNK'S
    TRANSCRIPT TEXT contains the curator's answer terms. Does NOT
    validate against any frame, OCR, or visual judge. See module
    docstring for the scope of the metric.

    For each example:

    * If ``evidence_by_example`` has no (or empty) entry for the example,
      that row is NOT evaluable — its slot in ``per_example_passes`` is
      ``None``, and it contributes to neither the numerator nor the
      denominator of the returned ``score``.
    * Otherwise, the row passes iff some top-k chunk both span-overlaps
      the gold AND its transcript text satisfies the curator-supplied
      evidence conjunction (see ``answer_term_matches_at_k``).

    Returns ``(score, per_example_passes, per_example_top_k_chunks)``.
    ``score`` is the mean over evaluable rows only. If no rows are
    evaluable, ``score`` is 0.0 — callers should also inspect the
    per-example list and the lift summary's ``n_evaluable`` before
    interpreting the headline number."""
    fn = chunk_fn or (lambda c, q: visual.retrieve(c, q, top_k=k))
    if not examples:
        return 0.0, [], []
    passes: list[bool | None] = []
    top_ks: list[list[dict]] = []
    n_evaluable = 0
    n_pass = 0
    for ex in examples:
        results = list(fn(conn, ex.question))
        top_ks.append([_chunk_to_dict(r) for r in results[:k]])
        evidence = evidence_by_example.get(ex.example_id) or ()
        if not evidence:
            passes.append(None)
            continue
        hit = answer_term_matches_at_k(results, ex, evidence, k=k)
        passes.append(hit)
        n_evaluable += 1
        n_pass += int(hit)
    score = (n_pass / n_evaluable) if n_evaluable else 0.0
    return score, passes, top_ks


# TODO(v4): Replace the boolean return with a structured audit payload so any
# future `vag = true` row can be defended. The 0/10 result on the 2026-05-28
# headline run does not depend on this, but any positive result from a future
# 720p re-ingest does. Minimum payload fields documented in
# `eval/reports/2026-05-28_visual_eval_v3/integrity_check.md` §B:
# evaluated_frame_ids, evaluated_image_paths, ocr_excerpts, matched_term,
# failure_reason.
def visual_answer_grounding_at_k(
    conn: psycopg.Connection,
    examples: Sequence[VisualGoldQuery],
    evidence_by_example: dict[str, Sequence[VisualEvidenceItem]],
    *,
    k: int = 5,
    tolerance_sec: float = FRAME_SAMPLE_EVERY_SEC,
    frame_fn: FrameFn | None = None,
    ocr_fn: OcrFn | None = None,
) -> tuple[float, list[bool | None], list[list[dict]]]:
    """Aggregate VisualAnswerGrounding@k over ``examples``.

    Frame-side answer-grounding: for each example with curator-supplied
    ``visual_evidence``, fetch the top-k frames from the visual channel
    (default ``retrieve.visual.retrieve_frames``), OCR each frame's PNG
    (default ``frame_ocr_text``), and pass iff at least one returned
    frame on the right video, inside the gold span ± tolerance, has OCR
    text satisfying the curator-supplied conjunction.

    For each example:

    * If ``evidence_by_example`` has no (or empty) entry for the example,
      that row is NOT evaluable — its slot in ``per_example_passes`` is
      ``None``, and it contributes to neither the numerator nor the
      denominator of the returned ``score``.
    * Otherwise, the row passes iff ``frame_ocr_matches_at_k`` is True.

    Returns ``(score, per_example_passes, per_example_top_k_frames)``.
    ``score`` is the mean over evaluable rows only. If no rows are
    evaluable, ``score`` is 0.0 — callers should also inspect the
    per-example list and the n_evaluable count before interpreting the
    headline number.

    This is the only metric in the module that validates the visual
    channel surfaced an answer-bearing frame. See module docstring for
    why it is reported as a standalone rate, not as a lift against
    text-only retrieval."""
    fn = frame_fn or (lambda c, q: visual.retrieve_frames(c, q, top_k=k))
    ocr = ocr_fn or frame_ocr_text
    if not examples:
        return 0.0, [], []
    passes: list[bool | None] = []
    top_ks: list[list[dict]] = []
    n_evaluable = 0
    n_pass = 0
    for ex in examples:
        frames = list(fn(conn, ex.question))
        top_ks.append([_frame_to_dict(f) for f in frames[:k]])
        evidence = evidence_by_example.get(ex.example_id) or ()
        if not evidence:
            passes.append(None)
            continue
        hit = frame_ocr_matches_at_k(
            frames, ex, evidence, k=k, ocr_fn=ocr, tolerance_sec=tolerance_sec
        )
        passes.append(hit)
        n_evaluable += 1
        n_pass += int(hit)
    score = (n_pass / n_evaluable) if n_evaluable else 0.0
    return score, passes, top_ks


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
    evidence_by_example: dict[str, Sequence[VisualEvidenceItem]] | None = None,
) -> dict:
    """Compare 5-channel RRF (text + visual) against 4-channel RRF
    (text only) on the same visual gold set, on TWO metric tiers:

    1. ``VisualChunkTR@k`` span-overlap — the original phase-1 metric.
       Reported as ``lift_chunk_tr_pp`` (also aliased to ``lift_pp`` for
       back-compat with the bakeoff-#1 ``summary["lift_pp"]`` contract).

    2. ``AnswerTermHit@k`` — span overlap AND curator-supplied
       answer-bearing terms must appear in the chunk TRANSCRIPT TEXT.
       Reported as ``lift_answer_term_pp`` and only computed when
       ``evidence_by_example`` provides at least one item; rows without
       evidence are skipped from BOTH numerator and denominator of the
       answer-term rates. The check runs on text, not on frames — see
       module docstring.

    Per-example dict carries the four booleans (two metrics × two stacks):
    ``with_visual`` / ``without_visual`` (span-overlap) plus
    ``answer_term_with_visual`` / ``answer_term_without_visual``. The
    answer-term booleans are ``None`` for rows without curator evidence.

    Empty ``examples`` returns zeros and empty per_example. Interpretation
    is the runner's job — small dev sets make even +/-10pp swings noisy.
    """
    if not examples:
        return {
            "with_visual_pass_rate": 0.0,
            "without_visual_pass_rate": 0.0,
            "lift_chunk_tr_pp": 0.0,
            "lift_pp": 0.0,  # back-compat alias for lift_chunk_tr_pp
            "lift_answer_term_pp": None,
            "answer_term_with_visual_rate": None,
            "answer_term_without_visual_rate": None,
            "answer_term_n_evaluable": 0,
            "per_example": {},
        }
    evidence_by_example = evidence_by_example or {}
    per_example: dict[str, dict] = {}
    with_pass = without_pass = 0
    at_with_pass = at_without_pass = 0
    n_evaluable = 0
    for ex in examples:
        with_results = _rrf_with_visual(conn, ex.question, k=k)
        without_results = _rrf_text_only(conn, ex.question, k=k)
        with_pass_ex = chunk_passes_at_k(with_results, ex, k=k)
        without_pass_ex = chunk_passes_at_k(without_results, ex, k=k)
        entry: dict = {
            "with_visual": bool(with_pass_ex),
            "without_visual": bool(without_pass_ex),
            "answer_term_with_visual": None,
            "answer_term_without_visual": None,
        }
        evidence = evidence_by_example.get(ex.example_id) or ()
        if evidence:
            at_with_ex = answer_term_matches_at_k(with_results, ex, evidence, k=k)
            at_without_ex = answer_term_matches_at_k(without_results, ex, evidence, k=k)
            entry["answer_term_with_visual"] = bool(at_with_ex)
            entry["answer_term_without_visual"] = bool(at_without_ex)
            at_with_pass += int(at_with_ex)
            at_without_pass += int(at_without_ex)
            n_evaluable += 1
        per_example[ex.example_id] = entry
        with_pass += int(with_pass_ex)
        without_pass += int(without_pass_ex)
    n = len(examples)
    with_rate = with_pass / n
    without_rate = without_pass / n
    lift_chunk_tr_pp = round((with_rate - without_rate) * 100, 2)
    out: dict = {
        "with_visual_pass_rate": with_rate,
        "without_visual_pass_rate": without_rate,
        "lift_chunk_tr_pp": lift_chunk_tr_pp,
        "lift_pp": lift_chunk_tr_pp,  # back-compat alias
        "per_example": per_example,
    }
    if n_evaluable:
        at_with_rate = at_with_pass / n_evaluable
        at_without_rate = at_without_pass / n_evaluable
        out["lift_answer_term_pp"] = round((at_with_rate - at_without_rate) * 100, 2)
        out["answer_term_with_visual_rate"] = at_with_rate
        out["answer_term_without_visual_rate"] = at_without_rate
    else:
        out["lift_answer_term_pp"] = None
        out["answer_term_with_visual_rate"] = None
        out["answer_term_without_visual_rate"] = None
    out["answer_term_n_evaluable"] = n_evaluable
    return out


def measure_visual(
    conn: psycopg.Connection,
    examples: Sequence[VisualGoldQuery],
    *,
    k: int = 5,
    tolerance_sec: float = FRAME_SAMPLE_EVERY_SEC,
    include_lift: bool = True,
    evidence_by_example: dict[str, Sequence[VisualEvidenceItem]] | None = None,
    ocr_fn: OcrFn | None = None,
) -> tuple[dict, dict[str, dict]]:
    """Run all visual metrics and return ``(summary, per_example_detail)``
    in the same shape ``eval.runners.measure_embeddings.measure_all_channels``
    uses, so the runner script can reuse the bakeoff #1 transactional
    write pattern.

    ``include_lift`` runs an extra 2-channel comparison per example
    (5-channel vs 4-channel RRF). It is optional because lift requires
    BOTH the visual stack AND the text stack to be populated for the
    same examples — and on tiny dev sets the result is mostly anecdotal.

    ``evidence_by_example`` enables BOTH curator-evidence-bearing
    metrics:

    * ``AnswerTermHit@k`` (text side) — span overlap AND required terms
      in chunk transcript text.
    * ``VisualAnswerGrounding@k`` (frame side) — top-k visual-channel
      frames + OCR + required terms in OCR text.

    Rows whose example_id maps to at least one ``VisualEvidenceItem``
    are scored against both gates; rows without evidence are skipped
    from BOTH headline rates (numerator and denominator unaffected).
    The lift block gains an answer-term tier when evidence is provided
    (text-side only — there is no meaningful "without visual" baseline
    for frame OCR). None is the back-compat default.

    ``ocr_fn`` is injectable for tests so the real Tesseract subprocess
    is not invoked on every fast test. Defaults to ``frame_ocr_text``."""
    evidence_by_example = evidence_by_example or {}
    frame_score, frame_passes, frame_topks = visual_frame_recall_at_k(
        conn, examples, k=k, tolerance_sec=tolerance_sec
    )
    chunk_score, chunk_passes, chunk_topks = visual_chunk_tr_at_k(conn, examples, k=k)
    answer_term_passes: list[bool | None] = [None] * len(examples)
    answer_term_score = 0.0
    n_evaluable = 0
    grounding_passes: list[bool | None] = [None] * len(examples)
    grounding_score = 0.0
    n_grounding_evaluable = 0
    if evidence_by_example:
        answer_term_score, answer_term_passes, _at_topks = answer_term_hit_at_k(
            conn, examples, evidence_by_example, k=k
        )
        n_evaluable = sum(1 for p in answer_term_passes if p is not None)
        grounding_score, grounding_passes, _g_topks = visual_answer_grounding_at_k(
            conn,
            examples,
            evidence_by_example,
            k=k,
            tolerance_sec=tolerance_sec,
            ocr_fn=ocr_fn,
        )
        n_grounding_evaluable = sum(1 for p in grounding_passes if p is not None)

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
                "answer_term_hit": at,  # may be None (row not evaluable)
                "visual_answer_grounding": vg,  # may be None (row not evaluable)
            }
            for ex, fp, cp, at, vg in zip(
                examples,
                frame_passes,
                chunk_passes,
                answer_term_passes,
                grounding_passes,
                strict=True,
            )
        },
    }
    if evidence_by_example:
        summary["answer_term_hit_at_k"] = answer_term_score
        summary["answer_term_hit_n_evaluable"] = n_evaluable
        summary["answer_term_hit_n_skipped"] = len(examples) - n_evaluable
        summary["visual_answer_grounding_at_k"] = grounding_score
        summary["visual_answer_grounding_n_evaluable"] = n_grounding_evaluable
        summary["visual_answer_grounding_n_skipped"] = len(examples) - n_grounding_evaluable

    per_example_detail: dict[str, dict] = {
        ex.example_id: {
            "gold_span": {
                "video_id": ex.video_id,
                "start_sec": ex.start_sec,
                "end_sec": ex.end_sec,
            },
            "frame_pass_at_k": bool(fp),
            "chunk_pass_at_k": bool(cp),
            "answer_term_hit_at_k": at,
            "visual_answer_grounding_at_k": vg,
            "top_k_frames": ftk,
            "top_k_chunks": ctk,
        }
        for ex, fp, cp, at, vg, ftk, ctk in zip(
            examples,
            frame_passes,
            chunk_passes,
            answer_term_passes,
            grounding_passes,
            frame_topks,
            chunk_topks,
            strict=True,
        )
    }

    if include_lift and examples:
        lift = visual_lift_at_k(conn, examples, k=k, evidence_by_example=evidence_by_example)
        summary["visual_lift"] = {
            "with_visual_pass_rate": lift["with_visual_pass_rate"],
            "without_visual_pass_rate": lift["without_visual_pass_rate"],
            "lift_chunk_tr_pp": lift["lift_chunk_tr_pp"],
            "lift_pp": lift["lift_pp"],  # back-compat alias
            "lift_answer_term_pp": lift["lift_answer_term_pp"],
            "answer_term_with_visual_rate": lift["answer_term_with_visual_rate"],
            "answer_term_without_visual_rate": lift["answer_term_without_visual_rate"],
            "answer_term_n_evaluable": lift["answer_term_n_evaluable"],
        }
        for ex_id, p in lift["per_example"].items():
            per_example_detail[ex_id]["lift"] = p

    return summary, per_example_detail
