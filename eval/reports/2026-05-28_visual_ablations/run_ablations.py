"""Visual retrieval ranking ablations — prefilter_k sweep + MaxSim demotion diagnosis.

WARNING — heavy ColQwen load. Do NOT run concurrently with any other ML job
on this machine. ColQwen 2.5 takes ~7-8 GB on Mac MPS unified memory; a
parallel load alongside another eval will crash the IDE on 24 GB machines.

Side-channel report; does NOT write to ``eval_runs`` / ``eval_results``.
Does NOT modify any production code in ``retrieve/``.

Builds on the 2026-05-28 visual diagnostics report. The diagnostics showed
two failure modes on the 10 visual-required rows:

  * 5/10 lose at stage 1 — gold-window frame absent from top-200 pooled HNSW
    prefilter (cosine-blind miss).
  * 4/10 lose at stage 3 — gold-overlapping chunk surfaced post-join but
    MaxSim demoted it (ranks 93/94/111 of ~110, near-last).
  * 1/10 passes at @5.

This script ablates the two knobs that could rescue these failures:

Ablation 1 — prefilter_k width
    For each example, replicate the production 3-stage pipeline at
    prefilter_k ∈ {200, 500, 1000}. Note that K=1000 is capped by the
    substrate (529 frames with pooled_embedding); we report the actual
    candidate count returned.

    Per (example, K) row:
      * frame_rank_in_prefilter — rank of first gold-window frame in the
        pooled prefilter (None if absent at this K).
      * chunk_tr_rank — rank of first gold-overlapping chunk in the final
        MaxSim-scored output (None if absent).
      * stage_2_gold_overlap_count — how many of the post-join chunks
        overlap the gold span.
      * stage_3_scored_chunks — denominator after MaxSim.

Ablation 2 — MaxSim demotion deep-dive
    For the stage-3-failure rows (frame found in prefilter at K=200, but
    MaxSim demoted the gold-overlapping chunk to near-last), expose:

      * Top-5 chunks MaxSim returned at K=200, with their scores and the
        first 200 chars of transcript text.
      * The gold-overlapping chunk's MaxSim score and rank, plus its
        transcript text.
      * Score gap = top-5 cutoff score − gold chunk score (positive means
        gold is below the cutoff by that margin in MaxSim units).
      * Hypothesis check fields — to test "BG-heavy / whole-frame visual
        similarity" vs "text-on-slide": for each of the top-5, record the
        gold span video_id (lets the reader eyeball whether the top-5 are
        from the same video as the gold or from elsewhere, and whether
        their text is topically related to the question).

Per-example failure classification:
    (i)   gold frame absent from prefilter at ALL K values — cosine-blind miss
    (ii)  gold frame present at higher K but MaxSim still demotes — MaxSim
          degeneracy
    (iii) gold frame + chunk reach top-K at some K — passes
    (iv)  something else (e.g. frame present, post-join produces no overlap)

Outputs:
    ablation_data.jsonl   — one row per (example, K) plus stage_3_deepdive rows
    report.md             — human-readable narrative summary

Caveat: frames are currently 360p PNGs (ingester drift flagged in the
band check). Agent B is investigating. Numbers below are on the current
360p substrate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from db.conn import dsn
from embed.colqwen import encode_text_query, encode_text_query_patches
from eval.runners.measure_visual import FRAME_SAMPLE_EVERY_SEC, load_visual_gold

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parent.parent.parent
VISUAL_GOLD_PATH = PROJECT_ROOT / "eval" / "corpora" / "ai_engineering_v0" / "visual_gold.jsonl"

PREFILTER_KS: tuple[int, ...] = (200, 500, 1000)
FINAL_TOP_K = 200  # final MaxSim output cap — wide so we can find the gold's rank
PASS_AT_KS: tuple[int, ...] = (5, 10, 20, 50)
TOLERANCE_SEC = FRAME_SAMPLE_EVERY_SEC  # 10.0


def load_visual_required_ids() -> set[str]:
    ids: set[str] = set()
    for raw in VISUAL_GOLD_PATH.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        ex = json.loads(line)
        tags = ex.get("tags") or []
        if "visual-required" in tags:
            ids.add(ex["id"])
    return ids


def load_question_lookup() -> dict[str, str]:
    """Map example_id → question text. Used in report.md narrative."""
    out: dict[str, str] = {}
    for raw in VISUAL_GOLD_PATH.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        ex = json.loads(line)
        out[ex["id"]] = ex.get("question", "")
    return out


def frame_in_gold_window(frame_video_id: str, frame_sec: float, gold) -> bool:
    return (
        frame_video_id == gold.video_id
        and gold.start_sec - TOLERANCE_SEC <= frame_sec <= gold.end_sec + TOLERANCE_SEC
    )


def chunk_overlaps_gold(chunk_video_id: str, start_sec: float, end_sec: float, gold) -> bool:
    return chunk_video_id == gold.video_id and end_sec > gold.start_sec and start_sec < gold.end_sec


@dataclass
class StageRun:
    """Result of running the 3-stage pipeline at a given prefilter_k."""

    prefilter_k_requested: int
    prefilter_k_returned: int
    gold_frame_rank_in_prefilter: int | None
    stage2_candidate_chunk_count: int
    stage2_gold_overlapping_chunk_count: int
    stage3_scored_chunk_count: int
    gold_chunk_rank_after_maxsim: int | None
    # Detailed score table for the top-N + gold (for stage-3 deepdive).
    top5_chunks: list[dict]
    gold_chunk_detail: dict | None


def run_pipeline_at_k(
    conn: psycopg.Connection,
    pooled_q: np.ndarray,
    qpatches: np.ndarray,
    gold,
    prefilter_k: int,
) -> StageRun:
    """Replicate retrieve.visual.retrieve at the given prefilter_k, with
    the gold-rank instrumented at each stage. Returns the rank of the
    gold-overlapping chunk in the final MaxSim ranking (or None) plus the
    top-5 chunks for stage-3 deepdive."""

    # Stage (a): pooled HNSW prefilter.
    pre_rows = conn.execute(
        """
        SELECT frame_id, video_id, frame_sec
        FROM frames
        WHERE pooled_embedding IS NOT NULL
        ORDER BY pooled_embedding <=> %s
        LIMIT %s
        """,
        (pooled_q, prefilter_k),
    ).fetchall()

    prefilter_returned = len(pre_rows)
    gold_frame_rank: int | None = None
    for i, (_fid, video_id, frame_sec) in enumerate(pre_rows, start=1):
        if frame_in_gold_window(video_id, float(frame_sec), gold):
            gold_frame_rank = i
            break

    if not pre_rows:
        return StageRun(
            prefilter_k_requested=prefilter_k,
            prefilter_k_returned=prefilter_returned,
            gold_frame_rank_in_prefilter=None,
            stage2_candidate_chunk_count=0,
            stage2_gold_overlapping_chunk_count=0,
            stage3_scored_chunk_count=0,
            gold_chunk_rank_after_maxsim=None,
            top5_chunks=[],
            gold_chunk_detail=None,
        )

    candidate_frame_ids = [r[0] for r in pre_rows]

    # Stage (b): frame → chunk join.
    map_rows = conn.execute(
        """
        WITH cand_frames AS (
            SELECT frame_id, video_id, frame_sec
            FROM frames
            WHERE frame_id = ANY(%s)
        )
        SELECT c.chunk_id, c.video_id, c.start_sec, c.end_sec, c.text, cf.frame_id
        FROM cand_frames cf
        JOIN chunks c
          ON c.video_id = cf.video_id
         AND cf.frame_sec >= c.start_sec
         AND cf.frame_sec <  c.end_sec
        """,
        (candidate_frame_ids,),
    ).fetchall()

    if not map_rows:
        return StageRun(
            prefilter_k_requested=prefilter_k,
            prefilter_k_returned=prefilter_returned,
            gold_frame_rank_in_prefilter=gold_frame_rank,
            stage2_candidate_chunk_count=0,
            stage2_gold_overlapping_chunk_count=0,
            stage3_scored_chunk_count=0,
            gold_chunk_rank_after_maxsim=None,
            top5_chunks=[],
            gold_chunk_detail=None,
        )

    chunk_frames: dict[int, list[int]] = defaultdict(list)
    chunk_meta: dict[int, tuple[str, float, float, str]] = {}
    for chunk_id, video_id, start_sec, end_sec, text, frame_id in map_rows:
        chunk_frames[chunk_id].append(frame_id)
        chunk_meta.setdefault(chunk_id, (video_id, float(start_sec), float(end_sec), text))

    stage2_overlap_count = sum(
        1
        for cid, meta in chunk_meta.items()
        if chunk_overlaps_gold(meta[0], meta[1], meta[2], gold)
    )

    # Stage (c): MaxSim refine — replicates retrieve.visual.retrieve exactly.
    all_candidate_frame_ids = sorted({fid for fids in chunk_frames.values() for fid in fids})
    patch_rows = conn.execute(
        """
        SELECT frame_id, patch_index, embedding
        FROM frame_patches
        WHERE frame_id = ANY(%s)
        ORDER BY frame_id, patch_index
        """,
        (all_candidate_frame_ids,),
    ).fetchall()
    if not patch_rows:
        return StageRun(
            prefilter_k_requested=prefilter_k,
            prefilter_k_returned=prefilter_returned,
            gold_frame_rank_in_prefilter=gold_frame_rank,
            stage2_candidate_chunk_count=len(chunk_meta),
            stage2_gold_overlapping_chunk_count=stage2_overlap_count,
            stage3_scored_chunk_count=0,
            gold_chunk_rank_after_maxsim=None,
            top5_chunks=[],
            gold_chunk_detail=None,
        )

    frame_patches: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame_id, _patch_idx, embedding in patch_rows:
        frame_patches[frame_id].append(embedding)

    if qpatches.size == 0:
        return StageRun(
            prefilter_k_requested=prefilter_k,
            prefilter_k_returned=prefilter_returned,
            gold_frame_rank_in_prefilter=gold_frame_rank,
            stage2_candidate_chunk_count=len(chunk_meta),
            stage2_gold_overlapping_chunk_count=stage2_overlap_count,
            stage3_scored_chunk_count=0,
            gold_chunk_rank_after_maxsim=None,
            top5_chunks=[],
            gold_chunk_detail=None,
        )

    scored: list[tuple[int, float]] = []
    for chunk_id, frame_ids in chunk_frames.items():
        patches: list[np.ndarray] = []
        for fid in frame_ids:
            patches.extend(frame_patches.get(fid, ()))
        if not patches:
            continue
        cpatches = np.stack(patches).astype(np.float32)
        sims = qpatches @ cpatches.T
        maxsim = float(sims.max(axis=1).sum())
        scored.append((chunk_id, maxsim))

    scored.sort(key=lambda x: (-x[1], x[0]))

    gold_rank: int | None = None
    gold_detail: dict | None = None
    for i, (cid, score) in enumerate(scored, start=1):
        meta = chunk_meta[cid]
        if chunk_overlaps_gold(meta[0], meta[1], meta[2], gold):
            gold_rank = i
            gold_detail = {
                "chunk_id": cid,
                "video_id": meta[0],
                "start_sec": meta[1],
                "end_sec": meta[2],
                "text_preview": meta[3][:200],
                "maxsim_score": score,
                "rank": i,
                "frames_in_prefilter": len(chunk_frames[cid]),
            }
            break

    top5: list[dict] = []
    for i, (cid, score) in enumerate(scored[:5], start=1):
        meta = chunk_meta[cid]
        top5.append(
            {
                "rank": i,
                "chunk_id": cid,
                "video_id": meta[0],
                "start_sec": meta[1],
                "end_sec": meta[2],
                "text_preview": meta[3][:200],
                "maxsim_score": score,
                "frames_in_prefilter": len(chunk_frames[cid]),
                "same_video_as_gold": meta[0] == gold.video_id,
                "overlaps_gold": chunk_overlaps_gold(meta[0], meta[1], meta[2], gold),
            }
        )

    return StageRun(
        prefilter_k_requested=prefilter_k,
        prefilter_k_returned=prefilter_returned,
        gold_frame_rank_in_prefilter=gold_frame_rank,
        stage2_candidate_chunk_count=len(chunk_meta),
        stage2_gold_overlapping_chunk_count=stage2_overlap_count,
        stage3_scored_chunk_count=len(scored),
        gold_chunk_rank_after_maxsim=gold_rank,
        top5_chunks=top5,
        gold_chunk_detail=gold_detail,
    )


def classify_failure_mode(per_k_runs: dict[int, StageRun]) -> str:
    """Per-example category across the K sweep.

    (i)   cosine_blind   — gold frame absent from prefilter at EVERY K tested
    (ii)  maxsim_demote  — gold frame appears at some K (any) but the chunk
                            is demoted out of top-5 by MaxSim at every K
    (iii) passes         — gold chunk reaches top-5 in the final MaxSim
                            ranking at at least one K
    (iv)  other          — frame present, but post-join produces no
                            overlap, or other structural anomaly
    """
    # iii — at least one K puts the gold chunk in the top-5.
    for run in per_k_runs.values():
        if run.gold_chunk_rank_after_maxsim is not None and run.gold_chunk_rank_after_maxsim <= 5:
            return "iii_passes"

    # i — frame absent at all K values.
    if all(r.gold_frame_rank_in_prefilter is None for r in per_k_runs.values()):
        return "i_cosine_blind"

    # ii — frame appears at some K AND the post-join produced overlapping
    # chunks at that K AND MaxSim still demoted them past top-5.
    frame_found_anywhere = any(
        r.gold_frame_rank_in_prefilter is not None for r in per_k_runs.values()
    )
    overlap_chunk_anywhere = any(
        r.stage2_gold_overlapping_chunk_count > 0 for r in per_k_runs.values()
    )
    demoted_everywhere = all(
        r.gold_chunk_rank_after_maxsim is None or r.gold_chunk_rank_after_maxsim > 5
        for r in per_k_runs.values()
    )
    if frame_found_anywhere and overlap_chunk_anywhere and demoted_everywhere:
        return "ii_maxsim_demote"

    return "iv_other"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_report(
    path: Path,
    per_example: dict[str, dict],
    question_lookup: dict[str, str],
) -> None:
    lines: list[str] = []
    lines.append("# Visual Retrieval Ranking Ablations — 2026-05-28")
    lines.append("")
    lines.append(
        "Side-channel report. n = 10 visual-required examples "
        "(tagged `visual-required` in `eval/corpora/ai_engineering_v0/visual_gold.jsonl`)."
    )
    lines.append("")
    lines.append(
        "Two ablations, both diagnostic, no production code modified, "
        "no writes to `eval_runs` / `eval_results`."
    )
    lines.append("")
    lines.append(
        "Substrate: 529 frames with pooled_embedding, 163,990 frame_patches, "
        "212 chunks. Frames are **360p PNGs** (ingester drift flagged in the "
        "ColQwen band check; Agent B is investigating). Numbers below are on "
        "the current 360p substrate."
    )
    lines.append("")
    lines.append(
        "Pipeline replicated stage-for-stage from `retrieve.visual.retrieve`. "
        f"Final top-k cap = `{FINAL_TOP_K}` (wide so we can locate the gold-overlapping "
        "chunk's true rank in the MaxSim ordering)."
    )
    lines.append("")

    # --- Ablation 1 summary table ---
    lines.append("## Ablation 1 — Prefilter K width")
    lines.append("")
    lines.append(
        "Per example, run the 3-stage pipeline at `prefilter_k ∈ {200, 500, 1000}`. "
        "Note: 1000 is capped by the substrate (529 frames have a non-null pooled "
        "embedding); the `returned` column shows the actual candidate count."
    )
    lines.append("")
    lines.append(
        "Each cell shows `frame_rank_in_prefilter / chunk_tr_rank_after_maxsim`. "
        "`-` means the gold-window frame was absent from the prefilter at that K; "
        "`(no overlap)` means stage 2 produced no gold-overlapping chunks; "
        "`abs` means the gold-overlapping chunk reached stage 3 but fell out of the "
        f"final top-{FINAL_TOP_K}."
    )
    lines.append("")
    lines.append(
        "| example_id | K=200 | K=500 | K=1000 (cap=529) | category |"
    )
    lines.append("|---|---|---|---|---|")

    def cell(run: StageRun) -> str:
        if run.gold_frame_rank_in_prefilter is None:
            return f"`-` / `-` ({run.prefilter_k_returned})"
        if run.stage2_gold_overlapping_chunk_count == 0:
            return (
                f"`{run.gold_frame_rank_in_prefilter}` / `(no overlap)` "
                f"({run.prefilter_k_returned})"
            )
        rank_str = (
            f"`{run.gold_chunk_rank_after_maxsim}`"
            if run.gold_chunk_rank_after_maxsim is not None
            else "`abs`"
        )
        return f"`{run.gold_frame_rank_in_prefilter}` / {rank_str} ({run.prefilter_k_returned})"

    for ex_id in sorted(per_example.keys()):
        rec = per_example[ex_id]
        runs: dict[int, StageRun] = rec["runs"]
        cat = rec["category"]
        row = (
            f"| `{ex_id}` "
            f"| {cell(runs[200])} "
            f"| {cell(runs[500])} "
            f"| {cell(runs[1000])} "
            f"| {cat} |"
        )
        lines.append(row)
    lines.append("")

    # --- Aggregate K-rescue counts ---
    rescued_at_500 = 0
    rescued_at_1000 = 0
    pass_at_200 = 0
    for rec in per_example.values():
        runs = rec["runs"]

        def passed_at_5(r: StageRun) -> bool:
            return (
                r.gold_chunk_rank_after_maxsim is not None
                and r.gold_chunk_rank_after_maxsim <= 5
            )

        p200 = passed_at_5(runs[200])
        p500 = passed_at_5(runs[500])
        p1000 = passed_at_5(runs[1000])
        if p200:
            pass_at_200 += 1
        if p500 and not p200:
            rescued_at_500 += 1
        if p1000 and not p500 and not p200:
            rescued_at_1000 += 1

    lines.append("### Aggregate K-rescue")
    lines.append("")
    lines.append(f"- `pass@5` at K=200: **{pass_at_200} / 10**")
    lines.append(
        f"- additional rescues at K=500: **{rescued_at_500}** "
        "(examples that pass@5 only when K is widened to 500)"
    )
    lines.append(
        f"- additional rescues at K=1000 (cap=529): **{rescued_at_1000}** "
        "(examples that pass@5 only when K is widened past 500)"
    )
    lines.append("")

    # --- Ablation 2: MaxSim deepdive ---
    lines.append("## Ablation 2 — MaxSim demotion diagnosis")
    lines.append("")
    lines.append(
        "For rows where the gold frame reached the prefilter (at any K) but MaxSim "
        "still demoted the gold-overlapping chunk past top-5, expose the top-5 "
        "chunks MaxSim returned vs the gold chunk."
    )
    lines.append("")
    lines.append(
        "The hypothesis under test: MaxSim is preferring whole-frame visual "
        "similarity (e.g. speaker-on-stage frames, similar slide template) over "
        "actual slide-text relevance. We check this by inspecting whether the "
        "top-5 chunks come from the same video as the gold and whether their "
        "transcript text is topically related to the question."
    )
    lines.append("")
    for ex_id in sorted(per_example.keys()):
        rec = per_example[ex_id]
        if rec["category"] != "ii_maxsim_demote":
            continue
        runs = rec["runs"]
        # Use the K=200 run for the deepdive — matches production default.
        run = runs[200]
        if run.gold_chunk_rank_after_maxsim is None:
            # gold chunk fell out of the top-FINAL_TOP_K at K=200; use the widest K
            # where the gold did surface so we can compare scores.
            for k in (500, 1000):
                if runs[k].gold_chunk_rank_after_maxsim is not None:
                    run = runs[k]
                    break
        gold = run.gold_chunk_detail
        top5 = run.top5_chunks
        question = question_lookup.get(ex_id, "")
        lines.append(f"### `{ex_id}`  (deepdive at prefilter_k={run.prefilter_k_requested})")
        lines.append("")
        lines.append(f"Question: {question}")
        lines.append("")
        if gold is None:
            lines.append(
                "Gold-overlapping chunk did not surface in any tested K. "
                "Classifying under category (iv) on this row would be more "
                "accurate; check raw data."
            )
            lines.append("")
            continue
        cutoff = top5[-1]["maxsim_score"] if top5 else float("nan")
        gap = cutoff - gold["maxsim_score"]
        lines.append(
            f"Gold chunk: `chunk_id={gold['chunk_id']}` "
            f"`[{gold['start_sec']:.0f}, {gold['end_sec']:.0f}]` sec, "
            f"MaxSim = `{gold['maxsim_score']:.3f}`, "
            f"rank = `{gold['rank']}` / `{run.stage3_scored_chunk_count}` chunks. "
            f"Gap to top-5 cutoff (cutoff = `{cutoff:.3f}`): "
            f"**`{gap:+.3f}`** MaxSim units."
        )
        lines.append("")
        lines.append(f"Gold text preview: _{gold['text_preview']!r}_")
        lines.append("")
        lines.append(
            "| rank | chunk_id | video_id | span (sec) | MaxSim | same_video | "
            "overlaps_gold | text preview |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in top5:
            lines.append(
                f"| {c['rank']} | `{c['chunk_id']}` | `{c['video_id']}` "
                f"| [{c['start_sec']:.0f}, {c['end_sec']:.0f}] "
                f"| {c['maxsim_score']:.3f} "
                f"| {c['same_video_as_gold']} "
                f"| {c['overlaps_gold']} "
                f"| _{c['text_preview']!r}_ |"
            )
        lines.append("")

    # --- Per-example classification roll-up ---
    lines.append("## Per-example failure classification")
    lines.append("")
    lines.append("Categories:")
    lines.append("")
    lines.append(
        "- **(i) cosine_blind** — gold frame absent from prefilter at ALL K values."
    )
    lines.append(
        "- **(ii) maxsim_demote** — gold frame present at some K, but MaxSim "
        "demotes the gold-overlapping chunk past top-5 at every K."
    )
    lines.append(
        "- **(iii) passes** — gold-overlapping chunk reaches top-5 in the final "
        "MaxSim ranking at at least one K."
    )
    lines.append(
        "- **(iv) other** — frame present at some K, but post-join produces no "
        "gold-overlapping chunk, or other structural anomaly."
    )
    lines.append("")
    counts: dict[str, int] = defaultdict(int)
    for rec in per_example.values():
        counts[rec["category"]] += 1
    lines.append("| category | n |")
    lines.append("|---|---|")
    for c in ("i_cosine_blind", "ii_maxsim_demote", "iii_passes", "iv_other"):
        lines.append(f"| {c} | {counts.get(c, 0)} |")
    lines.append("")
    lines.append("| example_id | category | gold_frame_at_K=200 | gold_frame_at_K=500 "
                 "| gold_frame_at_K=1000 | chunk_rank_at_K=200 |")
    lines.append("|---|---|---|---|---|---|")
    for ex_id in sorted(per_example.keys()):
        rec = per_example[ex_id]
        runs = rec["runs"]

        def fmt_frame(r: StageRun) -> str:
            return (
                f"`{r.gold_frame_rank_in_prefilter}`/`{r.prefilter_k_returned}`"
                if r.gold_frame_rank_in_prefilter is not None
                else f"absent (`0`/`{r.prefilter_k_returned}`)"
            )

        chunk_rank_200 = runs[200].gold_chunk_rank_after_maxsim
        lines.append(
            f"| `{ex_id}` | {rec['category']} "
            f"| {fmt_frame(runs[200])} | {fmt_frame(runs[500])} "
            f"| {fmt_frame(runs[1000])} "
            f"| {chunk_rank_200 if chunk_rank_200 is not None else 'absent'} |"
        )
    lines.append("")

    # --- Dominant failure-mode statement ---
    lines.append("## Dominant failure mode")
    lines.append("")
    n = len(per_example)
    i_n = counts.get("i_cosine_blind", 0)
    ii_n = counts.get("ii_maxsim_demote", 0)
    iii_n = counts.get("iii_passes", 0)
    iv_n = counts.get("iv_other", 0)
    lines.append(
        f"Counts (n={n}): "
        f"(i) cosine_blind = **{i_n}**, "
        f"(ii) maxsim_demote = **{ii_n}**, "
        f"(iii) passes = **{iii_n}**, "
        f"(iv) other = **{iv_n}**."
    )
    lines.append("")
    if rescued_at_500 == 0 and rescued_at_1000 == 0:
        lines.append(
            "**Widening `prefilter_k` does not rescue any examples** on the "
            "current 360p substrate. The cosine-blind misses are not 'right "
            "frame just beyond rank 200' — the pooled cosine signal does not "
            "rank the gold frame anywhere in the top 529 candidates either. "
            "This points at the embedding side (frame resolution / quality), "
            "not the ranking width."
        )
    else:
        lines.append(
            f"**Widening `prefilter_k`** rescues {rescued_at_500} additional "
            f"examples at K=500 and {rescued_at_1000} more at K=1000."
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    visual_required_ids = load_visual_required_ids()
    examples_all = load_visual_gold()
    examples = [e for e in examples_all if e.example_id in visual_required_ids]
    if len(examples) != 10:
        raise RuntimeError(
            f"expected 10 visual-required examples; got {len(examples)} "
            f"(tagged ids: {sorted(visual_required_ids)})"
        )

    question_lookup = load_question_lookup()
    per_example: dict[str, dict] = {}
    flat_rows: list[dict] = []

    with psycopg.connect(dsn()) as conn:
        register_vector(conn)
        for i, ex in enumerate(examples, start=1):
            print(f"[{i}/{len(examples)}] {ex.example_id}")
            pooled_q = encode_text_query(ex.question)
            qpatches = encode_text_query_patches(ex.question).astype(np.float32)

            per_k: dict[int, StageRun] = {}
            for k_val in PREFILTER_KS:
                run = run_pipeline_at_k(conn, pooled_q, qpatches, ex, k_val)
                per_k[k_val] = run
                gold_rank = run.gold_chunk_rank_after_maxsim
                pass_at = {
                    f"chunk_tr_at_{k}": gold_rank is not None and gold_rank <= k
                    for k in PASS_AT_KS
                }
                flat_rows.append(
                    {
                        "example_id": ex.example_id,
                        "video_id": ex.video_id,
                        "gold_start_sec": ex.start_sec,
                        "gold_end_sec": ex.end_sec,
                        "prefilter_k_requested": run.prefilter_k_requested,
                        "prefilter_k_returned": run.prefilter_k_returned,
                        "gold_frame_rank_in_prefilter": run.gold_frame_rank_in_prefilter,
                        "stage2_candidate_chunk_count": run.stage2_candidate_chunk_count,
                        "stage2_gold_overlapping_chunk_count": (
                            run.stage2_gold_overlapping_chunk_count
                        ),
                        "stage3_scored_chunk_count": run.stage3_scored_chunk_count,
                        "gold_chunk_rank_after_maxsim": gold_rank,
                        **pass_at,
                        "top5_chunks": run.top5_chunks,
                        "gold_chunk_detail": run.gold_chunk_detail,
                    }
                )
            category = classify_failure_mode(per_k)
            per_example[ex.example_id] = {"runs": per_k, "category": category}

    write_jsonl(REPORT_DIR / "ablation_data.jsonl", flat_rows)
    write_report(REPORT_DIR / "report.md", per_example, question_lookup)
    print(f"wrote {REPORT_DIR / 'ablation_data.jsonl'}")
    print(f"wrote {REPORT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
