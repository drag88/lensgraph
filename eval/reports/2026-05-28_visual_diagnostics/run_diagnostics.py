"""Visual retrieval diagnostics — per-example rank curves + stage attribution.

Side-channel report; does NOT write to ``eval_runs`` / ``eval_results``.

For each of the 10 visual-required examples in ``visual_gold.jsonl`` (tagged
``visual-required``), we compute:

* ``frame_recall_rank`` — rank of the first gold-window frame returned by
  ``retrieve.visual.retrieve_frames(top_k=200)``.
* ``chunk_tr_rank`` — rank of the first gold-overlapping chunk returned by
  ``retrieve.visual.retrieve(top_k=200, prefilter_k=200)``.
* Pass booleans at k ∈ {5, 10, 20, 50} for both.
* Per-stage attribution:
    - stage 1 (pooled HNSW prefilter): does the gold-window frame appear in
      the top-200 pooled candidates? If yes, at what rank?
    - stage 2 (frame→chunk SQL join): given the stage-1 prefilter candidates,
      does the SQL join return any chunk overlapping the gold span?
    - stage 3 (MaxSim refine): among the stage-2 candidate chunks, where does
      the gold-overlapping chunk rank after MaxSim?
    - stage 4 (RRF): informational only — pulled from the 2026-05-27 eval's
      per_example.jsonl (``with_visual`` vs ``without_visual``).

We re-use the encoder + retrieval primitives directly so this script measures
the production pipeline as-is. No changes to ``retrieve/visual.py``.

Outputs:
    rank_curves.jsonl   — one row per visual-required example
    report.md           — human-readable summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from db.conn import dsn
from embed.colqwen import encode_text_query, encode_text_query_patches
from eval.runners.measure_visual import FRAME_SAMPLE_EVERY_SEC, load_visual_gold
from retrieve.visual import retrieve, retrieve_frames

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parent.parent.parent
VISUAL_GOLD_PATH = PROJECT_ROOT / "eval" / "corpora" / "ai_engineering_v0" / "visual_gold.jsonl"
PRIOR_PER_EXAMPLE_PATH = (
    PROJECT_ROOT / "eval" / "reports" / "2026-05-27_visual_eval" / "per_example.jsonl"
)

DEEP_K = 200
KS = (5, 10, 20, 50)
TOLERANCE_SEC = FRAME_SAMPLE_EVERY_SEC  # 10.0


def load_visual_required_ids() -> set[str]:
    """Return the set of example IDs tagged ``visual-required``."""
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


def load_prior_rrf_per_example() -> dict[str, dict[str, bool]]:
    """Pull ``with_visual`` / ``without_visual`` pass flags from the
    2026-05-27 eval per_example.jsonl. Informational stage-4 context."""
    out: dict[str, dict[str, bool]] = {}
    if not PRIOR_PER_EXAMPLE_PATH.exists():
        return out
    for raw in PRIOR_PER_EXAMPLE_PATH.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        lift = (row.get("metrics") or {}).get("lift") or {}
        if not lift:
            continue
        out[row["example_id"]] = {
            "with_visual": bool(lift.get("with_visual")),
            "without_visual": bool(lift.get("without_visual")),
        }
    return out


def frame_in_gold_window(frame_video_id: str, frame_sec: float, gold) -> bool:
    return (
        frame_video_id == gold.video_id
        and gold.start_sec - TOLERANCE_SEC <= frame_sec <= gold.end_sec + TOLERANCE_SEC
    )


def chunk_overlaps_gold(chunk_video_id: str, start_sec: float, end_sec: float, gold) -> bool:
    return (
        chunk_video_id == gold.video_id
        and end_sec > gold.start_sec
        and start_sec < gold.end_sec
    )


def measure_stage_1(conn: psycopg.Connection, query: str, gold) -> dict:
    """Pooled HNSW prefilter — top-DEEP_K frames by cosine. Returns rank of
    first gold-window frame (or None) plus all candidate frame rows
    (frame_id, video_id, frame_sec) for stage 2."""
    qvec = encode_text_query(query)
    register_vector(conn)
    rows = conn.execute(
        """
        SELECT frame_id, video_id, frame_sec,
               1 - (pooled_embedding <=> %s) AS score
        FROM frames
        WHERE pooled_embedding IS NOT NULL
        ORDER BY pooled_embedding <=> %s
        LIMIT %s
        """,
        (qvec, qvec, DEEP_K),
    ).fetchall()
    rank: int | None = None
    for i, (_frame_id, video_id, frame_sec, _score) in enumerate(rows, start=1):
        if frame_in_gold_window(video_id, float(frame_sec), gold):
            rank = i
            break
    return {
        "stage1_pooled_frame_count": len(rows),
        "stage1_gold_frame_rank": rank,
        "stage1_candidate_frames": [
            {"frame_id": r[0], "video_id": r[1], "frame_sec": float(r[2])} for r in rows
        ],
    }


def measure_stages_2_and_3(
    conn: psycopg.Connection,
    query: str,
    gold,
    candidate_frame_rows: list[dict],
) -> dict:
    """Replicates ``retrieve.visual.retrieve`` from stage (b) onward to
    expose intermediate state — specifically the count of chunks that
    overlap the gold span post-join (stage 2) and the rank of the first
    gold-overlapping chunk after MaxSim (stage 3).

    Mirrors the exact SQL the production retriever uses.
    """
    if not candidate_frame_rows:
        return {
            "stage2_candidate_chunk_count": 0,
            "stage2_gold_overlapping_chunk_count": 0,
            "stage3_gold_chunk_rank": None,
            "stage3_scored_chunk_count": 0,
        }

    candidate_frame_ids = [r["frame_id"] for r in candidate_frame_rows]

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
        return {
            "stage2_candidate_chunk_count": 0,
            "stage2_gold_overlapping_chunk_count": 0,
            "stage3_gold_chunk_rank": None,
            "stage3_scored_chunk_count": 0,
        }

    chunk_frames: dict[int, list[int]] = defaultdict(list)
    chunk_meta: dict[int, tuple[str, float, float]] = {}
    for chunk_id, video_id, start_sec, end_sec, _text, frame_id in map_rows:
        chunk_frames[chunk_id].append(frame_id)
        chunk_meta.setdefault(chunk_id, (video_id, float(start_sec), float(end_sec)))

    stage2_gold_overlap_count = sum(
        1 for cid, meta in chunk_meta.items() if chunk_overlaps_gold(meta[0], meta[1], meta[2], gold)
    )

    # Stage 3: MaxSim refine — replicate the production code path exactly.
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
        return {
            "stage2_candidate_chunk_count": len(chunk_meta),
            "stage2_gold_overlapping_chunk_count": stage2_gold_overlap_count,
            "stage3_gold_chunk_rank": None,
            "stage3_scored_chunk_count": 0,
        }

    frame_patches: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame_id, _patch_idx, embedding in patch_rows:
        frame_patches[frame_id].append(embedding)

    qpatches = encode_text_query_patches(query).astype(np.float32)
    if qpatches.size == 0:
        return {
            "stage2_candidate_chunk_count": len(chunk_meta),
            "stage2_gold_overlapping_chunk_count": stage2_gold_overlap_count,
            "stage3_gold_chunk_rank": None,
            "stage3_scored_chunk_count": 0,
        }

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
    for i, (cid, _s) in enumerate(scored, start=1):
        meta = chunk_meta[cid]
        if chunk_overlaps_gold(meta[0], meta[1], meta[2], gold):
            gold_rank = i
            break

    return {
        "stage2_candidate_chunk_count": len(chunk_meta),
        "stage2_gold_overlapping_chunk_count": stage2_gold_overlap_count,
        "stage3_gold_chunk_rank": gold_rank,
        "stage3_scored_chunk_count": len(scored),
    }


def measure_example(conn: psycopg.Connection, ex, prior_rrf: dict[str, dict[str, bool]]) -> dict:
    """Full per-example diagnostic row."""
    # --- frame recall path ---
    frames = list(retrieve_frames(conn, ex.question, top_k=DEEP_K))
    frame_rank: int | None = None
    for f in frames:
        if frame_in_gold_window(f.video_id, f.frame_sec, ex):
            frame_rank = f.rank
            break

    # --- chunk TR path ---
    chunks = list(retrieve(conn, ex.question, top_k=DEEP_K, prefilter_k=DEEP_K))
    chunk_rank: int | None = None
    for c in chunks:
        if chunk_overlaps_gold(c.video_id, c.start_sec, c.end_sec, ex):
            chunk_rank = c.rank
            break

    # --- stage attribution (stage 1 candidates re-used for stages 2/3) ---
    stage1 = measure_stage_1(conn, ex.question, ex)
    stages_23 = measure_stages_2_and_3(conn, ex.question, ex, stage1["stage1_candidate_frames"])

    rrf_note = prior_rrf.get(ex.example_id, {})

    def _at(k: int, rank: int | None) -> bool:
        return rank is not None and rank <= k

    return {
        "example_id": ex.example_id,
        "video_id": ex.video_id,
        "gold_start_sec": ex.start_sec,
        "gold_end_sec": ex.end_sec,
        "frame_recall_rank": frame_rank,
        "chunk_tr_rank": chunk_rank,
        **{f"frame_recall_at_{k}": _at(k, frame_rank) for k in KS},
        **{f"chunk_tr_at_{k}": _at(k, chunk_rank) for k in KS},
        "stage_breakdown": {
            "stage_1_pooled_prefilter": {
                "candidates_returned": stage1["stage1_pooled_frame_count"],
                "gold_frame_rank_in_prefilter": stage1["stage1_gold_frame_rank"],
                "passed": stage1["stage1_gold_frame_rank"] is not None,
            },
            "stage_2_frame_to_chunk": {
                "candidate_chunks": stages_23["stage2_candidate_chunk_count"],
                "gold_overlapping_chunks": stages_23["stage2_gold_overlapping_chunk_count"],
                "passed": stages_23["stage2_gold_overlapping_chunk_count"] > 0,
            },
            "stage_3_maxsim": {
                "scored_chunks": stages_23["stage3_scored_chunk_count"],
                "gold_chunk_rank_after_maxsim": stages_23["stage3_gold_chunk_rank"],
                "passed_at_5": (
                    stages_23["stage3_gold_chunk_rank"] is not None
                    and stages_23["stage3_gold_chunk_rank"] <= 5
                ),
            },
            "stage_4_rrf_note": {
                "with_visual": rrf_note.get("with_visual"),
                "without_visual": rrf_note.get("without_visual"),
                "agree": rrf_note.get("with_visual") == rrf_note.get("without_visual"),
            },
        },
    }


def classify_loss_stage(row: dict) -> str:
    """Attribute the loss to the earliest pipeline stage where the gold
    falls out of the top-5 chunk band.

    Stage 1 miss: gold-window frame absent from the top-200 pooled prefilter
        → if you can't find the right frame, you can't surface the right chunk.
    Stage 2 miss: stage 1 found the frame but no chunk overlapping the gold
        survives the frame→chunk join.
    Stage 3 miss: stage 2 produced a gold-overlapping chunk but MaxSim
        ranked it outside the top 5.
    No loss: gold chunk ranks at or above 5.
    """
    s = row["stage_breakdown"]
    if not s["stage_1_pooled_prefilter"]["passed"]:
        return "stage_1_miss"
    if not s["stage_2_frame_to_chunk"]["passed"]:
        return "stage_2_miss"
    if not s["stage_3_maxsim"]["passed_at_5"]:
        return "stage_3_miss"
    return "passed_at_5"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_report(path: Path, rows: list[dict]) -> None:
    n = len(rows)

    def _rate(field: str) -> str:
        passed = sum(1 for r in rows if r[field])
        return f"{passed}/{n} ({100 * passed / n:.0f}%)"

    def _rank_bucket(rank: int | None) -> str:
        if rank is None:
            return "absent"
        if rank <= 5:
            return "1-5"
        if rank <= 20:
            return "6-20"
        if rank <= 50:
            return "21-50"
        return ">50"

    near_miss_frame = sum(1 for r in rows if r["frame_recall_rank"] and 6 <= r["frame_recall_rank"] <= 20)
    total_miss_frame = sum(
        1 for r in rows if r["frame_recall_rank"] is None or r["frame_recall_rank"] > 50
    )
    near_miss_chunk = sum(1 for r in rows if r["chunk_tr_rank"] and 6 <= r["chunk_tr_rank"] <= 20)
    total_miss_chunk = sum(
        1 for r in rows if r["chunk_tr_rank"] is None or r["chunk_tr_rank"] > 50
    )

    stage_counts = defaultdict(int)
    for r in rows:
        stage_counts[classify_loss_stage(r)] += 1

    lines: list[str] = []
    lines.append("# Visual Retrieval Diagnostics — 2026-05-28")
    lines.append("")
    lines.append(
        f"Side-channel report. n = {n} visual-required examples "
        "(tagged `visual-required` in `eval/corpora/ai_engineering_v0/visual_gold.jsonl`)."
    )
    lines.append("")
    lines.append("Scope: 3-stage `retrieve.visual.retrieve` pipeline + the `retrieve_frames` path.")
    lines.append("No production code modified. No `eval_runs` / `eval_results` written.")
    lines.append("")
    lines.append("## Aggregate rank curves")
    lines.append("")
    lines.append("| Metric | @5 | @10 | @20 | @50 |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| `VisualFrameRecall@k` | {_rate('frame_recall_at_5')} | "
        f"{_rate('frame_recall_at_10')} | {_rate('frame_recall_at_20')} | "
        f"{_rate('frame_recall_at_50')} |"
    )
    lines.append(
        f"| `VisualChunkTR@k` | {_rate('chunk_tr_at_5')} | "
        f"{_rate('chunk_tr_at_10')} | {_rate('chunk_tr_at_20')} | "
        f"{_rate('chunk_tr_at_50')} |"
    )
    lines.append("")
    lines.append("## Near-miss vs total-miss split")
    lines.append("")
    lines.append("- Near miss = gold rank 6-20 (within reach of a small reranker or wider k).")
    lines.append("- Total miss = gold rank > 50 or absent from top-200.")
    lines.append("")
    lines.append("| Path | near-miss (6-20) | total-miss (>50 or absent) |")
    lines.append("|---|---|---|")
    lines.append(f"| frame_recall | {near_miss_frame}/{n} | {total_miss_frame}/{n} |")
    lines.append(f"| chunk_tr | {near_miss_chunk}/{n} | {total_miss_chunk}/{n} |")
    lines.append("")
    lines.append("## Stage-loss attribution")
    lines.append("")
    lines.append("Where the gold-overlapping chunk falls out of the top-5 band, earliest stage first.")
    lines.append("")
    lines.append("| Stage | n | description |")
    lines.append("|---|---|---|")
    lines.append(
        f"| stage_1_miss | {stage_counts['stage_1_miss']} | gold-window frame absent from top-200 "
        "pooled HNSW prefilter |"
    )
    lines.append(
        f"| stage_2_miss | {stage_counts['stage_2_miss']} | prefilter found the frame but "
        "frame→chunk SQL join produced no gold-overlapping chunk |"
    )
    lines.append(
        f"| stage_3_miss | {stage_counts['stage_3_miss']} | gold-overlapping chunk surfaced post-join "
        "but MaxSim ranked it outside top-5 |"
    )
    lines.append(
        f"| passed_at_5  | {stage_counts['passed_at_5']} | gold chunk ranks at or above 5 |"
    )
    lines.append("")
    lines.append("## Per-example detail")
    lines.append("")
    lines.append(
        "| example_id | frame_rank | chunk_rank | stage 1 | stage 2 | stage 3 | RRF (with → without) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["stage_breakdown"]
        s1 = s["stage_1_pooled_prefilter"]
        s2 = s["stage_2_frame_to_chunk"]
        s3 = s["stage_3_maxsim"]
        s4 = s["stage_4_rrf_note"]
        s1_cell = (
            f"rank {s1['gold_frame_rank_in_prefilter']}/{s1['candidates_returned']}"
            if s1["passed"]
            else f"MISS (0/{s1['candidates_returned']})"
        )
        s2_cell = (
            f"{s2['gold_overlapping_chunks']}/{s2['candidate_chunks']} chunks"
            if s2["passed"]
            else f"MISS (0/{s2['candidate_chunks']})"
        )
        if s3["gold_chunk_rank_after_maxsim"] is not None:
            s3_cell = f"rank {s3['gold_chunk_rank_after_maxsim']}/{s3['scored_chunks']}"
        elif s2["passed"]:
            s3_cell = f"DROPPED (0/{s3['scored_chunks']})"
        else:
            s3_cell = "n/a"
        rrf_cell = f"{s4['with_visual']} → {s4['without_visual']}"
        lines.append(
            f"| `{r['example_id']}` | "
            f"{_rank_bucket(r['frame_recall_rank'])} ({r['frame_recall_rank']}) | "
            f"{_rank_bucket(r['chunk_tr_rank'])} ({r['chunk_tr_rank']}) | "
            f"{s1_cell} | {s2_cell} | {s3_cell} | {rrf_cell} |"
        )
    lines.append("")
    lines.append("## Per-example narrative")
    lines.append("")
    for r in rows:
        lines.append(f"### `{r['example_id']}`")
        lines.append("")
        s = r["stage_breakdown"]
        s1 = s["stage_1_pooled_prefilter"]
        s2 = s["stage_2_frame_to_chunk"]
        s3 = s["stage_3_maxsim"]
        s4 = s["stage_4_rrf_note"]
        lines.append(
            f"Gold span: `{r['video_id']}` "
            f"[{r['gold_start_sec']:.0f}, {r['gold_end_sec']:.0f}] sec. "
            f"frame_recall_rank = `{r['frame_recall_rank']}`; chunk_tr_rank = `{r['chunk_tr_rank']}`."
        )
        lines.append("")
        if s1["passed"]:
            lines.append(
                f"- **Stage 1 (pooled HNSW prefilter):** PASS — the first gold-window frame appears at "
                f"rank `{s1['gold_frame_rank_in_prefilter']}` out of `{s1['candidates_returned']}` pooled candidates."
            )
        else:
            lines.append(
                f"- **Stage 1 (pooled HNSW prefilter):** MISS — no gold-window frame in the top "
                f"`{s1['candidates_returned']}` pooled candidates. The right frame is buried below the "
                "prefilter horizon; widening `prefilter_k` may or may not help, but the cosine signal "
                "on the pooled vector is not surfacing it within 200."
            )
        if s2["passed"]:
            lines.append(
                f"- **Stage 2 (frame→chunk join):** PASS — "
                f"`{s2['gold_overlapping_chunks']}` of `{s2['candidate_chunks']}` candidate chunks overlap the gold span."
            )
        elif s1["passed"]:
            lines.append(
                f"- **Stage 2 (frame→chunk join):** MISS — stage 1 found the frame but none of the "
                f"`{s2['candidate_chunks']}` post-join candidate chunks overlap the gold span. Likely "
                "a chunking-window alignment artifact: the gold-window frame sits in a chunk that does not "
                "cover the gold time interval, or the frame falls outside any chunk's half-open window."
            )
        else:
            lines.append(
                "- **Stage 2 (frame→chunk join):** n/a — stage 1 already failed."
            )
        if s3["gold_chunk_rank_after_maxsim"] is not None:
            lines.append(
                f"- **Stage 3 (MaxSim refine):** the gold-overlapping chunk ranks "
                f"`{s3['gold_chunk_rank_after_maxsim']}` of `{s3['scored_chunks']}` after MaxSim. "
                + ("Passes at @5." if s3["passed_at_5"] else "Does not pass at @5.")
            )
        elif s2["passed"]:
            lines.append(
                f"- **Stage 3 (MaxSim refine):** DROPPED — gold chunk had no patches in `frame_patches` "
                f"to score against (chunk fell out of the scored set of `{s3['scored_chunks']}`)."
            )
        else:
            lines.append("- **Stage 3 (MaxSim refine):** n/a — no gold-overlapping chunk reached this stage.")
        lines.append(
            f"- **Stage 4 (RRF, informational):** prior eval shows `with_visual={s4['with_visual']}`, "
            f"`without_visual={s4['without_visual']}` "
            + ("(agree)." if s4["agree"] else "(disagree).")
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
            f"(tagged ids in jsonl: {sorted(visual_required_ids)})"
        )

    prior_rrf = load_prior_rrf_per_example()

    rows: list[dict] = []
    with psycopg.connect(dsn()) as conn:
        for i, ex in enumerate(examples, start=1):
            print(f"[{i}/{len(examples)}] {ex.example_id}")
            row = measure_example(conn, ex, prior_rrf)
            rows.append(row)

    # Stable per-example_id order for the report.
    rows.sort(key=lambda r: r["example_id"])

    write_jsonl(REPORT_DIR / "rank_curves.jsonl", rows)
    write_report(REPORT_DIR / "report.md", rows)
    print(f"wrote {REPORT_DIR / 'rank_curves.jsonl'}")
    print(f"wrote {REPORT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
