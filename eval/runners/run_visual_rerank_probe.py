"""Offline two-phase hybrid visual diagnostic — recall fix (A) then ranking fix (B).

Side-channel probe. Reads existing `frame_patches` + the 10 `visual-required`
rows. Writes ONLY to a report dir. Does NOT touch `retrieve/visual.py`, the
ingest write path, the DB `eval_runs`/`eval_results` tables, or any schema.

Acts on the diagnosis in `eval/reports/2026-05-28_visual_ablations/report.md`:

  1. Production `prefilter_k=200` is prefilter-limited — the mean-pool single
     descriptor ranks 6/10 gold frames past rank 200 (the K=200 column shows 4
     present, 6 absent). Mean-pooling 731 unit patches into one 128-d vector
     washes out the few discriminative patches.
  2. Widening to K=500 lands all 10 gold frames but MaxSim then demotes the
     gold-overlapping chunk to rank 60-172 of ~210 — a near-uniform similarity
     floor.

WARNING — Phase A loads ColQwen ONCE to encode the 10 text queries, then frees
it before the numpy/scipy pooling math. Phase B loads a reranker ALONE after
ColQwen is freed. NEVER run two ML-loading processes at once on this 24 GB Mac.

Phase A — recall fix, no new model. Re-score all 514 pooled frames per query
under several prefilter signals and measure `VisualFrameRecall@{5,10,20,200}`:

  * `prod_mean_cosine` — cosine(mean(query token patches), mean(frame patches)),
    the production stage-(a) signal. NOTE: this arm did NOT match the committed
    ablation ranks (6/10 vs 4/10 in prefilter@200) — that mismatch is what
    exposed the broken, non-deterministic ColQwen load. Treat every Phase A
    recall number as diagnostic only; it must not guide any model or pooling
    decision until ColQwen loading is fixed (see methodology.mdx).
  * `maxsim_full` — late-interaction ceiling: MaxSim(query token patches, ALL
    731 frame patches). No pooling, no compression. Upper bound on what a
    late-interaction prefilter can recall.
  * `maxsim_hier_pf3` / `maxsim_hier_pf2` — the deployable recall fix:
    MaxSim against colpali-engine `HierarchicalTokenPooler` clusters
    (pool_factor 3 / 2). 97.8% nDCG retained at pf=3 per the research matrix.
  * `maxsim_gauss` — model-correct same-length Gaussian smoothing of the 731
    patches (Visual RAG Toolkit, arXiv:2602.12510) then MaxSim. Secondary arm.

The acceptance gate for Phase A: some reversible arm reaches
`VisualFrameRecall@200 >= 9/10`. The deployable arm is hierarchical pooling;
`maxsim_full` is the ceiling reference.

Phase B — ranking fix (one model). Built only after Phase A passes. See
`run_phase_b`.

Usage:
    uv run python -m eval.runners.run_visual_rerank_probe --phase a --prefilter-k 500
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from db.conn import dsn
from eval.runners.measure_visual import (
    FRAME_SAMPLE_EVERY_SEC,
    load_visual_gold,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VISUAL_GOLD_PATH = PROJECT_ROOT / "eval" / "corpora" / "ai_engineering_v0" / "visual_gold.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "eval" / "reports" / "2026-05-30_visual_rerank_probe"

TOLERANCE_SEC = FRAME_SAMPLE_EVERY_SEC  # 10.0
RECALL_KS: tuple[int, ...] = (5, 10, 20, 200)
POOLED_DIM = 128

# Arms scored in Phase A. `prod_mean_cosine` is the single-vector cosine
# baseline (the production stage-(a) signal); the rest are MaxSim over
# (optionally pooled) frame vectors. The baseline mismatched the committed
# ablation, which surfaced the broken ColQwen load — see module docstring.
PHASE_A_ARMS: tuple[str, ...] = (
    "prod_mean_cosine",
    "maxsim_full",
    "maxsim_hier_pf3",
    "maxsim_hier_pf2",
    "maxsim_gauss",
)


def load_visual_required_ids() -> set[str]:
    ids: set[str] = set()
    for raw in VISUAL_GOLD_PATH.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        ex = json.loads(line)
        if "visual-required" in (ex.get("tags") or []):
            ids.add(ex["id"])
    return ids


@dataclass
class FrameRow:
    frame_id: int
    video_id: str
    frame_sec: float
    image_path: str
    patches: np.ndarray  # (P, 128) float32, unit-norm rows


def load_frames(conn: psycopg.Connection) -> list[FrameRow]:
    """Load every frame that carries patches, with its full patch matrix.

    Returns frames in a stable order (frame_id ascending). Frames without a
    pooled embedding are excluded by the join — they carry no patches either
    (the NaN/inf clear-on-skip path NULLs pooled AND deletes patches)."""
    register_vector(conn)
    meta = {
        r[0]: (r[1], float(r[2]), r[3])
        for r in conn.execute(
            "SELECT frame_id, video_id, frame_sec, image_path "
            "FROM frames WHERE pooled_embedding IS NOT NULL ORDER BY frame_id"
        ).fetchall()
    }
    patch_rows = conn.execute(
        "SELECT frame_id, patch_index, embedding FROM frame_patches "
        "ORDER BY frame_id, patch_index"
    ).fetchall()
    by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    for fid, _pidx, emb in patch_rows:
        by_frame[fid].append(emb)
    frames: list[FrameRow] = []
    for fid in sorted(by_frame):
        if fid not in meta:
            continue
        vid, sec, path = meta[fid]
        patches = np.stack(by_frame[fid]).astype(np.float32)
        frames.append(FrameRow(fid, vid, sec, path, patches))
    return frames


def encode_queries(questions: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Encode the queries with ColQwen, then free the model.

    Returns (pooled (N,128), token_patches [list of (T_i,128)]). The model is
    released (lru_cache cleared + MPS cache emptied) before returning so the
    pooling/scoring math runs with no ML model resident — and so Phase B can
    later load its reranker into a clean memory state."""
    from embed.colqwen import _model, encode_text_query, encode_text_query_patches

    pooled = np.stack([encode_text_query(q) for q in questions]).astype(np.float32)
    token_patches = [encode_text_query_patches(q).astype(np.float32) for q in questions]

    _model.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    return pooled, token_patches


def _l2norm(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def _hier_pool(frames: list[FrameRow], pool_factor: int) -> list[np.ndarray]:
    """HierarchicalTokenPooler clusters per frame → list of (m,128) float32.

    Pure CPU (scipy ward linkage). The pooler L2-normalizes each cluster mean.
    """
    import torch
    from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

    pooler = HierarchicalTokenPooler()
    tensors = [torch.from_numpy(f.patches) for f in frames]
    pooled = pooler.pool_embeddings(tensors, pool_factor=pool_factor, num_workers=4)
    return [p.to(torch.float32).cpu().numpy() for p in pooled]


def _gauss_smooth(frames: list[FrameRow], sigma: float = 1.0) -> list[np.ndarray]:
    """Same-length Gaussian smoothing along the patch (raster) axis, renormalized.

    Crude: ColQwen's 731 patches are a flattened dynamic-resolution grid, so a
    1-D smoother over raster order only approximates spatial smoothing. Reported
    as a secondary arm per the Visual RAG Toolkit; not the deployable fix."""
    from scipy.ndimage import gaussian_filter1d

    out: list[np.ndarray] = []
    for f in frames:
        smoothed = gaussian_filter1d(f.patches, sigma=sigma, axis=0, mode="nearest")
        out.append(_l2norm(smoothed.astype(np.float32)))
    return out


def _maxsim_scores(q_patches: np.ndarray, vall: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """MaxSim of one query (T,128) against concatenated frame vectors.

    `vall` is (sum_m, 128); `offsets` are the per-frame segment starts. Returns
    one MaxSim score per frame = sum_t max over the frame's vectors of cosine.
    All vectors are unit-norm, so dot == cosine."""
    if q_patches.size == 0:
        return np.full(len(offsets), -np.inf, dtype=np.float32)
    sims = q_patches @ vall.T  # (T, sum_m)
    seg_max = np.maximum.reduceat(sims, offsets, axis=1)  # (T, n_frames)
    return seg_max.sum(axis=0)


def _concat(reps: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(reps), dtype=np.int64)
    acc = 0
    for i, r in enumerate(reps):
        offsets[i] = acc
        acc += r.shape[0]
    vall = np.concatenate(reps, axis=0).astype(np.float32)
    return vall, offsets


def _gold_window_frame_idxs(frames: list[FrameRow], gold) -> list[int]:
    lo = gold.start_sec - TOLERANCE_SEC
    hi = gold.end_sec + TOLERANCE_SEC
    return [
        i
        for i, f in enumerate(frames)
        if f.video_id == gold.video_id and lo <= f.frame_sec <= hi
    ]


def _best_gold_rank(scores: np.ndarray, gold_idxs: list[int]) -> int | None:
    """1-indexed rank of the highest-scored gold-window frame, or None."""
    if not gold_idxs:
        return None
    order = np.argsort(-scores, kind="stable")
    rank_of = np.empty(len(scores), dtype=np.int64)
    rank_of[order] = np.arange(1, len(scores) + 1)
    return int(min(rank_of[i] for i in gold_idxs))


def run_phase_a(out_dir: Path, prefilter_k: int) -> None:
    visual_required = load_visual_required_ids()
    examples = [e for e in load_visual_gold() if e.example_id in visual_required]
    if len(examples) != 10:
        raise RuntimeError(f"expected 10 visual-required examples; got {len(examples)}")

    with psycopg.connect(dsn()) as conn:
        frames = load_frames(conn)
    print(f"loaded {len(frames)} frames, {sum(f.patches.shape[0] for f in frames)} patches")

    pooled_q, token_patches = encode_queries([e.question for e in examples])
    print("queries encoded; ColQwen freed")

    gold_idxs = [_gold_window_frame_idxs(frames, e) for e in examples]
    for e, gi in zip(examples, gold_idxs, strict=True):
        if not gi:
            print(f"  WARNING: no gold-window frame for {e.example_id}")

    # --- Precompute per-arm frame representations ---
    frame_means = _l2norm(np.stack([f.patches.mean(axis=0) for f in frames]).astype(np.float32))
    arm_reps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    print("pooling: hier pf3 ...")
    arm_reps["maxsim_hier_pf3"] = _concat(_hier_pool(frames, 3))
    print("pooling: hier pf2 ...")
    arm_reps["maxsim_hier_pf2"] = _concat(_hier_pool(frames, 2))
    print("pooling: full ...")
    arm_reps["maxsim_full"] = _concat([f.patches for f in frames])
    print("pooling: gauss ...")
    arm_reps["maxsim_gauss"] = _concat(_gauss_smooth(frames))

    # --- Score every example under every arm ---
    rows: list[dict] = []
    qn = _l2norm(pooled_q)
    for ei, ex in enumerate(examples):
        gi = gold_idxs[ei]
        for arm in PHASE_A_ARMS:
            if arm == "prod_mean_cosine":
                scores = frame_means @ qn[ei]
            else:
                vall, offsets = arm_reps[arm]
                scores = _maxsim_scores(token_patches[ei], vall, offsets)
            rank = _best_gold_rank(scores, gi)
            in_prefilter = rank is not None and rank <= prefilter_k
            rows.append(
                {
                    "example_id": ex.example_id,
                    "arm": arm,
                    "gold_frame_rank": rank,
                    "n_frames": len(frames),
                    "in_prefilter_k": bool(in_prefilter),
                    "prefilter_k": prefilter_k,
                    **{f"recall_at_{k}": bool(rank is not None and rank <= k) for k in RECALL_KS},
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "phase_a_data.jsonl"
    with data_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    _write_phase_a_report(out_dir / "phase_a_report.md", examples, rows, prefilter_k, len(frames))
    print(f"wrote {data_path}")
    print(f"wrote {out_dir / 'phase_a_report.md'}")

    # --- Console summary: FrameRecall@k per arm ---
    print("\n=== Phase A — VisualFrameRecall@k (n=10) ===")
    header = f"{'arm':<20}" + "".join(f"@{k:<6}" for k in RECALL_KS)
    print(header)
    for arm in PHASE_A_ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        cells = "".join(f"{sum(r[f'recall_at_{k}'] for r in arm_rows):>2}/10  " for k in RECALL_KS)
        print(f"{arm:<20}{cells}")


def _write_phase_a_report(
    path: Path, examples, rows: list[dict], prefilter_k: int, n_frames: int
) -> None:
    by_ex_arm = {(r["example_id"], r["arm"]): r for r in rows}
    lines: list[str] = []
    lines.append("# Phase A — recall fix (offline pooling probe)")
    lines.append("")
    lines.append(
        f"Side-channel probe. n = 10 visual-required rows. {n_frames} frames re-scored "
        f"per query under each prefilter signal. No model load except one-time query "
        f"encoding; no production/DB writes. prefilter_k = {prefilter_k}."
    )
    lines.append("")
    lines.append(
        "**These numbers are DIAGNOSTIC ONLY and must not guide any model or pooling "
        "decision.** The `prod_mean_cosine` arm did NOT reproduce the committed ablation "
        "ranks (6/10 vs 4/10 in prefilter@200). That mismatch exposed a broken, "
        "non-deterministic ColQwen load (the LoRA adapter is randomly re-initialized "
        "each process). Until ColQwen loading is fixed, every recall count below is "
        "computed from a random-draw query encoder and is not reproducible. See "
        "`methodology.mdx` for the root cause."
    )
    lines.append("")

    # Aggregate table
    lines.append("## VisualFrameRecall@k by arm")
    lines.append("")
    lines.append("| arm | @5 | @10 | @20 | @200 |")
    lines.append("|---|---|---|---|---|")
    for arm in PHASE_A_ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        cells = " | ".join(
            f"{sum(r[f'recall_at_{k}'] for r in arm_rows)}/10" for k in RECALL_KS
        )
        lines.append(f"| `{arm}` | {cells} |")
    lines.append("")

    # Per-example gold-frame rank by arm
    lines.append("## Gold-frame rank by arm (of " + str(n_frames) + ")")
    lines.append("")
    lines.append("| example_id | " + " | ".join(f"`{a}`" for a in PHASE_A_ARMS) + " |")
    lines.append("|---|" + "---|" * len(PHASE_A_ARMS))
    for ex in examples:
        cells = []
        for arm in PHASE_A_ARMS:
            r = by_ex_arm[(ex.example_id, arm)]
            cells.append(str(r["gold_frame_rank"]) if r["gold_frame_rank"] is not None else "—")
        lines.append(f"| `{ex.example_id}` | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["a", "b"], required=True)
    parser.add_argument("--prefilter-k", type=int, default=500)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.phase == "a":
        run_phase_a(args.out, args.prefilter_k)
    else:
        raise SystemExit("Phase B not yet implemented — run Phase A and gate on it first.")


if __name__ == "__main__":
    main()
