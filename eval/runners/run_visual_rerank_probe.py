"""Offline proof that the fixed ColQwen2.5 loader restores visual frame recall.

Side-channel probe. Reads the 10 ``visual-required`` rows and the frame PNGs on
disk. Writes ONLY to a report dir. Does NOT touch ``retrieve/visual.py``, the
ingest write path, the DB ``eval_runs``/``eval_results`` tables, or any schema,
and it does NOT write the re-encoded vectors back to the ``frames`` table — the
DB patches stay stale until a separately-approved re-ingest.

Context: the visual-required failure diagnosed in
``eval/reports/2026-05-30_visual_rerank_probe/methodology.mdx`` was not a pooling
or MaxSim problem. It was a broken ColQwen load — the LoRA adapter and the token
embeddings were both randomly re-initialized on every process (a transformers
key-rename). ``embed/colqwen.py`` now fixes that. This probe proves the fix at the
metric level: re-encode the 514 frame images AND the 10 queries with the current
(fixed, deterministic) encoder, then measure ``VisualFrameRecall@{5,10,20,200}``.

For contrast it also ranks the fixed query against the STORED (broken-encoder)
pooled embeddings — the production path as it stands today, before re-ingest — so
the before/after is explicit.

WARNING — one ML model only (ColQwen). Image encoding is one-at-a-time (~3s/image
on MPS; multi-image batches trip a get_rope_index error). The re-encoded frame
vectors are cached to an npz in the report dir so re-runs of the scoring need not
repeat the ~27-minute encode.

Usage:
    uv run python -m eval.runners.run_visual_rerank_probe --phase a-fixed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from db.conn import dsn
from eval.runners.measure_visual import FRAME_SAMPLE_EVERY_SEC, load_visual_gold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VISUAL_GOLD_PATH = PROJECT_ROOT / "eval" / "corpora" / "ai_engineering_v0" / "visual_gold.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "eval" / "reports" / "2026-05-30_visual_rerank_probe"

TOLERANCE_SEC = FRAME_SAMPLE_EVERY_SEC  # 10.0
RECALL_KS: tuple[int, ...] = (5, 10, 20, 200)
POOLED_DIM = 128


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


def _l2norm(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def _best_gold_rank(scores: np.ndarray, gold_idxs: list[int]) -> int | None:
    """1-indexed rank of the highest-scored gold-window frame, or None."""
    if not gold_idxs:
        return None
    order = np.argsort(-scores, kind="stable")
    rank_of = np.empty(len(scores), dtype=np.int64)
    rank_of[order] = np.arange(1, len(scores) + 1)
    return int(min(rank_of[i] for i in gold_idxs))


def run_phase_a_fixed(out_dir: Path) -> None:
    """Recall proof under the FIXED, deterministic ColQwen encoder.

    The DB's stored frame patches were produced by the broken (random-adapter)
    encoder, so they cannot be trusted. This re-encodes the 514 frame PNGs AND
    the 10 queries in memory with the current encoder (single consistent
    encoder), ranks frames by the production stage-(a) signal (cosine of
    mean-pooled query vs mean-pooled frame), and reports
    ``VisualFrameRecall@{5,10,20,200}``. The ``stored_broken`` arm ranks the
    fixed query against the stale stored embeddings for contrast.

    One ML model only (ColQwen). No DB write; stored embeddings are read-only.
    """
    from PIL import Image

    from embed.colqwen import encode_image_pooled, encode_text_query

    visual_required = load_visual_required_ids()
    examples = [e for e in load_visual_gold() if e.example_id in visual_required]
    if len(examples) != 10:
        raise RuntimeError(f"expected 10 visual-required examples; got {len(examples)}")

    with psycopg.connect(dsn()) as conn:
        register_vector(conn)
        rows = conn.execute(
            "SELECT frame_id, video_id, frame_sec, image_path, pooled_embedding "
            "FROM frames WHERE pooled_embedding IS NOT NULL ORDER BY frame_id"
        ).fetchall()
    frame_ids = [r[0] for r in rows]
    vids = [r[1] for r in rows]
    secs = [float(r[2]) for r in rows]
    paths = [r[3] for r in rows]
    stored = _l2norm(np.stack([np.asarray(r[4], dtype=np.float32) for r in rows]))
    print(f"{len(rows)} frames; re-encoding images under the fixed encoder ...")

    # Re-encode all frame images (mean-pooled), one at a time — the same way the
    # ingest handler does it. Multi-image batches trip a get_rope_index bounds
    # error in the Qwen2.5-VL forward on this stack. ~3s/image on MPS, so cache
    # the result to npz (keyed by frame_id): the encode is the slow part and is
    # deterministic, so a re-run of the scoring need not repeat it.
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "phase_a_fixed_frame_embeds.npz"
    fresh: np.ndarray | None = None
    if cache.exists():
        z = np.load(cache)
        if list(z["frame_ids"]) == frame_ids:
            fresh = z["fresh"].astype(np.float32)
            print(f"loaded {len(rows)} fixed-encoder frame embeds from cache")
        else:
            cache.unlink()
    if fresh is None:
        fresh = np.empty((len(rows), POOLED_DIM), dtype=np.float32)
        for i, path in enumerate(paths):
            with Image.open(path) as img:
                fresh[i] = encode_image_pooled([img.convert("RGB")])[0]
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"  encoded {i + 1}/{len(rows)}", flush=True)
        np.savez(cache, fresh=fresh, frame_ids=np.array(frame_ids, dtype=np.int64))
    fresh_n = _l2norm(fresh)

    qvecs = _l2norm(np.stack([encode_text_query(e.question) for e in examples]).astype(np.float32))

    def recall_row(scores: np.ndarray, gi: list[int]) -> dict:
        rank = _best_gold_rank(scores, gi)
        return {
            "gold_frame_rank": rank,
            **{f"recall_at_{k}": bool(rank is not None and rank <= k) for k in RECALL_KS},
        }

    out_rows: list[dict] = []
    for ei, ex in enumerate(examples):
        lo, hi = ex.start_sec - TOLERANCE_SEC, ex.end_sec + TOLERANCE_SEC
        gi = [j for j in range(len(rows)) if vids[j] == ex.video_id and lo <= secs[j] <= hi]
        for arm, mat in (("stored_broken", stored), ("fixed_reencode", fresh_n)):
            out_rows.append(
                {"example_id": ex.example_id, "arm": arm, **recall_row(mat @ qvecs[ei], gi)}
            )

    with (out_dir / "phase_a_fixed_data.jsonl").open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")

    print("\n=== VisualFrameRecall@k under the fixed encoder (n=10) ===")
    print(f"{'arm':<18}" + "".join(f"@{k:<6}" for k in RECALL_KS))
    for arm in ("stored_broken", "fixed_reencode"):
        ar = [r for r in out_rows if r["arm"] == arm]
        cells = "".join(f"{sum(r[f'recall_at_{k}'] for r in ar):>2}/10  " for k in RECALL_KS)
        print(f"{arm:<18}{cells}")
    print(f"\nwrote {out_dir / 'phase_a_fixed_data.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["a-fixed"], default="a-fixed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_phase_a_fixed(args.out)


if __name__ == "__main__":
    main()
