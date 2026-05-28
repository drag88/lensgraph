# Visual Ablations — Execution Deferred

The script `run_ablations.py` in this directory is **written but not yet run**.

## Why deferred

This session was memory-constrained on a 24 GB Mac (7.36 GB swap used). Two
prior parallel ColQwen loads crashed Cursor. ColQwen 2.5 takes ~7-8 GB on
MPS unified memory, and we are consolidating to one load per session.
Execution waits for a follow-up session that owns the only ML job on the
machine.

## What the script measures when it runs

1. **Prefilter K sweep** — for each of the 10 visual-required examples,
   runs the 3-stage pipeline (pooled HNSW → frame→chunk join → MaxSim) at
   `prefilter_k ∈ {200, 500, 1000}`. Records rank of first gold-window
   frame in the prefilter pool, rank of first gold-overlapping chunk in
   the final MaxSim ranking, and pass@5/@10/@20/@50.
2. **MaxSim demotion deep-dive** — for the 4 stage-3-failure rows,
   exposes the top-5 MaxSim chunks (text snippet + score) and the gold
   chunk's score, rank, and score gap to the top-5 cutoff.

## Prerequisites before running

- Postgres up: `make db-up`.
- No other ColQwen-loading process active (`run_visual_eval.py`,
  `run_diagnostics.py`, `embed_frames` worker, any other agent's script
  that imports `embed.colqwen`).
- Recommended: `sudo purge` before launch to flush page cache.

## Command

```bash
uv run python eval/reports/2026-05-28_visual_ablations/run_ablations.py
```

## Outputs (land in this directory)

- `ablation_data.jsonl` — one row per (example × prefilter_k).
- `report.md` — human-readable narrative with K-rescue table, MaxSim
  deep-dive, and per-example failure classification.

## What numbers to look for

- Which `prefilter_k` value (if any) rescues the 5 stage-1 misses. If
  K=1000 (cap=529) rescues none, the failure is embedding-side
  (frame resolution / ColQwen on 360p), not prefilter-width.
- Whether the MaxSim score gap between top-5 cutoff and the gold chunk
  narrows at higher K. If it does not, the demotion is a MaxSim ranking
  pathology, not a sampling artifact.
