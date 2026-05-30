# prompt49 — visual retrieval fix experiment (offline hybrid diagnostic)

Repo: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph

## What happened last session (week-8 review + research)

The v4 working tree was verified against the DB and committed with three Codex
fixes (NaN/inf clear-on-skip + tests, ablation report 360p→720p, accurate
two-stage blocker framing). A web-research matrix was produced at
`dev/active/phase-2-week-8/research_matrix.md`. The diagnosis is settled:

1. **Prefilter-limited** at production `prefilter_k=200` — 7/10 gold frames miss
   the pooled-cosine HNSW top-200.
2. **MaxSim demotion** once prefilter is widened — at K=500 all 10 gold frames
   enter the candidate set (ranks 136-444 of 514), then MaxSim ranks the
   gold-overlapping chunk at 60-172 of ~210. Near-uniform similarity floor.

Both are known pipeline pathologies, not a model defect. Fix the pipeline before
swapping the model.

## This session's task — run ONE experiment, in two phases, fully offline

Matrix rows implemented: "Pooling fix (late-interaction)" then "Visual reranker
(top pick)". This neither retires visual retrieval nor switches to OCR-only.

Read first:
- `dev/active/phase-2-week-8/research_matrix.md` (the chosen rows + acceptance criteria)
- `dev/active/phase-2-week-8/handoff.md`
- `eval/reports/2026-05-28_visual_ablations/report.md` (the diagnosis)
- `retrieve/visual.py`, `eval/runners/measure_visual.py`, `eval/runners/run_visual_eval.py`

### Phase A — recall fix (no new model, pure offline)

Recompute the prefilter signal over the existing 514 frames' patch arrays using
`colpali-engine`'s hierarchical token pooler — the verified local API is
`from colpali_engine.compression.token_pooling import HierarchicalTokenPooler`,
then `HierarchicalTokenPooler().pool_embeddings(embeddings, pool_factor=3)` (the
`pool_factor` is an argument to `pool_embeddings`, not the constructor). As a
second arm, Gaussian same-length smoothing per arXiv:2602.12510. Set
`prefilter_k=500`,
and measure `VisualFrameRecall@{5,10,20,200}` on the 10-example visual-required
slice. Zero model load, zero MPS contention. This is a precondition — ranking
fixes are pointless until gold frames are in the candidate set.

### Phase B — ranking fix (one model, swapped in) — only if Phase A lands gold frames

Dump the **top-200** candidate frames per example (gold sits at ranks 137-172, so
never top-100). Unload ColQwen2.5, load **Qwen3-VL-Reranker-2B** alone (respects
the one-model-at-a-time 24 GB limit — see memory rule below), score each
(question, frame_png), re-sort, map frames→chunks via the existing stage (b), and
recompute `VisualChunkTR@{5,10,20}` + `VisualAnswerGrounding@{5,10}`. Optional
comparison arms on the same top-200: the frame-OCR-text channel (Tesseract →
installed `bge-reranker-v2-m3`) and a one-off hosted Qwen2.5-VL-7B judge (~$0.50).

## Constraints (non-negotiable)

- Do NOT retire visual retrieval. Do NOT switch to OCR-only. Do NOT start the
  generator/judge comparison (bakeoff #2).
- Do NOT touch `test_gold.jsonl`. Do NOT touch the 8 text-saturated rows.
- One ML model loaded at a time. ColQwen2.5 ≈ 7-8 GB on MPS. NEVER run two
  ML-loading processes concurrently (it crashed Cursor last time). `sudo purge`
  between heavy loads if swap > 7 GB.
- Keep unrelated `.claude/.cursor/prompts/`/formatting churn out of commits. Use
  explicit `git add <paths>`, never `git add -A`.
- Any positive `VisualAnswerGrounding` result requires the structured audit
  payload FIRST — implement `dev/active/phase-2-week-8/audit_payload_design.md`
  (option (a), `metrics.visual_answer_grounding` JSONB) before claiming any pass.

## Files likely to change

- New offline script under `eval/runners/` (e.g. `eval/runners/run_visual_rerank_probe.py`).
- `embed/colqwen.py` — an offline-callable alternate prefilter pooling function
  (HierarchicalTokenPooler / Gaussian smoothing), NOT yet wired into the write path.
- `eval/config/model_candidates.yaml` — one row for the Qwen3-VL-Reranker-2B candidate.
- If a positive lands: `eval/runners/measure_visual.py` + `eval/runners/run_visual_eval.py`
  for the audit payload (per the design doc), plus regression tests in
  `eval/tests/test_run_visual_eval.py`.
- Possibly `docs/decisions/004-model-selection.md` (ADR amendment) only if a new
  model is promoted from probe to production — not for the offline probe itself.

## Commands

```bash
git status --short --untracked-files=all
make db-up && make db-migrate
# Phase A (no model):
uv run python -m eval.runners.run_visual_rerank_probe --phase a --prefilter-k 500
# Phase B (Qwen3-VL-Reranker-2B, ColQwen unloaded first):
uv run python -m eval.runners.run_visual_rerank_probe --phase b --top-frames 200
# Gates before any commit touching retrieve/eval-runner/embed:
make validate-evals-strict
make phase0-gate
make lint
make test
# Slow integration test only if eval/runners/ production code changed:
uv run pytest -m slow eval/tests/test_run_visual_eval.py -q
make db-down
```

## Acceptance criteria (tied to existing metrics)

- **VisualFrameRecall@200 ≥ 9/10** after Phase A (recall bug fixed only if gold
  frames actually enter the candidate set).
- **VisualChunkTR@5 ≥ 5/10** after Phase B reranking the top-200.
- **VisualAnswerGrounding@5 strictly > 0/10**, target ≥4/10 (requires the audit
  payload landed first).
- If FrameRecall@200 ≥ 9/10 but ChunkTR@5 stays low → failure isolated to
  ranking; promote the trained-reranker / OCR-text arm. If even the reranker
  cannot lift ChunkTR → the MaxSim floor is intrinsic to the 3-talk corpus; the
  corpus needs more talks before the visual channel can be evaluated fairly
  (document this honestly, do not force a positive).

## Rollback plan (every change reversible without losing the v3 baseline)

- Phase A/B are offline scripts that write nothing to production code and nothing
  to the DB until results justify it. If a phase fails, delete the script — the
  committed v3/v4 baseline (HEAD) is untouched.
- The alternate pooling function in `embed/colqwen.py` is additive and not wired
  into the ingest write path, so production retrieval keeps using the current mean
  pool. Reverting is a single function deletion.
- The `model_candidates.yaml` reranker row is a candidate, not a default (ADR 004
  selection rule). No model becomes the default until a bakeoff runs.
- The audit payload change drops the old scalar `visual_answer_grounding_at_k`
  per-row key; the run-level float in `eval_runs.summary` is preserved, so prior
  reports still read. Revert by `git checkout` of the two runner files.
- Verified baseline to return to: `git log --oneline` HEAD after the week-8 commits.
