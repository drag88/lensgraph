# Handoff — Phase 2 Week 8

Opened: 2026-05-28. Refreshed: 2026-05-29 (review + research session close).
Status: v4 reviewed and committed with Codex fixes; research matrix shipped; next
experiment scoped in `prompts/prompt49.md`. Not implemented this session.

## Five-line read

1. The v4 working tree was verified against the DB — every claimed number holds
   exactly (529 frames / 514 pooled / 375,734 patches / 1280×720 / vag=0.0,n=10).
2. Three Codex findings fixed and committed: NaN/inf clear-on-skip + tests; the
   ablation report's stale 360p preamble; the self-contradicting conclusion.
3. The blocker is reframed accurately: production is prefilter-limited at K=200
   (7/10 gold frames miss), and widening to K=500 exposes a MaxSim/ranking floor.
4. A web-research matrix (`research_matrix.md`, 13 candidates, links + dates) and
   an audit-payload design (`audit_payload_design.md`) were produced.
5. Next session runs ONE offline hybrid diagnostic (`prompts/prompt49.md`):
   Phase A pooling fix + prefilter_k=500 for recall, Phase B Qwen3-VL-Reranker-2B
   over top-200 for ranking. No production/DB writes until results justify.

## V4 disposition (per file)

| File | Disposition |
|---|---|
| `ingest/handlers.py` | **Committed, revised.** Kept the NaN/inf skip; added a call to the new `frames_repo.clear_embeddings` before each early return so a skipped frame is genuinely absent from retrieval (Codex finding 1). |
| `db/repos/frames.py` | **Committed (new).** Added `clear_embeddings(conn, frame_id)` — NULL pooled + DELETE patches. (Also carries pre-existing ruff line-wrap churn, which is correct `make fmt` output.) |
| `eval/tests/test_handlers_fast.py` | **Committed.** 3 new fast tests: non-finite patches clear+skip, non-finite pooled clear+skip, finite path does not clear. |
| `eval/reports/2026-05-28_visual_ablations/report.md` | **Committed, revised.** Preamble 360p/529-pooled/163,990 → 720p/514/375,734/1280×720; "529 have pooled" → "514 of 529"; replaced the stale "cosine-blind, embedding-side" conclusion with the accurate two-stage reading (Codex findings 2-4). |
| `eval/reports/2026-05-28_visual_eval_v4/{methodology.mdx,summary.json,per_example.jsonl}` | **Committed.** methodology.mdx reframed (two stacked failures; top-100→≥200 rerank window; recommendation points at the research matrix). summary.json + per_example.jsonl verified against the DB row and committed as-is. |
| `eval/reports/2026-05-28_visual_ablations/ablation_data.jsonl` | **Committed.** Truth source for the K-sweep; 30 rows (10×3), confirms the 720p reality. |
| `dev/active/phase-2-week-7/handoff.md` | **Committed.** v4 mid-session refresh; reviewed, consistent with verified facts. |

Discarded: nothing. Everything in v4 scope was verifiable and is kept.

## What was verified against the DB

```
visual-eval-0acd933a1c18: frame_recall=0.0556 chunk_tr=0.0556 answer_term=0.0 vag=0.0 n_eval=10
frames:  nXafozNIk3c 318/306   W_CYk2ogcDI 113/110   aie_sg_2026_d2_arize_alyx 98/98   (529 / 514 pooled)
frame_patches: 375,734   (223686 / 80410 / 71638)
frame PNGs: 1280×720 (sips, all 3 talks)
chunks: 212
```

## Commands run

```
make db-up && make db-migrate           OK (no pending migrations)
uv run pytest eval/tests/test_handlers_fast.py -q   14 passed (was 11; +3 new)
make validate-evals-strict              OK
make phase0-gate                        OK (29 verified / 3 talks)
make test                               145 passed, 201 deselected
make lint                               All checks passed
```

## What was NOT run (and why)

- `pytest -m slow eval/tests/test_run_visual_eval.py` — not required: no
  production code under `eval/runners/` changed this session (only
  `ingest/handlers.py`, `db/repos/frames.py`, a fast test, and report docs). The
  handler change only adds a clear-on-skip call on the NaN path, which does not
  touch the happy path the slow test exercises.
- No model was loaded. The research was web-only; the experiment is deferred to
  the next session.

## Decision: review done in main context, not TeamCreate

The code/doc fixes (NaN clear-on-skip, tests, report rewrites) were done in main
context because they are sequential edits needing verification and commits — the
project rule keeps final edits in main context. The research fan-out (5 category
researchers + audit-payload design + synthesis = 7 agents) ran as a background
workflow, since it is read-only web work with no memory contention. This satisfies
"parallelize the research, keep implementation in main context."

## Next command

```bash
# open the next session with prompts/prompt49.md, then:
git status --short --untracked-files=all
make db-up && make db-migrate
uv run python -m eval.runners.run_visual_rerank_probe --phase a --prefilter-k 500
```

## Files likely to change next

New `eval/runners/run_visual_rerank_probe.py`; an offline pooling function in
`embed/colqwen.py`; one `eval/config/model_candidates.yaml` row. If a positive
grounding row lands: the audit payload in `eval/runners/measure_visual.py` +
`run_visual_eval.py` per `audit_payload_design.md`. See `prompts/prompt49.md` for
the full plan, acceptance criteria, and rollback.

## Stop condition

Postgres stopped at session end (`make db-down`). The verified baseline to return
to is HEAD after the week-8 commits listed in the disposition table.
