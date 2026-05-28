# Handoff — Phase 2 Week 7 (visual eval v2)

Opened: 2026-05-28
Status: visual eval v2 shipped — text-side answer-term metric (`AnswerTermHit@k`) is live, lift is `0.0 pp` (visual channel does not help under either gate as currently measured). A real visual answer-grounding metric remains out of scope and is tracked under "Next steps".
Prior handoff: `dev/active/phase-2-week-6/handoff.md` (CLOSED at `7c84cff` / `31d0984`).

## What is true now

1. `eval/schemas/gold_example.schema.json` carries an optional `visual_evidence` array. The description explicitly states the metric runs on transcript text, not on frames. Three fixtures gate the conditional and the `minItems: 1` rule.
2. 10 visual-required rows in `visual_gold.jsonl` carry curator-confirmed `visual_evidence` terms.
3. `eval/runners/measure_visual.py` exposes `AnswerTermHit@k` primitives (`answer_term_matches_at_k`) + aggregator (`answer_term_hit_at_k`) + extended `visual_lift_at_k` reporting `lift_chunk_tr_pp` and `lift_answer_term_pp`. 14 new fast tests including the headline regression test for `adk-resumability-config-code`.
4. `eval/runners/run_visual_eval.py` loads `visual_evidence` and writes the text-side metric into `eval_runs.summary` + `eval_results.metrics`.
5. Run `visual-eval-b3e64cf97a6b` landed: `VisualFrameRecall@5 = 0.056`, `VisualChunkTR@5 = 0.056`, `AnswerTermHit@5 = 0.000` over 10 evaluable rows, `lift_chunk_tr_pp = -5.56`, `lift_answer_term_pp = 0.0`. (Across re-runs the headline counts shift by 1 row on n=18 — non-deterministic tie-breaking in patch retrieval. Treat `lift_chunk_tr_pp` between `[-5.56, 0]` as the noise band.)
6. Diagnostics report at `eval/reports/2026-05-28_visual_diagnostics/report.md` attributes failures: 5/10 lose at the pooled HNSW prefilter, 4/10 at MaxSim refine, 1/10 passes at @5.
7. ColQwen band check confirms the processor does no resize (frames sit inside [3,136 ; 602,112] pixels). It also flags an unrelated drift: on-disk frames are 360p (640×360), not 720p.

## What `AnswerTermHit@k` is, and is not

`AnswerTermHit@k` is a **text-side diagnostic**. It checks whether the surfaced transcript chunk's text contains the curator-supplied answer-bearing strings (`NDCG@10`, `is_resumable=True`, etc.). It does NOT validate the retrieved frame, does NOT OCR frames, does NOT call a visual judge. A real visual answer-grounding metric is out of scope and is the first item under "Next steps". The metric was renamed from `VisualAnswerHit` during a review fix to remove that overclaim.

## Headline numbers

| Metric | Value | Counts |
|---|---:|---|
| `VisualFrameRecall@5` (n=18) | 0.056 | 1/18 |
| `VisualChunkTR@5` (n=18) | 0.056 | 1/18 |
| `AnswerTermHit@5` (n_evaluable=10) | 0.000 | 0/10 |
| `lift_chunk_tr_pp` (n=18, RRF 5ch vs 4ch span overlap) | -5.56 | 17/18 vs 18/18 |
| `lift_answer_term_pp` (n_evaluable=10, RRF 5ch vs 4ch text-side) | 0.0 | 0/10 vs 0/10 |

Methodology writeup: `eval/reports/2026-05-28_visual_eval_v2/methodology.mdx`.

## Commands run this session

```
make db-up && make db-migrate          # OK, volume pgdata
make validate-evals-strict             # OK (5 valid + 15 invalid fixtures)
make phase0-gate                       # OK (29/3)
make test                              # 130 passed (was 116; +14 measure_visual tests)
make lint                              # clean
uv run python -m eval.runners.run_visual_eval
                                       # OK → visual-eval-b3e64cf97a6b
uv run python eval/reports/2026-05-28_visual_diagnostics/run_diagnostics.py
                                       # OK → rank_curves.jsonl + report.md
PYTHONPATH=. uv run python eval/reports/2026-05-28_visual_diagnostics/run_colqwen_band_check.py
                                       # OK → colqwen_band_check.md
uv run pytest -m slow eval/tests/test_run_visual_eval.py eval/tests/test_run_embeddings_bakeoff.py -q
                                       # OK (6 passed in 15:19) — re-run after rename
```

## What failed or was not run

- Full ColQwen ablation sweep (cadence × resolution × crop × prefilter_k) — deferred. The band check + diagnostics give enough to plan it; the sweep itself is a separate session.
- A real visual answer-grounding metric — explicitly deferred (see "Next steps"). Would need OCR over retrieved frames or a visual judge with logged outputs.

## Next command

```bash
# 1. Confirm clean working tree before staging.
git status --short

# 2. Commit in small chunks (see "Files likely to change next" below).
#    Note: keep .claude/CLAUDE.md deletion + CLAUDE.md at repo root OUT of these commits —
#    that relocation was already in the working tree at session start; if it
#    should land, do it as a separate, explicit commit reviewed on its own.
```

## Files likely to change next

In commit order:

1. **Schema + fixtures** — `eval/schemas/gold_example.schema.json` + three new files under `eval/tests/fixtures/`.
2. **Curated visual_evidence on visual_gold** — `eval/corpora/ai_engineering_v0/visual_gold.jsonl` (10 rows updated).
3. **Metric code + tests** — `eval/runners/measure_visual.py`, `eval/runners/run_visual_eval.py`, `eval/tests/test_measure_visual.py`.
4. **Diagnostic reports** — `eval/reports/2026-05-28_visual_diagnostics/*` + `eval/reports/2026-05-28_visual_eval_v2/*`.
5. **Design doc + handoff** — `dev/active/phase-2-week-7/agent_a_metric_design.md` + this file.

Bakeoff #2 (generator + judge) is the next session's scope, not this one.

## Open items

- **A real visual answer-grounding metric.** OCR retrieved frames or call a visual judge per (retrieved frame, expected visual claim) tuple with logged outputs. Until that lands, the visual eval gate is span overlap + a text-side answer-term diagnostic; visual answer support is not measured.
- 720p → 360p ingester drift (flagged in the band check). Check `ingest/frames.py` defaults + ffmpeg invocation against ADR 005 intent. Re-ingest if intent was 720p; re-run this eval; expect visual numbers to improve if slide text becomes legible.
- Stage-1 prefilter ablation (`prefilter_k = 200 → 500`) on the 5/10 visual-required rows that lose at stage 1.
- Stage-3 MaxSim diagnosis on the 3 rows where MaxSim ranks the gold-overlapping chunk at 93/94/111 of ~110 — near-last, not just slightly low.
