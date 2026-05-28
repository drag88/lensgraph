# v3 integrity check — executed 2026-05-28

Headline run audited: `visual-eval-239f681cbf69` (visual_eval, n=18, k=5).
This file records the actual command output for each check. Status: **all PASS** for the run as published.

## A. DB landing — PASS

Every new eval run has a row in `eval_runs` with `summary` populated, AND matching rows in `eval_results`.

```
('visual-eval-239f681cbf69', 'visual_eval', 2026-05-28, vag=0.0, vag_n=10, ath=0.0, frame_recall=0.0556, chunk_tr=0.0,   n_results=18)
('visual-eval-b3e64cf97a6b', 'visual_eval', 2026-05-28, vag=None, vag_n=None, ath=0.0, frame_recall=0.0556, chunk_tr=0.0556, n_results=18)
('visual-eval-0f1487296367', 'visual_eval', 2026-05-28, vag=None, vag_n=None, ath=None, frame_recall=0.0,   chunk_tr=0.0556, n_results=18)
```

- [x] Every row has non-null `summary`.
- [x] Every row has `n_results > 0` (18 each — matches visual_gold corpus size).
- [x] The headline run_id `visual-eval-239f681cbf69` appears as the most recent visual_eval row.
- [x] `vag` column is populated only on the new run (`visual-eval-239f681cbf69`); the two earlier runs predate the OCR metric and correctly have `vag=None`.

## B. No frame-side overclaim — PASS (trivially)

Audit query: rows where `metrics.visual_answer_grounding_at_k = true` for the headline run.

```
rows where vag_pass=true: 0
```

- [x] Zero rows claim `visual_answer_grounding = true`. No overclaim is possible because nothing claims a pass.

**Audit-payload gap — FLAGGED (required before relying on any future positive result):**

The current runner writes only the boolean `visual_answer_grounding_at_k` into `eval_results.metrics`. Section B was originally designed to spot-check a non-null OCR payload (`ocr_text`, `matched_term`, `evaluated_frame_id`) on every `vag_pass = true` row. Today no such payload exists. With 0 positive rows on this run the gap is inert; with any future positive row it becomes load-bearing.

**Required before interpreting any future positive `VisualAnswerGrounding` result:** `eval_results.metrics.visual_answer_grounding` must be extended to a structured payload, not a bare bool. Minimum fields:

- `passed: bool`
- `evaluated_frame_ids: [int]` — the in-window top-k frame IDs OCR was run against
- `evaluated_image_paths: [str]` — the on-disk PNG paths
- `ocr_excerpts: {frame_id: str}` — first ~500 chars of OCR per evaluated frame
- `matched_term: str | null` — which `required_any` term hit, or null
- `failure_reason: "no_in_window_frame" | "ocr_no_terms_matched" | "row_not_evaluable" | null`

A `TODO(v4)` marker is added to `eval/runners/measure_visual.py` and `eval/runners/run_visual_eval.py` (see the new comment blocks). The next session that runs visual_eval on 720p substrate must land this payload before claiming any visual-required row passed.

## C. Per-example claims back-reference per_example.jsonl — PASS

```
wc -l eval/reports/2026-05-28_visual_eval_v3/per_example.jsonl
      18 eval/reports/2026-05-28_visual_eval_v3/per_example.jsonl
```

- [x] 18 rows in `per_example.jsonl` matches `n_examples=18` in the methodology table.
- [x] Each row carries `frame_pass_at_k`, `chunk_pass_at_k`, `answer_term_hit_at_k`, `visual_answer_grounding_at_k` in `metrics`.
- [x] `system_output.top_k_frames` is populated per row so the methodology can quote retrieved frame IDs / paths.

## D. Per-channel slicing — PASS

The methodology comparison table separates `visual-required` (n=10) from `text-saturated` (n=8) and reports `n_evaluable` separately from `n_examples`.

- [x] `n_evaluable=10` for `VisualAnswerGrounding@k` matches the count of rows with `visual_evidence` in `visual_gold.jsonl`.
- [x] Text-saturated rows (8 in current corpus) carry `n/a` in the OCR and text-answer-term columns of the per-example table (not 0 or false).
- [x] Lift numbers in the headline table call out `n_evaluable=10` for the answer-term tier and `n=18` for the span-overlap tier, computed over their respective subsets.

## E. test_gold untouched — PASS

```
git log --since='2026-05-28' --name-only -- eval/corpora/ai_engineering_v0/test_gold.jsonl
# (empty)
shasum -a 256 eval/corpora/ai_engineering_v0/test_gold.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  eval/corpora/ai_engineering_v0/test_gold.jsonl
```

- [x] No commits touched `test_gold.jsonl` since session start.
- [x] File SHA is `e3b0c442...8855` — the well-known SHA256 of an empty file. `test_gold.jsonl` is the scaffolded zero-byte placeholder per phase-2 invariant.

## F. Gates pass — PASS (final pass before commit)

```
make validate-evals-strict     OK: corpora valid (1 corpus dir, 0 warnings, strict=True)
                               SELF-TEST OK: 5 valid + 15 invalid fixtures all behaved as expected
make phase0-gate               OK: 29 verified non-negative examples across 3 distinct talks (≥10/2)
make test                      142 passed, 201 deselected, 1.57s
make lint                      All checks passed!
pytest -m slow eval/tests/test_run_visual_eval.py    2 passed in 859.23s (0:14:19)
```

All five gates green.

## G. Recommendation backed by data — PASS as `(b) keep fixing`

Recommendation in `methodology.mdx`: **(b) keep fixing**. Defense:

- [x] One single blocker named: the window gate — ColQwen surfaces 0/10 visual-required rows within `gold_span ± 10s tolerance`. OCR is downstream and never reached.
- [x] The smallest experiment to decide is named with command: re-fetch + re-ingest at 720p (operator bumps `yt-dlp >= 2026.3.17`, then `uv run python -m eval.runners.run_visual_eval`).
- [x] If the 720p re-run is still 0/10, the second decision experiment is named: `uv run python eval/reports/2026-05-28_visual_ablations/run_ablations.py` (prefilter_k ∈ {200, 500, 1000} sweep). Script is staged + deferred per `DEFERRED.md`.

Note: pass criteria (a) is explicitly NOT met — `VisualAnswerGrounding@5 = 0.0`, `lift_answer_term_pp = 0.0`. No proceed claim is made anywhere in the report.

## H. Bakeoff #2 untouched — PASS

```
git diff main -- eval/reports/2026-05-27_embeddings_bakeoff/    # empty
git log main..HEAD --oneline -- eval/runners/measure_embeddings.py eval/runners/minimal_generation.py   # empty
```

- [x] No commits to bakeoff #1 / minimal_generation files since main.
- [x] No new code under `eval/runners/` for bakeoff #2 was touched.
