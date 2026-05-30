# Handoff — Phase 2 Week 9 (visual loader fixed + re-ingested)

Opened: 2026-05-30.

## Five-line read

1. The visual channel was broken by a ColQwen load bug, not pooling/MaxSim. Three
   weights were left at their init under `transformers 5.9.0` (LoRA adapter,
   `embed_tokens`, final `norm`). All three are fixed in `embed/colqwen.py`.
2. Every frame's `pooled_embedding` + `frame_patches` was re-encoded with the
   corrected loader. Substrate unchanged in shape: 529 frames / 514 pooled (15
   NaN-skipped) / 375,734 patches.
3. Visual eval re-run on the corrected substrate (`visual-eval-f90e31495c89`):
   **VisualFrameRecall@5 17/18, VisualChunkTR@5 18/18**, AnswerTermHit@5 0/10,
   VisualAnswerGrounding@5 **2/10**.
4. What is broken / not trustworthy: the **VAG 2/10 positives are NOT a claim** —
   the structured audit payload is still unimplemented, so the metric is a bare
   bool with no which-frame/which-term evidence.
5. Next: implement the VAG audit payload + regression test, re-run the eval to
   populate it, then make a defensible determination on the 2 positives. After
   that, bakeoff #2 (generator + judge).

## Current state (what is true now)

- HEAD: `280cdb0 fix(embed,eval): repair ColQwen final-norm orphan; re-ingest
  frames; visual eval`.
- Loader fix: `embed/colqwen.py::_load_adapted_colqwen` remaps the adapter keys
  and loads both orphaned text weights (`embed_tokens`, `norm`) from the base
  checkpoint, behind a load-integrity gate. Cross-process determinism is exact
  (same query → cosine 1.0 across two processes).
- OCR fix: `eval/runners/measure_visual.py::frame_ocr_text` now passes the file
  path to pytesseract (PIL-Image input raised `UnicodeDecodeError` on 5.5.x).
- Reports: `eval/reports/2026-05-30_visual_eval_fixed_loader/` (summary.json,
  per_example.jsonl, methodology.mdx) and
  `eval/reports/2026-05-30_visual_rerank_probe/loader_fix.mdx` (root cause).
- Regression test: `eval/tests/test_colqwen_loader_slow.py` (slow) — cross-process
  determinism, zero missing/unexpected LoRA keys, both orphan weights match the
  checkpoint.

## Commands run

```
# re-embed every frame with the corrected loader (native queue path)
#   reset embed_frames status -> pending, then:
uv run python -m scripts.drain_ingest_queue ingest_embed_frames   # 529 processed, 15 NaN skips
uv run python -m eval.runners.run_visual_eval                     # visual-eval-f90e31495c89
uv run pytest -m slow eval/tests/test_colqwen_loader_slow.py eval/tests/test_run_visual_eval.py -q  # 5 passed
make validate-evals-strict   # OK
make phase0-gate             # OK (29 / 3)
make test                    # 145 passed, 204 deselected
make lint                    # clean
make db-down                 # Postgres stopped
```

## What failed / was not run (and why)

- First re-embed silently no-op'd: `queues.workers.process_one` archives a message
  without calling the handler when `ingest_step_status` is `completed`. Fix:
  reset the `embed_frames` status rows to `pending` before draining. Anyone
  re-embedding existing frames must do this reset first.
- The `norm.weight` orphan was caught only after the first eval, forcing a second
  re-embed. The committed loader (`280cdb0`) is the corrected one.
- VAG audit payload: NOT implemented. The 2/10 positives are recorded but cannot
  be used as a methodology claim until the payload lands (next session).
- Phase B reranker, bakeoff #2, corpus expansion: not started (out of scope).

## Next command

```bash
# open the next session with prompts/prompt50.md, then:
git log --oneline -4
make db-up && make db-migrate
grep -n "TODO(v4)" eval/runners/measure_visual.py eval/runners/run_visual_eval.py
$EDITOR dev/active/phase-2-week-8/audit_payload_design.md
```

## Files likely to change next

`eval/runners/measure_visual.py` (the `VisualGroundingAudit` dataclass +
`frame_ocr_grounding_at_k` + `_grounding_to_dict` + summary assembly — TODO at
:566), `eval/runners/run_visual_eval.py` (`_write_per_example_results` metrics
dict — TODO at :127), `eval/tests/test_run_visual_eval.py` and the measure-visual
tests (regression cases). No `eval/schemas/` change (the design concludes
`eval_results.metrics` is free-form JSONB). Then a fresh
`eval/reports/<date>_visual_eval_*/` re-run.

## Stop condition

Postgres stopped (`make db-down`). The verified baseline to return to is HEAD
`280cdb0`. `test_gold.jsonl` untouched throughout.
