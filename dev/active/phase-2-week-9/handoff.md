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

---

# Session 2 close — VAG audit payload (prompt50) — DONE

Closed: 2026-05-30. HEAD now `4855f70`.

## Five-line read

1. The `VisualAnswerGrounding@k` per-row scalar is replaced by a structured JSONB
   audit sub-object `metrics.visual_answer_grounding` (dict | None). A positive row
   now records which frames were OCR'd, the OCR excerpt, and which curator term
   matched. The scalar key is gone (no shim); the run-level float stays in
   `eval_runs.summary`.
2. Eval re-ran on the unchanged fixed-loader substrate (`visual-eval-13edcf554e60`).
   Headline reproduced exactly: **frame_recall@5 17/18, chunk_tr@5 18/18,
   answer_term 0/10, VAG 2/10** — payload SHAPE changed, metric logic did not.
3. Both VAG positives were hand-verified by opening the named PNG:
   - `visual-required-sally-current-plan-block` → frame 995 (`frame_000035.png`,
     340s, in window) shows "Sort LLM spans by latency" and
     `todo_update(id=1, status="completed")`. **Survives.**
   - `visual-required-sally-lessons-recoverable-exceptions` → frame 1021
     (`frame_000061.png`, 600s, in window) shows "RecoverableExceptions create
     feedback loops". **Survives.**
   VAG 2/10 is now a defensible number, not a bare bool.
4. One honest caveat recorded in the report: positive #1 passed on a single frame
   of four — with `normalize: exact`, Tesseract read a curly `”` on frames
   992/993/994 and a straight `"` only on 995. The slide renders a straight quote,
   so the three misses are OCR artifacts, not 995 a false positive. If VAG is ever
   published, move that term off `exact` or note the fragility.
5. Next: bakeoff #2 (generator + judge) — but it is NOT shovel-ready. See the
   prerequisites block below and `prompts/prompt51.md`.

## Commands run (session 2)

```
uv run python -m eval.runners.run_visual_eval                 # visual-eval-13edcf554e60
uv run pytest eval/tests/test_measure_visual.py -q           # 54 passed
make test                                                    # 152 passed, 204 deselected
make validate-evals-strict / phase0-gate / lint              # all OK
uv run pytest -m slow eval/tests/test_run_visual_eval.py -q  # 2 passed in 678.98s
make db-down                                                 # Postgres stopped
```

## Commits (session 2)

- `d67200c feat(eval): VisualAnswerGrounding audit payload`
- `4855f70 eval: visual answer-grounding audit re-run (visual-eval-13edcf554e60)`

## What failed / was not run (and why)

- Nothing failed. Headline reproduced exactly as required.
- `frame_ocr_matches_at_k` is now production-uncalled (the aggregator moved to
  `frame_ocr_grounding_at_k`). Left as a tested public bool primitive, consistent
  with the module's per-primitive-window-check style. Candidate for deletion or
  delegation in a future cleanup — flagged, not silently kept.
- Did NOT touch the loader, did NOT re-ingest, did NOT start bakeoff #2.

## Prerequisites blocking bakeoff #2 (read before opening prompt51)

Bakeoff #2 as written in `model_candidates.yaml` cannot run end-to-end today:

- `eval/corpora/ai_engineering_v0/boundary_audit.jsonl` is **empty (0 rows)** →
  the judge minimum `cohens_kappa_vs_human_min: 0.60` cannot be computed → no
  kappa-validated judge selection.
- `eval/corpora/ai_engineering_v0/negative.jsonl` is **empty (0 rows)** → the
  generator minimum `corpus_negative_refusal_min: 0.90` cannot be measured.
- `eval/runners/judges/` is **empty** → no versioned, content-hashed judge prompt.
- `generate/` + `eval/runners/{providers,minimal_generation}.py` carry
  **uncommitted WIP** (pre-existing dirty tree) that must be triaged before a clean
  generator run. `dev_gold.jsonl` has 10 rows (the generator eval set).

These are curation + scaffolding tasks, not just "run the bakeoff." prompt51
sequences them.

## Next command

```bash
# open the next session with prompts/prompt51.md, then:
git log --oneline -4
make db-up && make db-migrate
git diff --stat generate/ eval/runners/providers.py eval/runners/minimal_generation.py
wc -l eval/corpora/ai_engineering_v0/{dev_gold,negative,boundary_audit}.jsonl
```

## Files likely to change next (bakeoff #2)

`eval/corpora/ai_engineering_v0/{negative,boundary_audit}.jsonl` (curation),
`eval/runners/judges/<judge>.md` (new judge prompt), a new generator-bakeoff
runner under `eval/runners/` (analogous to `run_embeddings_bakeoff.py`),
`generate/*` (triage the WIP), `docs/decisions/004-*` (selection result amendment),
`eval/reports/<date>_generator_bakeoff/`.

## Stop condition (session 2)

Postgres stopped (`make db-down`, confirmed `docker compose ps postgres` empty).
Verified baseline to return to is HEAD `4855f70`. `test_gold.jsonl` untouched.

---

# Session 3 close — bakeoff #2 generator + judge (PROVISIONAL)

Closed: 2026-05-31. HEAD now `84b8257`.

## Five-line read

1. The "answer-loop WIP" was NOT logic — it was 100% `ruff format` reflow on 8
   files (HEAD failed `ruff format --check`; the working tree was the formatted
   output). Committed as a scoped `chore:` (`44f09c7`). `make answer` runs
   end-to-end against DeepInfra (verified on a dev_gold query).
2. Curated 20 corpus-scope negatives (`negative.jsonl`, all `verified: true`),
   each confirmed out-of-corpus by grepping the three transcripts. Wrote a
   content-hashed cross-family judge prompt (`eval/runners/judges/faithfulness_v1.md`,
   sha `3fb7316ac981…`).
3. Built the generator bakeoff: `eval/runners/measure_generation.py` +
   `scripts/run_generator_bakeoff.py` + 14 fast stub tests. Ran ONE real
   provisional run (judge `deepseek-v3.2`; generators gemma-4-31b + qwen3-235b;
   deepseek excluded as generator — shares judge family).
4. Result: faithfulness saturates (claims 1.0, citation 1.0, parse 1.0 for both)
   — ceiling effect on 10 easy dev_gold. Discriminator is abstention: qwen
   refuses 20/20 corpus-negatives, gemma 17/20. Both fail the 2.0s p95 latency
   minimum (40.7s / 66.4s) → **NO winner**. Provisional: no lock, no ADR amendment.
5. Why provisional, not selected: `boundary_audit.jsonl` is empty and judge↔human
   kappa needs HUMAN labels (cannot be fabricated). Plus a tier mismatch:
   `boundary_audit` scores the boundary tier (edge/standalone) but this judge
   scores the answer tier (claims/citation) — kappa on boundary_audit does not
   validate the faithfulness judge.

## Commands run (session 3)

```
git diff generate/ eval/runners/{providers,minimal_generation}.py   # all formatter reflow
make answer QUERY="..." GENERATOR=gemma-4-31b JUDGE=deepseek-v3.2    # end-to-end OK
make validate-evals-strict / phase0-gate / lint                     # all OK (29/3)
make test                                                           # 166 passed (+14)
uv run python -m scripts.run_generator_bakeoff                      # 2 rows, no winner
make db-down                                                        # Postgres stopped
```

## Commits (session 3)

- `44f09c7 chore: ruff format answer-loop and provider modules`
- `9aeb7b7 eval: add 20 corpus-scope negatives for abstention metric`
- `de0466d eval: add content-hashed cross-family faithfulness judge prompt`
- `c45b749 feat(eval): generator bakeoff #2 runner + faithfulness measurement`
- `7af8f9e eval: provisional generator bakeoff #2 results (no winner)`
- `84b8257 docs(handoff): week-9 session-3 close — provisional bakeoff #2`

## What failed / was not run (and why)

- `boundary_audit.jsonl` stays EMPTY. Judge↔human kappa needs human edge/standalone
  labels; an AI agent cannot validly produce the human side. Generator faithfulness
  is therefore PROVISIONAL, not a published claim. No ADR 004 amendment.
- The 2.0s `p95_latency_sec_max` minimum is almost certainly mis-scoped for the
  non-streaming JSON-decomposition harness (real p95 40–66s). Flagged, not changed.
- Faithfulness saturated at 1.0/1.0 — `dev_gold` (n=10, mostly easy) is too small
  and too easy to separate generators. Needs growth + `hard` examples.
- One real run only; the runner's DB persistence is covered by that run + the fast
  stub tests (no slow DB/API test added this session).

## Next command

```bash
git log --oneline -6
make db-up && make db-migrate
# Then, to unblock a real (non-provisional) bakeoff #2 selection:
#  1. curate boundary_audit.jsonl WITH human edge/standalone labels (needs Aswin)
#  2. resolve the answer-tier vs boundary-tier judge mismatch
#  3. recalibrate p95_latency_sec_max; grow dev_gold + add hard examples
```

## Files likely to change next (bakeoff #2 → real selection)

`eval/corpora/ai_engineering_v0/boundary_audit.jsonl` (human labels),
`eval/config/model_candidates.yaml` (`p95_latency_sec_max` recalibration; possibly a
new answer-tier audit minimum), `eval/runners/measure_generation.py` (kappa wiring),
`docs/decisions/004-*` (amend only once the judge is validated AND a winner qualifies),
`dev_gold.jsonl` (grow + harder examples).

## Stop condition (session 3)

Postgres stopped (`make db-down`). Verified baseline to return to is HEAD `84b8257`.
`test_gold.jsonl` untouched. The two provisional run rows are in `eval_runs`
(`code='generator_bakeoff'`); no winner is locked, so
`load_bakeoff_winner('generator', ...)` still returns `None`.

## Post-close review fixes (2026-06-01)

A review pass added two commits ON TOP of the `84b8257` close — the current
tip is past `84b8257`, so run `git log --oneline -6` to get the real HEAD:

- `589474e docs(handoff): correct session-3 close to HEAD 84b8257`
- `c10fe7d fix(eval): strict judge booleans + missing-latency fails minimum`
  - judge parsing now credits ONLY a JSON literal `true` (`is True`);
    `"true"`/`"false"`/`1`/`0`/null/missing get no credit (can deflate, never
    inflate faithfulness).
  - missing/None `p95_latency_ms` now FAILS `p95_latency_sec_max` instead of
    coercing to 0ms.
  - +5 fast tests; `test_run_generator_bakeoff.py` now 19 passed; `make test` 171.

Provisional bakeoff results are UNCHANGED (the real run's judge emitted proper
JSON booleans, `judge_failures=0`). No live rerun. This addendum is itself a
later commit; treat the branch tip from `git log` as the baseline.

## Rolling-caption VTT dedup fix + re-chunk (2026-06-01)

Setting up the `boundary_audit` worksheet surfaced a corpus-wide chunk-text
bug: opening a clip at 17s did not match its stored text. Root cause in
`chunking/fixed_window.py::_parse_vtt`: YouTube roll-up captions repeat each
line as residue, and the blank-line flush dropped the freshly-spoken cues and
kept only the time-shifted residue — so chunk text was triplicated AND
~1 cue early. Fix: blank lines are in-cue padding (cues end at the next
timestamp), `_normalize_line` strips tags + html-unescapes + drops `>>`, and
`_dedup_rolling` removes repeated leading lines. Commits `0ea9686` (fix + 5
TDD cases), `f9c8806` (doc/report refresh).

Re-chunked + re-embedded all 3 talks (delete-then-`chunk_handler`, drain
`ingest_embed_text`): **212 → 213 chunks** (W 45→46; arize 39; google 128),
every channel aligned (chunks = dense = sparse = token = tsv per video).
"Thanks for coming" now appears once across W chunks (was triplicated).
Boundaries shifted (recovering dropped active cues), re-verified by re-running
the embeddings bakeoff on the clean substrate: **`embeddings-bakeoff-ada840be66b9`**,
winner unchanged (`bge-m3-all-channels`, vector-best 1.00), `rrf_4ch`
**0.90 → 1.00**, `dense`/`multivec` 1.00, `sparse` 0.70.

Regenerated the boundary_audit worksheet from clean chunks
(`dev/active/phase-2-week-9/boundary_audit.{worksheet.md,_labels.csv}`,
untracked) — ready for human edge/standalone labels. The provisional generator
bakeoff (212-chunk substrate) is annotated; re-run it on the 213-chunk
substrate alongside the kappa-validated judge before any selection.

Gates green: `make validate-evals-strict` / `phase0-gate` (29/3) / `make test`
(176) / `make lint`. Postgres stopped. Current HEAD: see `git log` (past
`f9c8806`).

## Boundary audit + transcript_segment (2026-06-01, provisional)

Aswin hand-labeled 21 fixed_window clips (edge_sensibility + standalone, 1–5).
Findings: (a) a worksheet display bug — the arize chapter-slice showed
talk-relative spans while the watch link used absolute video time (+516s); fixed
by showing absolute spans. (b) The real signal: fixed_window cuts mid-sentence,
worst on continuous-speech talks.

**Boundary judge** `eval/runners/judges/boundary_v1.md` (sha `f7369216b385`),
run by `deepseek-v3.2`, scored the same clips blind. First kappa was negative
(judge scored mid-sentence as an EDGE problem; the human had scored it as a
STANDALONE problem — a rubric mismatch). After aligning the rubric (mid-sentence
→ lower edge) and Aswin re-judging the edge dimension:

- **edge kappa = 0.808 quadratic-weighted / 0.624 linear / 0.755 binarized**
  (n=13), within-1-point on every clip. **Clears the 0.60 gate.**
- **CAVEAT (load-bearing): PROVISIONAL.** Aswin accepted all 13 of my
  rubric-derived edge suggestions verbatim, so the human labels are not
  independent of an LLM applying the same rubric — this shows the judge applies
  the rubric *consistently*, not that it matches a *cold* human. standalone is
  still unvalidated (uniform labels → kappa ≈ 0.07). **Do NOT publish a
  boundary/chunking claim on this kappa.** A cold labeling pass (fresh ~20 clips,
  both dimensions, no suggestions shown) is required first — queue as its own
  task. `boundary_audit.jsonl` was NOT committed (data not publication-grade).

**transcript_segment** (`chunking/transcript_segment.py`, registered) is the
sentence-aligned strategy that answers the mid-sentence finding. VTT parsing was
extracted to `chunking/vtt.py` (shared by both strategies). Dry boundary-quality
comparison (clean-both-edges): google 7%→92%, arize 18%→100%, tengyu 2%→38%
(tengyu limited by sparse transcript punctuation). Tradeoff: longer/fewer chunks
(google 128→50, avg 30s→63s) — coarser retrieval granularity that the ablation
must weigh against the cleaner edges. Commits `fee637b` (vtt extract), `0c30d0b`
(transcript_segment). 184 tests pass.

**Next (the chunking ablation):** chunk + embed the 3 talks under
`transcript_segment` (chunks table keys on `chunking_strategy`, so both coexist),
then compare fixed_window vs transcript_segment on the SAME dev_gold under the
same retrieval/generation settings — TR@5 recall (does coarser granularity hurt?)
+ boundary edge/standalone (validated judge, after the cold kappa pass). That
comparison is the resume-grade chunking methodology. All of it stays provisional
until the cold kappa lands.

Gates green: `make test` (184) / `lint` / `validate-evals-strict` / `phase0-gate`
(29/3). Postgres stopped. HEAD past `0c30d0b` (see `git log`).
