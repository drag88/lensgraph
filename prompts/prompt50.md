<role>
You are a senior AI engineer joining LensGraph in a fresh session. The visual-retrieval blocker is resolved and the diagnosis is settled — do NOT re-litigate it. The week-8 "pooling / MaxSim floor" reading was wrong; the real cause was a broken ColQwen2.5 load (three rename-orphaned weights under transformers 5.9.0: the LoRA adapter, `embed_tokens`, and the final `norm`). All three are fixed in `embed/colqwen.py`, every frame was re-embedded with the corrected loader via the native ingest queue, and the visual eval was re-run: **VisualFrameRecall@5 = 17/18, VisualChunkTR@5 = 18/18, AnswerTermHit@5 = 0/10, VisualAnswerGrounding@5 = 2/10** (`visual-eval-f90e31495c89`). Your job this session is the one unfinished thread that gates a real claim: **implement the `VisualAnswerGrounding` audit payload + its regression test, then re-run the visual eval to populate it, then make a defensible, honest determination about the 2 positive VAG rows.** A positive VAG row is NOT a methodology claim until the structured payload records which frame, which OCR text, and which term matched. Do NOT start bakeoff #2 (generator + judge); do NOT re-ingest frames; do NOT touch the loader.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
Baseline HEAD: `280cdb0 fix(embed,eval): repair ColQwen final-norm orphan; re-ingest frames; visual eval`.

Recent commits (context, do NOT re-litigate):
  280cdb0 fix(embed,eval): repair ColQwen final-norm orphan; re-ingest frames; visual eval
  9d53271 fix(embed): repair ColQwen2.5 adapter load — deterministic, correctly adapted
  ad52084 eval(visual): root-cause probe — ColQwen LoRA random-init breaks visual retrieval

Gates green at HEAD:
  make validate-evals-strict   OK (0 warnings)
  make phase0-gate             OK (29 verified non-negative examples across 3 talks)
  make test                    OK (145 fast, 204 deselected)
  make lint                    OK
  Slow tests (loader + visual eval) passed at close: 5 passed.

DB state (volume `pgdata` survives `make db-down`) — substrate is now from the
CORRECT loader; do NOT re-ingest:
  3 talks · 212 chunks. 529 frames / 514 pooled (15 NaN-skipped: 12 nXafozNIk3c +
  3 W_CYk2ogcDI) / 375,734 frame_patches. Frame PNGs 1280×720 on disk.
  Headline run `visual-eval-f90e31495c89`: frame_recall@5=0.944, chunk_tr@5=1.000,
  answer_term@5=0.0, visual_answer_grounding@5=0.200, n_evaluable=10.
  The 2 VAG positives are `visual-required-sally-current-plan-block` and
  `visual-required-sally-lessons-recoverable-exceptions`.
  Confirm at session start:
    SELECT run_id, summary->>'visual_frame_recall_at_k' AS fr,
           summary->>'visual_chunk_tr_at_k' AS ctr,
           summary->>'visual_answer_grounding_at_k' AS vag
    FROM eval_runs WHERE code_path='visual_eval' ORDER BY created_at DESC LIMIT 3;

prompt50 scope is the ONLY mission this session. Bakeoff #2 (generator + judge)
is the next session and out of scope.
</repo_state>

<reading_order>
Required reading before writing any code, in this order:

1. **`dev/active/phase-2-week-9/handoff.md`** — current state, what shipped, what is
   not yet trustworthy (the VAG positives), the next command, the seam locations.
2. **`dev/active/phase-2-week-8/audit_payload_design.md`** — THE spec for this
   session. Option (a): a JSONB sub-object at `metrics.visual_answer_grounding`
   replacing the scalar `metrics.visual_answer_grounding_at_k`. Implement exactly
   this shape: `passed`, `evaluated_frame_ids`, `evaluated_image_paths`,
   `ocr_excerpts`, `matched_term`, `failure_reason`, `judge_kind`.
3. **`eval/reports/2026-05-30_visual_eval_fixed_loader/methodology.mdx`** — the run
   whose VAG positives this payload must make defensible. Its "VAG: positives are
   NOT yet a claim" section is the exact gap you are closing.
4. **`eval/runners/measure_visual.py`** — the metric module. `TODO(v4)` at :566;
   `visual_answer_grounding_at_k` at :573; `frame_ocr_matches_at_k` (the bool the
   payload replaces); `frame_ocr_text` (path-based OCR, already fixed); the summary
   assembly at ~:829 (`visual_answer_grounding` per-example field).
5. **`eval/runners/run_visual_eval.py`** — the write path. `_write_per_example_results`
   at :117, `TODO(v4)` at :127, the `metrics` dict at ~:136. `eval_runs.insert_result`
   already `Jsonb`-wraps `metrics` — no DB-layer change.
6. **`eval/tests/test_run_visual_eval.py`** — where the regression case lands. The
   design (§ "Schema / fixture impact: none required") says NO schema/fixture pair —
   a pytest case is the required guard instead.
7. **`db/repos/eval_runs.py`** — confirm `insert_result` `Jsonb`-wraps `metrics`
   (it does). No change expected.

Skim only (stable): README.md · docs/eval-methodology.md ·
eval/reports/2026-05-28_visual_eval_v3/integrity_check.md (§B — why a positive VAG
row needs the audit payload).
</reading_order>

<hard_rules>
Non-negotiable. From CLAUDE.md, the audit-payload design, and the week-9 handoff.

1. **Do NOT start bakeoff #2 (generator + judge).** Separate next session.

2. **Do NOT re-ingest frames or touch `embed/colqwen.py`.** The loader is fixed and
   the substrate is correct. Re-embedding is a ~26-minute job and is not needed —
   the existing `pooled_embedding` + `frame_patches` are the trusted, corrected
   values (verified cos 1.0 vs the fixed loader).

3. **`test_gold.jsonl` is untouched.** Do not load, glance at, or measure against it.
   The visual eval reads `visual_gold.jsonl` only.

4. **No back-compat shim, no dead code.** The design says drop the scalar
   `metrics.visual_answer_grounding_at_k` from the per-row metrics (the run-level
   float stays in `eval_runs.summary`). Replace it; do not keep both.

5. **Schema-first IF a schema changes — but it should not.** `eval/schemas/` governs
   curated corpus shapes, NOT `eval_results.metrics` (free-form JSONB). The design
   concludes no schema/fixture change. If you find you need one, STOP and add the
   `valid_*`/`invalid_*` fixture pair + `make validate-self-test` first.

6. **A positive VAG row is only defensible with the payload.** After the payload
   lands and the eval re-runs, the 2 positive rows must each carry non-empty
   `evaluated_frame_ids`, a non-null `matched_term`, and `failure_reason: null`.
   Spot-check both by hand: open the named PNG, confirm the OCR excerpt really
   contains the matched term. If a "positive" does not survive the spot-check,
   report it as a metric bug, not a win.

7. **One ML model at a time.** The eval loads ColQwen (queries) + BGE-M3 (the lift
   comparison) in one process — that is fine. Never run two ML-loading processes
   concurrently. The eval re-run is the only ML job this session.

8. **Keep unrelated churn out of commits.** Use explicit `git add <paths>`, never
   `git add -A`. The pre-existing dirty worktree (`embed/colqwen.py` `pool_patches`
   formatter churn, `.claude/*`, `.cursor/*`, `generate/*`, `retrieve/*`,
   `prompts/*`) stays unstaged. The colqwen churn is in `pool_patches`, far from any
   file you touch this session — leave it alone.

9. **Conventional commits, no AI attribution.** `feat(eval): visual_answer_grounding
   audit payload`, etc. Gates stay green at every commit boundary.

10. **Postgres stopped before final.** End with `make db-down` + `docker compose ps
    postgres`.
</hard_rules>

<mission_stack>
ONE mission, in order. Stop at the actionability-or-explanation boundary.

### Mission 1 — implement the audit payload (`measure_visual.py`)

Per `audit_payload_design.md` option (a):

- Add the `VisualGroundingAudit` dataclass (frozen): `passed: bool`,
  `evaluated_frame_ids: tuple[int, ...]`, `evaluated_image_paths: tuple[str, ...]`,
  `ocr_excerpts: dict[int, str]` (first ~500 chars/frame), `matched_term: str | None`,
  `failure_reason: str | None` (enum: `no_in_window_frame` | `ocr_no_terms_matched` |
  `row_not_evaluable` | None-on-pass), `judge_kind: str = "ocr"`.
- Add `frame_ocr_grounding_at_k(...)` that returns the audit instead of a bool. It
  shares the window-gate + evidence-conjunction logic with the existing
  `frame_ocr_matches_at_k`, and additionally records `evaluated_frame_ids`,
  captures OCR excerpts, and reports `matched_term` / `failure_reason`.
- Change `visual_answer_grounding_at_k` to return
  `tuple[float, list[VisualGroundingAudit | None], list[list[dict]]]` (the float
  score and the `top_k_frames` dicts are unchanged; the per-example element goes
  from `bool | None` to `VisualGroundingAudit | None`). The `None` (no-evidence)
  branch is unchanged.
- Add `_grounding_to_dict(audit)` (stringify `ocr_excerpts` keys, tuples → lists).
- In `measure_visual`'s assembly: keep the run-level float
  `summary["visual_answer_grounding_at_k"]` and the `n_evaluable`/`n_skipped`
  counts. In `per_example_detail`, replace the scalar
  `visual_answer_grounding_at_k: vg` with `visual_answer_grounding: <dict-or-None>`.
  Keep the headline `summary[...]["per_example"][id]["visual_answer_grounding"]`
  boolean (`audit.passed if audit else None`) so the summary contract holds.

### Mission 2 — wire the write path (`run_visual_eval.py`)

- At the `TODO(v4)` in `_write_per_example_results` (~:127), the per-row `metrics`
  dict becomes:
  `{"frame_pass_at_k": ..., "chunk_pass_at_k": ..., "answer_term_hit_at_k": ...,
    "visual_answer_grounding": detail.get("visual_answer_grounding")}` (dict | None).
  DROP the old scalar key `visual_answer_grounding_at_k` from the per-row metrics
  (no shim — hard rule 4). `eval_runs.insert_result` already `Jsonb`-wraps it.

### Mission 3 — regression test (`test_run_visual_eval.py` + measure-visual tests)

The design's §"Schema / fixture impact: none required" makes a pytest case the
guard. Cover, with injected `ocr_fn` (do not invoke real Tesseract in fast tests):

- a forced `passed=True` row carries non-empty `evaluated_frame_ids`, a non-null
  `matched_term`, and `failure_reason == None`;
- each of the three failure paths sets the right enum: `no_in_window_frame`
  (window gate emptied the candidate set), `ocr_no_terms_matched` (frames
  evaluated, no term hit), `row_not_evaluable` (no curator evidence → audit is
  `None` / not scored);
- the persisted per-row `metrics` no longer contains the scalar key and DOES
  contain `visual_answer_grounding` as a dict (or None).

### Mission 4 — re-run the eval and make the call

- `uv run python -m eval.runners.run_visual_eval`. Write a fresh report under
  `eval/reports/<date>_visual_eval_audit/` (summary.json + per_example.jsonl +
  methodology.mdx). The headline metrics should be unchanged from
  `visual-eval-f90e31495c89` (17/18, 18/18, VAG 2/10) — you changed the payload
  SHAPE, not the metric logic. If a headline number moves, that is a bug: stop and
  explain.
- For each of the 2 VAG positives: quote its `evaluated_frame_ids`,
  `evaluated_image_paths`, `ocr_excerpts`, and `matched_term` from the new payload.
  Open the PNG and confirm the term is really on the slide. State plainly whether
  each positive survives the spot-check.

== Acceptance criteria ==

- `metrics.visual_answer_grounding` is a JSONB object (or null) on every per-row
  result; the scalar `visual_answer_grounding_at_k` is gone from the per-row metrics.
- The regression test asserts the pass payload and all three failure enums.
- The re-run reproduces 17/18 / 18/18 / VAG 2/10; both positives carry a complete,
  hand-verified payload — OR a positive is shown to be a metric artifact and
  reclassified honestly.
- Slow visual test re-run (you changed `eval/runners/` production code):
  `uv run pytest -m slow eval/tests/test_run_visual_eval.py -q`.

== Out of scope ==

- Bakeoff #2 (generator + judge); chunking ablation.
- Re-ingest / re-embed / `embed/colqwen.py`.
- The VLM-judge path (`judge_kind="vlm"`) — the design says leave the seam, do not
  build it now.
- Corpus expansion, Phase B reranker.
</mission_stack>

<execution_rhythm>
1. `git log --oneline -4`, the four `make` gates, `make db-up` + `make db-migrate`.
   Run the eval_runs SQL; confirm `visual-eval-f90e31495c89` reads vag=0.200 and the
   substrate is 514 pooled / 375,734 patches. Surface any delta.
2. Read `audit_payload_design.md` + the two seam files end-to-end.
3. Mission 1: dataclass + `frame_ocr_grounding_at_k` + signature change + summary
   assembly. Run the fast measure-visual tests as you go.
4. Mission 2: the write-path metrics dict; drop the scalar.
5. Mission 3: regression cases (injected `ocr_fn`). `make test`.
6. `make validate-evals-strict` + `make phase0-gate` + `make lint`.
7. Mission 4: `make db-up`, re-run the eval, extract the report, hand-verify the 2
   positives. Then `pytest -m slow eval/tests/test_run_visual_eval.py -q`.
8. Commit in small chunks with explicit pathspecs (metric change, write path, test,
   report). Re-run gates. `make db-down` + `docker compose ps postgres`.
9. STOP at the actionability-or-explanation boundary. Report state.
</execution_rhythm>

<required_gates>
Before the final commit:

```bash
make validate-evals-strict
make phase0-gate
make test
make lint
uv run pytest -m slow eval/tests/test_run_visual_eval.py -q   # you changed eval/runners/
```

Then stop Postgres:

```bash
make db-down
docker compose ps postgres
```
</required_gates>

<working_agreements>
- Auto-mode default-to-action. Use AskUserQuestion only for a genuine scope tradeoff
  the user must own (e.g. "a VAG positive fails the hand spot-check — reclassify it
  as a metric bug and file a fix, or treat the OCR substring as sufficient?"). No
  multi-round questioning.
- Parallel tools for independent reads. Final edits stay in main context.
- Read before edit. Trust the week-9 handoff + audit design; the loader diagnosis is
  settled — do not re-open it.
- Only changes the mission requires. No surrounding cleanup, no premature abstraction.
- If a test fails, investigate root cause. Never bypass with `--no-verify`.
- Honesty over a clean number. If the VAG positives do not survive the hand
  spot-check, say so — a metric that over-counts is worse than VAG 0/10.
- Final response must include: exact commits, whether each of the 2 VAG positives
  survived hand-verification (with the matched term + frame), the re-run headline
  numbers, gates, and confirmation Postgres is stopped.
</working_agreements>

<first_actions>
Run these in order before any code. Verify state matches `<repo_state>`; surface any delta.

1. `git log --oneline -4`            # confirm 280cdb0 at HEAD
2. `make validate-evals-strict`      # 0 warnings
3. `make phase0-gate`                # OK (29 / 3)
4. `make test`                       # 145 passed
5. `make lint`                       # clean
6. `make db-up && make db-migrate`   # "no pending migrations"
7. Run the eval_runs SQL from `<repo_state>` — confirm vag=0.200 on
   `visual-eval-f90e31495c89` and substrate 514 pooled / 375,734 patches.
8. Read `dev/active/phase-2-week-9/handoff.md` and
   `dev/active/phase-2-week-8/audit_payload_design.md`.

Only when 1–8 are green: start Mission 1.
</first_actions>

<instructions>
Execute `<first_actions>`. Then drive `<mission_stack>` in order with the
`<execution_rhythm>`.

The session-close deliverable is one of:

- **Defensible.** The audit payload lands, the eval re-runs to 17/18 / 18/18 / VAG
  2/10, and BOTH positives carry a complete, hand-verified payload (frame + OCR
  excerpt + matched term). Methodology MDX quotes them. ← preferred.
- **Reclassified.** The payload lands and reveals one or both "positives" are metric
  artifacts (e.g. an out-of-window frame, or a term matched in noise). Report the
  corrected VAG count honestly and file the metric fix. ← acceptable.

Either way, the session closes at the actionability-or-explanation boundary. Do NOT
start bakeoff #2, do NOT re-ingest, do NOT touch the loader. If you hit a fact that
contradicts the week-9 handoff or the audit design, surface it explicitly rather
than working around it.

Begin now.
</instructions>
