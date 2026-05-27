# Handoff — Phase 2 Week 6 (Bakeoff #1: text embeddings) — CLOSED

**Opened:** 2026-05-27 (phase-1 week-5 closed at `8d6694a`)
**Closed:** 2026-05-27 — bakeoff #1 locked, all gates green; visual eval scaffolded.
**Status:** ✅ Bakeoff #1 (text embeddings) shipped. `bge-m3-all-channels` locked as the text-embeddings winner. Best-of vector channel TR@5 = 1.00 against the ADR 004 v3.1 minimum (≥ 0.75). No ADR amendment required. ⚪ Visual retrieval (ColQwen2.5 — the 5th channel) has its own eval gate (`eval_runs.code_path = 'visual_eval'`) — infrastructure landed this session as a STUB; the gate runs once verified visual gold + ingested ColQwen frames/patches exist. Bakeoff #1 did NOT measure or lock visual retrieval.
**Authoritative design:** `docs/phase-1-design.md` rev 4 (commit `e950583`) — still the contract.
**Prior handoff:** `dev/active/phase-1-week-5/handoff.md` (CLOSED week 5).
**Next:** Bakeoff #2 (generator + judge) — week 7, **not started**. Out of scope for this handoff.

---

## Outcome — Bakeoff #1 (text embeddings only)

| Channel | TR@5 (n=10 single_clip dev_gold) | Counts toward vector-only min? |
|---|---:|---|
| `dense` (pgvector HNSW cosine, 1024-d) | **1.00** | ✅ |
| `sparse` (pgvector sparsevec inner product) | 0.70 | ✅ (best-of) |
| `multivec` (per-token + MaxSim) | **1.00** | ✅ |
| `rrf_4ch` (BM25 + dense + sparse + multivec via RRF) | 0.90 | ❌ (blends BM25) |

Vector-best = **1.00** ≥ 0.75 minimum. `bge-m3-all-channels` cleared on the first measurement. Voyage / Gemini fallbacks not triggered; Mission 2 (ADR 004 v3.2 amendment) did not fire.

The single `rrf_4ch` failure is `sally-staying-on-task-attention` — dense and multivec both put the gold chunk in their top-5, but BM25's contributions to the fused list pushed it past rank 5. Likely a small-corpus RRF artifact; revisit at v1 scale before tuning `RRF_K`.

## Outcome — Visual retrieval eval (separate gate, STUB)

ColQwen2.5 is the 5th retrieval channel in production but was **not** measured by bakeoff #1 — the embeddings minimum in ADR 004 v3.1 is text-channel-only and the visual candidate set has no quality minimum encoded yet. A dedicated eval gate landed this session with `code_path = 'visual_eval'`:

- **Metrics:** `VisualFrameRecall@k` (frame timestamp ± frame-sample cadence), `VisualChunkTR@k` (open-interval overlap, same convention as text), `VisualLift@k` (5-ch RRF with visual vs 4-ch RRF without visual). Definitions and rationale in `eval/reports/2026-05-27_visual_eval/methodology.mdx`.
- **Status today:** STUB. The runner skips with a clear message because (a) `visual_gold.jsonl` is a committed empty scaffold (no verified visual examples yet — dev_gold is transcript-only) and (b) `frames` / `frame_patches` are empty (no MP4 + ColQwen patch ingest run yet). Both must be populated before the gate scores anything; the test asserts the skip text so the gate cannot silently rot.
- **Atomicity:** Same eval_runs + eval_results transactional contract as bakeoff #1 (review-fix #2 patch). No winner is locked — ColQwen2.5 is the only candidate.
- **Unblock path:** verify slide-likely candidate examples from existing dev_gold (the methodology MDX names three), run MP4 ingest + ColQwen patch worker, then `uv run python -m eval.runners.run_visual_eval`. (A back-compat shim under `scripts/run_visual_eval.py` still works; the canonical path lives under `eval/runners/` because the runner defines an eval run and writes `eval_runs` / `eval_results`.)
- **Substrate readiness is per-video, not global.** The runner checks `frames.pooled_embedding > 0 AND frame_patches > 0` for each `video_id` referenced by `visual_gold.jsonl`; frames for unrelated talks do NOT count. Skip text lists the missing video_ids.
- **visual_gold modality enforced.** Both the loader (`measure_visual.load_visual_gold`) and the validator (`eval/validate.py`) reject `visual_gold.jsonl` entries whose `modality` does not intersect `{slide, screen_code, diagram, whiteboard}` — transcript-only rows belong in `dev_gold.jsonl`, not here.
- **Candidate id resolved from yaml.** `eval/runners/run_visual_eval.py::resolve_visual_candidate_id` reads `candidates.visual_retrieval.options`, filters to `provider == 'local'`, and requires exactly one match — no hard-coded `colqwen2.5` string.

---

## Where we are at HEAD

```
d28307d eval(visual): scaffold visual retrieval eval gate (separate from bakeoff #1)
751c14f eval(bakeoff): bakeoff #1 review fixes #2 — close json loophole, atomic writes, handoff hash
21d2b93 eval(bakeoff): bakeoff #1 review fixes — eval_results writes + handoff closeout
1ec8bf1 eval(bakeoff): bakeoff #1 — text embeddings (winner: bge-m3-all-channels, TR@5 best-of-channel 1.00)
03365fd docs(handoff): open phase 2 week 6 — bakeoff #1 (text embeddings)
8d6694a docs(handoff): drop stale "code_path filter deferred" claim + correct text_embeddings read shape
a1fa1dd fix(generate,db): cite bounds-check answer_claim_index + load_bakeoff_winner code_path filter
a37f9b6 docs(handoff): close phase-1 week-5 — slice 3 shipped, EXIT GATE passed, step 35 unlocks phase 2
```

Further review-fix commits may sit on top of `d28307d` (per-video substrate +
visual-modality enforcement + candidate-resolver + runner-relocation under
`eval/runners/`). Check `git log --oneline` if the handoff seems out of date.

### Gates green at close

```
make validate-evals-strict   OK (0 warnings, 11 verified gold)
make phase0-gate             OK (11/3 across 3 talks)
make test                    OK (107 fast tests — adds 16 visual + 7 review-fix tests)
make lint                    OK
uv run pytest -m slow eval/tests/test_run_visual_eval.py \
                       eval/tests/test_run_embeddings_bakeoff.py
                             OK (6 targeted slow tests, ~2:50 warm)
```

### Locked-winner DB contract

```sql
-- Aggregate (eval_runs)
SELECT summary->>'winner_candidate_id',
       summary->'per_channel_tr_at_5',
       summary->>'vector_best_score'
  FROM eval_runs
 WHERE summary->>'winner_locked' = 'true'
   AND summary->>'component'      = 'text_embeddings'
 ORDER BY created_at DESC LIMIT 1;
-- → ("bge-m3-all-channels", {"dense":1,"sparse":0.7,"multivec":1,"rrf_4ch":0.9}, "1.0")

-- Per-example detail (eval_results, one row per dev_gold example, CASCADE on run_id)
SELECT example_id, metrics->'pass_at_5_per_channel', metrics->>'tr_at_5_best_vector'
  FROM eval_results
 WHERE eval_run_id = '<run_id from above>'
 ORDER BY example_id;
```

Production callers (e.g. ingest gating on the locked embedder, once added) read the lock via:

```python
db.repos.eval_runs.load_bakeoff_winner(
    conn,
    component='text_embeddings',
    code_path='embeddings_bakeoff',     # canonical per design §5 + a1fa1dd
)
# → 'bge-m3-all-channels'
```

### Resume artifacts

```
eval/reports/2026-05-27_embeddings_bakeoff/
├── methodology.mdx     # selection-rule application, per-channel table, cost-differential, runner-up framing
├── summary.json        # aggregate dump of eval_runs.summary
└── per_example.jsonl   # one row per dev_gold example with per-channel pass@5

eval/reports/2026-05-27_visual_eval/
└── methodology.mdx     # STUB: visual eval gate spec, metrics, skip semantics, unblock path
```

The bakeoff #1 `methodology.mdx` is the resume-grade writeup for text embeddings. The visual eval MDX is a stub — it will earn its row in the resume once the substrate + gold are populated and the runner produces real numbers.

---

## What changed this session

### Bakeoff #1 — text embeddings (initial → review fix #1 → review fix #2)

| File | Change |
|---|---|
| `eval/runners/measure_embeddings.py` *(new)* | Per-channel TR@5 sweep: `tr_at_k`, `_passes_at_k`, `CHANNEL_FNS`, `measure_all_channels` (returns aggregate summary + per-example detail tuple). |
| `scripts/run_embeddings_bakeoff.py` | Replaced synthetic-stub default with real sweep against `dev_gold.jsonl`. Failing-minimum path writes the row honestly + exits non-zero. Writes one `eval_results` row per example (review-fix #1). Transactional writes + from-file shape now requires `{measurements, per_example}` object + rejects bare-list with exit 2 (review-fix #2). |
| `eval/tests/test_run_embeddings_bakeoff.py` | Slow tests: real-sweep contract, lock contract, legacy-shape rejection, mid-loop failure rollback. Self-cleaning via CASCADE on `eval_runs` delete. |
| `eval/reports/2026-05-27_embeddings_bakeoff/` *(new)* | `methodology.mdx`, `summary.json`, `per_example.jsonl`. |

### Visual retrieval eval (scaffold + review fix #3)

| File | Change |
|---|---|
| `eval/corpora/ai_engineering_v0/visual_gold.jsonl` *(new)* | Committed empty scaffold. Validator now lists it in `GOLD_FILES` and enforces `verified: true` + visual modality intersection. |
| `eval/runners/measure_visual.py` *(new)* | `VisualFrameRecall@k`, `VisualChunkTR@k`, `VisualLift@k` primitives + `measure_visual()` aggregate. Loader rejects entries without a visual modality tag. |
| `eval/runners/run_visual_eval.py` *(new, canonical)* | Runner: load → per-video substrate check → measure → eval_runs + eval_results (transactional). Resolves candidate id from `candidates.visual_retrieval.options` (`provider == 'local'`, exactly one). Invoke via `python -m eval.runners.run_visual_eval`. |
| `scripts/run_visual_eval.py` | Reduced to a back-compat shim that delegates to `eval.runners.run_visual_eval.main`. No duplicated logic. |
| `eval/validate.py` | Adds `visual_gold.jsonl` to `GOLD_FILES` and enforces visual modality on every committed entry. |
| `eval/tests/test_measure_visual.py` *(new)* | Fast unit tests: metric primitives, lift math, gold loader (including modality rejection), candidate resolver, shim-delegation guard, validator visual-modality rejection. |
| `eval/tests/test_run_visual_eval.py` *(new)* | Slow tests: live-DB skip-or-run + per-video substrate regression on a test DB (frames staged for unrelated video, visual_gold points elsewhere → must skip). |
| `eval/reports/2026-05-27_visual_eval/methodology.mdx` *(new)* | STUB report — what bakeoff #1 did NOT cover, metric definitions, skip semantics, unblock path. |

DB state at close: `eval_runs` has 1 row with `code_path='embeddings_bakeoff'` (the locked winner) plus 15 `code_path='minimal_generation'` rows from phase 1. `eval_results` has 10 rows under the bakeoff run_id. No `code_path='visual_eval'` rows yet — the gate skips until `visual_gold.jsonl` is populated AND `frames`/`frame_patches` exist for the referenced videos.

---

## Hard rules that still apply (carry forward to bakeoff #2)

1. **Eval before code.** Bakeoff measurement code lands ONLY in `scripts/` and `eval/runners/`.
2. **`dev_gold` only for selection.** `test_gold.jsonl` reserved for the phase-4 writeup. `synthesis.jsonl` reported separately, never blended.
3. **All measurements land in `eval_runs` + `eval_results`.** Aggregate in `eval_runs.summary`; per-example in `eval_results.{system_output, metrics}`. Bakeoff #2 must follow this contract.
4. **No silent model defaults.** Production readers MUST pass `code_path` to `load_bakeoff_winner` (e.g. `code_path='minimal_generation'` for generator/judge, `code_path='embeddings_bakeoff'` for text embeddings).
5. **Cross-family judge discipline (relevant for bakeoff #2).** Generator and judge MUST come from different model families. ADR 004 v3.1 §"Per-component minimums" and `docs/eval-methodology.md` §"Judge discipline".
6. **All model + provider IDs from yaml.** Pre-commit grep guard from phase 1 still applies.
7. **No ADR violations without an amendment in the same commit.**
8. **Conventional commits, no AI attribution.**

---

## Open items deferred from phase 1 (still not scope for bakeoff #2 either)

- `embed/colqwen.py::encode_image_pooled` zero-patch defensive comment is wrong (says "NULL-pooled filter downstream" — a zero vector is not NULL). Fix only when a real reason brings you into the file.
- `make answer` first-call cold-start (~30s ColQwen + ~10s BGE-M3 load). Phase-4 perf work.
- No production `load_bakeoff_winner` reader for `text_embeddings` exists today. Add one when ingest tuning needs to gate on the locked embedder (post-week 7).
- `eval/runners/judges/` versioned prompts — week 7 (bakeoff #2).
- Async LangGraph (`ainvoke`) — phase 4.
- OpenRouter failover wiring — phase 4.

### Visual eval — gated on substrate + gold (not a bakeoff #2 dependency)

- Stage verified visual gold examples in `eval/corpora/ai_engineering_v0/visual_gold.jsonl` (committed; validator enforces `verified: true`). See `eval/reports/2026-05-27_visual_eval/methodology.mdx` for slide-likely candidate examples worth verifying first.
- Run MP4 + ColQwen patch ingest to populate `frames.pooled_embedding` and `frame_patches`.
- Then `uv run python -m scripts.run_visual_eval` produces real numbers; the slow test flips from skip-path to real-assertions path automatically.
- No ADR amendment needed — ColQwen2.5 is the only visual candidate; the gate is quality, not selection.

---

## Next session (week 7)

**Bakeoff #2: generator + judge.**

Per ADR 004 v3.1: pick judge first (cheapest hitting Cohen's kappa ≥ 0.60 against the boundary audit), then pick generator with the locked judge using the selection rule (cheapest within 3pp of leader, meeting all generator minimums). Cross-family rule: judge family must differ from generator family. Requires the thin generation harness in `eval/runners/minimal_generation.py` (already shipped phase 1) — do NOT reach into the LangGraph loop.

Expected outputs:
- `eval/reports/<date>_generator_bakeoff/methodology.mdx`
- One `eval_runs` row per (generator × judge) configuration with `code_path='minimal_generation'`
- One `eval_results` row per dev_gold example per run, carrying answer + cited spans + judge verdict
- Locked rows: one for `component='judge'`, one for `component='generator'`

Estimated cost: ~$2 (per ADR 004 v3.1 §"The bakeoffs").

A separate kickoff prompt will be authored when the week-7 session starts. The stale `prompts/phase-2-week-6-kickoff.md` was deleted this session.
