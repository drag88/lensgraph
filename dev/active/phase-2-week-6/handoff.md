# Handoff — Phase 2 Week 6 (Bakeoff #1: text embeddings) — CLOSED

**Opened:** 2026-05-27 (phase-1 week-5 closed at `8d6694a`)
**Closed:** 2026-05-27 — bakeoff #1 locked, visual eval scaffold + first real run shipped.
**Status:** ✅ Bakeoff #1 (text embeddings) shipped. `bge-m3-all-channels` locked as the text-embeddings winner. Best-of vector channel TR@5 = 1.00 against the ADR 004 v3.1 minimum (≥ 0.75). No ADR amendment required. ✅ Visual retrieval (ColQwen2.5 — the 5th channel) has its own eval gate (`eval_runs.code_path = 'visual_eval'`) — infrastructure landed, 8 verified visual gold rows curated, and a FIRST REAL RUN landed at `visual-eval-3d5384cd6446`: VisualFrameRecall@5=0.125, VisualChunkTR@5=0.125, VisualLift@5=0.0 pp (lift is pinned-by-construction because text RRF saturates at 8/8 on this corpus). Bakeoff #1 did NOT measure or lock visual retrieval.
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

## Outcome — Visual retrieval eval (separate gate, FIRST REAL RUN)

ColQwen2.5 visual retrieval ran end-to-end against 8 verified visual_gold examples on 529 ingested frames (~310 patches each) across the 3 talks. Run id `visual-eval-3d5384cd6446` (`code_path='visual_eval'`).

| Metric | Value | Counts |
|---|---:|---|
| `VisualFrameRecall@5` | **0.125** | 1/8 |
| `VisualChunkTR@5` | **0.125** | 1/8 |
| `VisualLift@5` (5-ch w/ visual vs 4-ch text-only) | **0.0 pp** | with: 8/8 · without: 8/8 |

Headline reading: visual lift is 0 pp **but the test is pinned by construction** — bakeoff #1's 4-channel text RRF already passes 8/8 on these examples, so this corpus cannot measure whether the visual channel rescues text-failure cases. Standalone visual recall (1/8 frame, 1/8 chunk) is below expectation; ADR 005 already flagged the ColQwen processor `min_pixels` / `max_pixels` band as unverified, which may be one cause. Full analysis + per-example breakdown + reproducibility commands in `eval/reports/2026-05-27_visual_eval/methodology.mdx`. **No model swap is justified by this run** — ADR 004 v3.1 has no visual minimum encoded, and removing ColQwen on text-saturated lift would be a methodology mistake.

Substrate that made the run possible:

- 3 source MP4s downloaded under `videos/ai_engineering_v0/` (gitignored). `m12vGjfbNlo.mp4` (parent of the Arize chapter) used the ADR 005 `--download-sections '*0-1500'` pattern → 39 MB instead of the projected 5–15 GB full-stream.
- 529 frames sampled at every_sec=10 via `frames_handler` (318 + 113 + 98 per talk).
- 163,990 frame_patches encoded by `embed_frames_handler` via ColQwen2.5 (~30 min Mac MPS).
- `scripts/drain_ingest_queue.py` ran both PGMQ queues end-to-end. Single-driver pattern documented in commit `d5b4bb6`.

ColQwen2.5 is the 5th retrieval channel in production but was **not** measured by bakeoff #1 — the embeddings minimum in ADR 004 v3.1 is text-channel-only and the visual candidate set has no quality minimum encoded yet. A dedicated eval gate landed this session with `code_path = 'visual_eval'`:

- **Metrics:** `VisualFrameRecall@k` (frame timestamp ± frame-sample cadence), `VisualChunkTR@k` (open-interval overlap, same convention as text), `VisualLift@k` (5-ch RRF with visual vs 4-ch RRF without visual). Definitions and rationale in `eval/reports/2026-05-27_visual_eval/methodology.mdx`.
- **Status today:** REAL RUN. `visual_gold.jsonl` has 8 verified examples curated from all 10 `dev_gold.jsonl` single-clip rows; 2 transcript-only rows were rejected. The first end-to-end run wrote `eval_runs` row `visual-eval-3d5384cd6446` + 8 `eval_results` rows after 529 frames + 163,990 ColQwen patches were ingested (~30 min Mac MPS via `scripts/drain_ingest_queue.py`). Numbers above; full analysis in the methodology MDX. The runner still has a working skip path for missing substrate (asserted by the slow integration test), so the gate cannot silently regress if a future video_id lands without ingest.
- **Atomicity:** Same eval_runs + eval_results transactional contract as bakeoff #1 (review-fix #2 patch). No winner is locked — ColQwen2.5 is the only candidate.
- **Unblock path completed this session.** Gold + substrate + first real run all landed. Future ColQwen runs follow the same path: drain `ingest_frames` → drain `ingest_embed_frames` → `python -m eval.runners.run_visual_eval`. A back-compat shim under `scripts/run_visual_eval.py` still works; the canonical path lives under `eval/runners/` because the runner defines an eval run and writes `eval_runs` / `eval_results`.
- **Substrate readiness is per-video, not global.** The runner checks `frames.pooled_embedding > 0 AND frame_patches > 0` for each `video_id` referenced by `visual_gold.jsonl`; frames for unrelated talks do NOT count. Skip text lists the missing video_ids.
- **visual_gold modality enforced.** Both the loader (`measure_visual.load_visual_gold`) and the validator (`eval/validate.py`) reject `visual_gold.jsonl` entries whose `modality` does not intersect `{slide, screen_code, diagram, whiteboard}` — transcript-only rows belong in `dev_gold.jsonl`, not here.
- **Candidate id resolved from yaml.** `eval/runners/run_visual_eval.py::resolve_visual_candidate_id` reads `candidates.visual_retrieval.options`, filters to `provider == 'local'`, and requires exactly one match — no hard-coded `colqwen2.5` string.

---

## Where we are at HEAD

```
ec7b5b7 eval(visual): first real visual_eval run — frame_recall@5=0.125, chunk_tr@5=0.125, lift=0pp (text-saturated)
d5b4bb6 docs(adr-005)+scripts: chapter-slice video capture pattern + ingest queue drain
0d646da eval(visual): curate 8 verified visual_gold rows + frame evidence (review-approved)
d20c024 eval(visual): review fixes #3 — per-video substrate, modality, candidate id from yaml, runner relocation
d28307d eval(visual): scaffold visual retrieval eval gate (separate from bakeoff #1)
751c14f eval(bakeoff): bakeoff #1 review fixes #2 — close json loophole, atomic writes, handoff hash
21d2b93 eval(bakeoff): bakeoff #1 review fixes — eval_results writes + handoff closeout
1ec8bf1 eval(bakeoff): bakeoff #1 — text embeddings (winner: bge-m3-all-channels, TR@5 best-of-channel 1.00)
03365fd docs(handoff): open phase 2 week 6 — bakeoff #1 (text embeddings)
8d6694a docs(handoff): drop stale "code_path filter deferred" claim + correct text_embeddings read shape
a1fa1dd fix(generate,db): cite bounds-check answer_claim_index + load_bakeoff_winner code_path filter
a37f9b6 docs(handoff): close phase-1 week-5 — slice 3 shipped, EXIT GATE passed, step 35 unlocks phase 2
```

Further review-fix commits may sit on top of `ec7b5b7` (handoff refresh,
ADR 005 consistency fixes, visual-required curation pass). Check
`git log --oneline` if the handoff seems out of date.

### Gates green after the first real visual_eval run

```
make validate-evals-strict   OK (0 warnings)
make phase0-gate             OK (19/3 across 3 talks)
make test                    OK (107 fast tests — adds 16 visual + 7 review-fix tests)
make lint                    OK
uv run python -m eval.runners.run_visual_eval
                             OK (frame_recall@5=0.125, chunk_tr@5=0.125, n=8 → run_id visual-eval-3d5384cd6446)
uv run pytest -m slow eval/tests/test_run_visual_eval.py \
                       eval/tests/test_run_embeddings_bakeoff.py
                             OK (6 targeted slow tests, ~8:41 warm; visual eval test runs the happy path
                             including a fresh ColQwen pass + self-cleans via CASCADE)
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
├── methodology.mdx     # REAL RUN: per-channel + lift numbers, per-example analysis, reproducibility
├── summary.json        # full dump of eval_runs.summary for run_id visual-eval-3d5384cd6446
└── per_example.jsonl   # one row per visual gold example with top-5 frames + chunks

eval/curation/visual_gold_review/2026-05-27/
└── README.md           # visual-gold curation notes plus contact sheets/frame evidence

docs/decisions/005-chapter-slice-ingest.md  # ADR for chapter-slice video capture
scripts/drain_ingest_queue.py               # operational driver for PGMQ workers
```

The bakeoff #1 `methodology.mdx` is the resume-grade writeup for text embeddings. The visual eval MDX is the REAL-RUN writeup as of `ec7b5b7`: standalone visual recall is weak (1/8 on both metrics), lift is 0 pp but pinned by construction (text channels saturate), and the writeup is explicit that no model swap is justified.

---

## What changed this session

### Bakeoff #1 — text embeddings (initial → review fix #1 → review fix #2)

| File | Change |
|---|---|
| `eval/runners/measure_embeddings.py` *(new)* | Per-channel TR@5 sweep: `tr_at_k`, `_passes_at_k`, `CHANNEL_FNS`, `measure_all_channels` (returns aggregate summary + per-example detail tuple). |
| `scripts/run_embeddings_bakeoff.py` | Replaced synthetic-stub default with real sweep against `dev_gold.jsonl`. Failing-minimum path writes the row honestly + exits non-zero. Writes one `eval_results` row per example (review-fix #1). Transactional writes + from-file shape now requires `{measurements, per_example}` object + rejects bare-list with exit 2 (review-fix #2). |
| `eval/tests/test_run_embeddings_bakeoff.py` | Slow tests: real-sweep contract, lock contract, legacy-shape rejection, mid-loop failure rollback. Self-cleaning via CASCADE on `eval_runs` delete. |
| `eval/reports/2026-05-27_embeddings_bakeoff/` *(new)* | `methodology.mdx`, `summary.json`, `per_example.jsonl`. |

### Visual retrieval eval (scaffold → curation → ADR 005 → first real run)

| File | Change |
|---|---|
| `eval/corpora/ai_engineering_v0/visual_gold.jsonl` *(new)* | 8 verified visual examples curated from the 10 dev single-clip examples. Validator lists it in `GOLD_FILES` and enforces `verified: true` + visual modality intersection. |
| `eval/runners/measure_visual.py` *(new)* | `VisualFrameRecall@k`, `VisualChunkTR@k`, `VisualLift@k` primitives + `measure_visual()` aggregate. Loader rejects entries without a visual modality tag. |
| `eval/runners/run_visual_eval.py` *(new, canonical)* | Runner: load → per-video substrate check → measure → eval_runs + eval_results (transactional). Resolves candidate id from `candidates.visual_retrieval.options` (`provider == 'local'`, exactly one). Invoke via `python -m eval.runners.run_visual_eval`. |
| `scripts/run_visual_eval.py` | Reduced to a back-compat shim that delegates to `eval.runners.run_visual_eval.main`. No duplicated logic. |
| `eval/validate.py` | Adds `visual_gold.jsonl` to `GOLD_FILES` and enforces visual modality on every committed entry. |
| `eval/tests/test_measure_visual.py` *(new)* | Fast unit tests: metric primitives, lift math, gold loader (including modality rejection), candidate resolver, shim-delegation guard, validator visual-modality rejection. |
| `eval/tests/test_run_visual_eval.py` *(new)* | Slow tests: live-DB skip-or-run + per-video substrate regression on a test DB. Now exercises the happy path (substrate ingested) end-to-end. |
| `eval/curation/visual_gold_review/2026-05-27/` *(new, commit `0d646da`)* | Curation evidence for all 10 dev single-clip rows: subagent notes, proposed rows, contact sheets, and representative frame grabs. Raw `.mp4` review clips are ignored. |
| `docs/decisions/005-chapter-slice-ingest.md` *(new, commit `d5b4bb6`)* | ADR for `yt-dlp --download-sections '*0-<max(end)+30>'` chapter-slice capture pattern. Preserves source t=0; no code/schema change. |
| `scripts/drain_ingest_queue.py` *(new, commit `d5b4bb6`)* | Operational driver for PGMQ workers (`ingest_frames`, `ingest_embed_frames`). |
| `eval/reports/2026-05-27_visual_eval/{methodology.mdx, summary.json, per_example.jsonl}` *(new/updated, commit `ec7b5b7`)* | REAL-RUN report with per-channel + lift numbers, per-example analysis, reproducibility commands. |

DB state at close: `eval_runs` has 1 row with `code_path='embeddings_bakeoff'` (the bakeoff #1 locked winner) + 1 row with `code_path='visual_eval'` (the first real visual run, `visual-eval-3d5384cd6446`) + 15 `code_path='minimal_generation'` rows from phase 1. `eval_results` has 10 rows under the bakeoff run_id and 8 rows under the visual run_id (one per visual_gold example).

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

### Visual eval — shipped this session; followups for v1 (not a bakeoff #2 dependency)

- First real run landed at `visual-eval-3d5384cd6446` (commit `ec7b5b7`). Visual gold, ingest pipeline, and methodology MDX all populated.
- **Visual-required gold pass is still owed.** The current 8 visual_gold rows are text-saturated (lift = 0 pp by construction). To make lift actionable, curate examples where the answer is on the slide and NOT in the transcript verbatim, so the 4-channel text RRF misses on at least one example.
- **ColQwen processor `min_pixels` / `max_pixels` verification** flagged in ADR 005 is still owed. Print the actual band against the 720p PNG dimensions before drawing strong conclusions from the 1/8 standalone numbers.
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
