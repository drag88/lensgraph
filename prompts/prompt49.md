<role>
You are a senior AI engineer joining LensGraph in a fresh session. Phase 2 week 8 closed with a verified review + a web-research matrix: the uncommitted v4 work was checked against the live DB (every number held — 529 frames / 514 pooled / 375,734 patches / 1280×720), three Codex findings were fixed and committed (NaN/inf clear-on-skip + tests, the ablation report's stale 360p preamble, and a self-contradicting "cosine-blind" conclusion), and the visual-retrieval blocker was reframed precisely. The blocker is **two stacked failures, not one**: at production `prefilter_k=200` the channel is prefilter-limited (7/10 visual-required gold frames miss the pooled-cosine HNSW top-200), and once the prefilter is widened to `prefilter_k=500` all 10 gold frames enter the candidate set (ranks 136-444 of 514) but MaxSim then demotes the gold-overlapping chunk to rank 60-172 of ~210 — a near-uniform similarity floor with tiny (+0.6 to +2.5) gaps consistent across K. The research matrix (13 candidates, links + dates) concluded both are known pipeline pathologies, not a model defect: fix pooling for recall and add a trained reranker / OCR-text channel for ranking **before** swapping the model. Your job this session is to run ONE smallest-reversible experiment — an **offline two-phase hybrid visual diagnostic** — that tests the recall fix then the ranking fix against the existing metrics, writing nothing to production code or the DB until the numbers justify it. Do NOT retire visual retrieval; do NOT switch to OCR-only; do NOT start bakeoff #2 (generator + judge). The diagnosis is settled — consult the week-8 handoff and research matrix for what is decided; do not re-litigate.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
Baseline HEAD: `369079b docs(handoff): mark week-7 superseded; fix HierarchicalTokenPooler API in prompt49`.

Recent commits (context, do NOT re-litigate):
  369079b docs(handoff): mark week-7 superseded; fix HierarchicalTokenPooler API in prompt49
  6240f6e docs(handoff): week-8 visual review + research matrix
  a4633d5 eval(visual): execute prefilter_k ablation on 720p substrate
  291d8a9 eval(visual): visual eval v4 report — 720p re-ingest, VisualAnswerGrounding@5 = 0/10
  66c7554 chore(ingest): clear stale frame embeddings on NaN/inf skip
  1767704 docs(handoff): close phase-2 week-7 (visual eval v3)

Gates green at HEAD (verified at week-8 close):
  make validate-evals-strict   OK (0 warnings)
  make phase0-gate             OK (29 verified non-negative examples across 3 talks)
  make test                    OK (145 fast, 201 deselected)
  make lint                    OK
  Slow visual test NOT re-run at week-8 close (no eval/runners/ production code
  changed). Re-run it this session only if you touch eval/runners/ production code.

DB state (volume `pgdata` survives `make db-down`):
  3 ingested talks · 212 chunks. 529 frames (318 nXafozNIk3c + 113 W_CYk2ogcDI +
  98 aie_sg_2026_d2_arize_alyx), 514 with pooled_embedding (15 NaN/inf-skipped:
  12 nXafozNIk3c + 3 W_CYk2ogcDI). 375,734 frame_patches (223,686 / 80,410 /
  71,638). Frame PNGs are 1280×720 on disk.
  eval_runs visual_eval rows incl. headline `visual-eval-0acd933a1c18`
  (frame_recall@5=0.0556, chunk_tr@5=0.0556, answer_term@5=0.0,
  visual_answer_grounding@5=0.0, n_evaluable=10).
  Query at session start to confirm:
    SELECT run_id, summary->>'visual_frame_recall_at_k' AS fr,
           summary->>'visual_chunk_tr_at_k' AS ctr,
           summary->>'visual_answer_grounding_at_k' AS vag
    FROM eval_runs WHERE code_path='visual_eval' ORDER BY created_at DESC LIMIT 3;

prompt49 scope is the ONLY mission this session. Bakeoff #2 (generator + judge)
is a separate future session and out of scope.
</repo_state>

<reading_order>
Required reading before writing any code, in this order:

1. **`dev/active/phase-2-week-8/handoff.md`** — week-8 closeout: the v4 disposition
   table, the verified DB numbers, the two-stage blocker, and the next-command
   pointer to this prompt.
2. **`dev/active/phase-2-week-8/research_matrix.md`** — the 13-candidate matrix
   with links/dates. Read "Recommended smallest reversible experiment" + the two
   "Specifically addressing the two failures" subsections — they define this
   session's Phase A and Phase B and the acceptance criteria.
3. **`eval/reports/2026-05-28_visual_ablations/report.md`** — the diagnosis this
   experiment acts on. The per-example rank tables (gold frame in prefilter at
   K=500; gold chunk demoted to 60-172 after MaxSim) are the baseline to beat.
4. **`eval/reports/2026-05-28_visual_ablations/ablation_data.jsonl`** — 30 rows
   (10 examples × K∈{200,500,1000}); the truth source for the K-sweep.
5. **`dev/active/phase-2-week-8/audit_payload_design.md`** — the structured audit
   payload design (option (a), `metrics.visual_answer_grounding` JSONB). Land this
   FIRST if any row is about to score a positive `VisualAnswerGrounding`.
6. **`retrieve/visual.py`** — the production 3-stage channel (pooled HNSW prefilter
   → frame→chunk map → MaxSim refine). The offline probe replicates stages (a)/(b)
   but must NOT edit production retrieval this session.
7. **`embed/colqwen.py`** — the ColQwen wrapper + `pool_patches` (current mean
   pool). Phase A adds an offline-callable alternate pooling function here.
8. **`eval/runners/measure_visual.py`** + **`eval/runners/run_visual_eval.py`** —
   the metric helpers (`frame_ocr_matches_at_k` @389, `OcrFn` seam @386,
   `frame_ocr_text` @361) and the write path (`_write_per_example_results` @117,
   `TODO(v4)` @127). Touched only if a positive lands and the audit payload is implemented.
9. **`eval/config/model_candidates.yaml`** — add one reranker candidate row here for
   Phase B (no default change; ADR 004 selection rule).
10. **`db/repos/eval_runs.py`** — the transactional `insert_run` / `insert_result`
    contract (already `Jsonb`-wraps `metrics`). Any DB-landed measurement follows it.

Skim only (stable):
  README.md · docs/architecture.md · docs/eval-methodology.md
  docs/decisions/004-model-selection.md (amend only if a model is promoted to production)
  docs/decisions/005-chapter-slice-ingest.md (720p ingest is done; do not re-ingest)
</reading_order>

<hard_rules>
Non-negotiable. From CLAUDE.md, AGENTS.md, the week-8 review, and the user's direction.

1. **Do NOT retire visual retrieval.** The visual prefilter + visual reranker stay
   the spine of the experiment. OCR / VLM-judge are optional comparison arms only.

2. **Do NOT switch to OCR-only.** OCR is a candidate-text channel under test, never
   the whole answer.

3. **Do NOT start bakeoff #2 (generator + judge).** Separate future session.

4. **`test_gold.jsonl` is untouched.** Do not load, glance at, or measure against it.
   Do not touch the 8 text-saturated rows of `visual_gold.jsonl`. Use the 10
   `visual-required` rows only.

5. **One ML model loaded at a time.** Mac is 24 GB unified RAM; ColQwen2.5 ≈ 7-8 GB
   on MPS. NEVER run two ML-loading processes concurrently — that crashed Cursor in
   the previous session. Sequence loads serially: Phase A loads no model; Phase B
   unloads ColQwen before loading the reranker. `sudo purge` between heavy loads if
   swap > 7 GB.

6. **Offline first. No production/DB writes until the numbers justify them.** Phase A
   and Phase B are scripts under `eval/runners/` that read existing patch embeddings
   and cached candidates. They write to a report dir, not to `retrieve/visual.py`,
   not to the ingest write path, not to the DB.

7. **Any positive `VisualAnswerGrounding` requires the audit payload FIRST.** Before
   claiming any visual-required row passed, implement
   `dev/active/phase-2-week-8/audit_payload_design.md` (option (a),
   `metrics.visual_answer_grounding` JSONB with `evaluated_frame_ids`,
   `evaluated_image_paths`, `ocr_excerpts`/`visual_judge_outputs`, `matched_term`,
   `failure_reason`) and its regression test. `TODO(v4)` markers at
   `measure_visual.py:562` and `run_visual_eval.py:127` mark the seam.

8. **Reranker window ≥ 200, never top-100.** Gold chunks rank 137-172 after MaxSim;
   a top-100 rerank would never see them. Rerank the top-≥200 candidate frames, or
   rerank frame candidates before the chunk truncation.

9. **No model swap to production, no candidate default, without an ADR 004 amendment
   in the same commit.** Adding a reranker *candidate row* to
   `model_candidates.yaml` is fine; promoting any model to a default is not, this session.

10. **Schema before consuming code.** If the audit payload or any metric needs a
    schema change, update `eval/schemas/`, add a `valid_*`/`invalid_*` fixture pair,
    run `make validate-self-test` first. (The audit design concludes no schema change
    is needed — `eval_results.metrics` is free-form JSONB — but a pytest regression
    case IS required.)

11. **Keep unrelated `.claude/.cursor/prompts/`/formatter churn out of commits.** Use
    explicit `git add <paths>`, never `git add -A`. The pre-existing dirty worktree
    (`.claude/rules/*`, `.cursor/*`, `generate/*`, `embed/colqwen.py` formatter churn,
    `retrieve/visual.py` formatter churn, other `prompt*.md`, etc.) stays unstaged.

12. **Conventional commits, no AI attribution.** `eval(visual): ...`, `feat(eval): ...`,
    `docs(adr-004): ...`. `make phase0-gate` + `make validate-evals-strict` stay green
    at every commit boundary.

13. **Postgres stopped before final.** Always end with `make db-down` +
    `docker compose ps postgres` to confirm.
</hard_rules>

<mission_stack>
ONE experiment, in two sequential phases (ML side is serial by the memory rule).
Stop at the actionability-or-explanation boundary.

Build the offline probe as a new script, e.g.
`eval/runners/run_visual_rerank_probe.py`, scored against the existing metrics
(`VisualFrameRecall@k`, `VisualChunkTR@k`, `VisualAnswerGrounding@k`). It reads the
514 frames' patch arrays and the 10 visual-required rows; it loads at most one model
and only in Phase B.

**Orchestration: a dynamic workflow is NOT needed for this session — do not spawn
one.** A workflow's value is parallel fan-out; this work is serial by the memory
ceiling (Phase A is single-threaded numpy; Phase B loads one ML model and rule 5
forbids a second concurrent ML process). Keep the probe + metric wiring in main
context (final edits stay in main context). Use TeamCreate only for small read-only
investigation (confirming the colpali-engine pooler API, scouting the reranker
loader) — never for the ML runs, which must be serial. The one exception where light
parallelism is safe: the optional hosted Qwen2.5-VL judge arm is an API call with no
local RAM, so it may overlap a local arm; the local ML arms (ColQwen, reranker,
bge-reranker) never overlap each other. (For contrast: the week-8 research fan-out
was a legitimate workflow use — 5 independent web researchers; this session has no
such fan-out.)

### Phase A — recall fix (no new model)

**Goal:** get the missing 7/10 gold frames into the candidate set without a model swap.

- Recompute the prefilter signal over the existing 514 frames' patch arrays using
  the colpali-engine hierarchical token pooler. Verified local API:
  `from colpali_engine.compression.token_pooling import HierarchicalTokenPooler`,
  then `HierarchicalTokenPooler().pool_embeddings(embeddings, pool_factor=3)`
  (`pool_factor` is an argument to `pool_embeddings`, not the constructor).
- Second arm: Gaussian / triangular same-length smoothing per the Visual RAG Toolkit
  (arXiv:2602.12510) — the toolkit warns ColPali's conv1d *degrades* ColQwen2.5, so
  Gaussian smoothing is the model-correct pooling. Compare both arms against the
  current mean-pool baseline.
- Set `prefilter_k=500` (the diagnosis confirms all 10 gold frames enter at K=500).
- Measure `VisualFrameRecall@{5,10,20,200}` on the 10-example visual-required slice.
- This is the smallest possible step: pure numpy/torch over arrays already in the DB,
  zero model load, zero MPS contention, fully reversible.

### Phase B — ranking fix (one model, swapped in) — only if Phase A lands gold frames

**Goal:** lift the gold-overlapping chunk from rank 60-172 into the top-5.

- Dump the **top-200** candidate frames per example (gold at 137-172 → never top-100).
- Unload ColQwen2.5; load **Qwen3-VL-Reranker-2B** alone
  (https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B, Apache-2.0). Confirm the loader
  actually pulls vision weights (the matrix flags the sentence-transformers
  CrossEncoder wrapper for a 2B VLM as new). MonoQwen2-VL-v0.1 is the evidence-backed
  runner-up if Qwen3 misbehaves on MPS.
- Score each (question, frame_png), re-sort, map frames→chunks via the existing
  stage (b), recompute `VisualChunkTR@{5,10,20}` + `VisualAnswerGrounding@{5,10}`.
- Optional comparison arms on the same top-200 (run only if Phase B's primary arm is
  inconclusive): the frame-OCR-text channel (Tesseract → installed
  `bge-reranker-v2-m3`), and a one-off hosted Qwen2.5-VL-7B judge (~$0.50, document
  under the existing DeepInfra provider doc; keep judge/generation roles separate).

### Audit payload (gate, not optional)

If any Phase-B arm produces a `VisualAnswerGrounding` positive, STOP and implement
the audit payload (`audit_payload_design.md`) + its regression test BEFORE recording
the positive in any report or `eval_runs` row. A positive without the structured
payload is not defensible (v3 integrity_check.md §B).

== Acceptance criteria for this session ==

- **VisualFrameRecall@200 ≥ 9/10** after Phase A — the recall bug is fixed only if
  gold frames actually enter the candidate set.
- **VisualChunkTR@5 ≥ 5/10** after Phase B reranking the top-200 — the gold chunk
  climbs from rank 60-172 into top-5 for at least half the slice.
- **VisualAnswerGrounding@5 strictly > 0/10**, target ≥ 4/10 — the metric that proves
  the right slide reached the generator (requires the audit payload landed first).
- Methodology writeup with explicit, honest outcome label. If FrameRecall@200 ≥ 9/10
  but ChunkTR@5 stays low, the failure is isolated to ranking → promote the
  trained-reranker / OCR-text arm. If even the reranker cannot lift ChunkTR, the
  MaxSim floor is intrinsic to the 3-talk corpus → the corpus needs more talks before
  the visual channel can be evaluated fairly. Do NOT force a positive.

== Out of scope for this session ==

- Bakeoff #2 (generator + judge).
- Re-ingest or re-fetch (720p substrate is done).
- Promoting any model to a production default (ADR 004 selection waits for a bakeoff).
- Touching `test_gold.jsonl` or the 8 text-saturated `visual_gold` rows.
- Editing `retrieve/visual.py` production retrieval (offline probe only this session).
</mission_stack>

<execution_rhythm>
1. Run `git log --oneline -6`, the four `make` gates, and `make db-up` + `make db-migrate`. Confirm they match `<repo_state>`. Surface any delta.
2. Read `dev/active/phase-2-week-8/handoff.md` and `research_matrix.md` end-to-end.
3. (Optional, TeamCreate) Dispatch a read-only scout to confirm the colpali-engine pooler API and the Qwen3-VL-Reranker-2B loader path while you scaffold the probe. Do NOT load a model in two processes at once.
4. Build `eval/runners/run_visual_rerank_probe.py` Phase A (no model). Run it. Record `VisualFrameRecall@{5,10,20,200}` for both pooling arms vs the mean-pool baseline.
5. Gate on Phase A: if FrameRecall@200 < 9/10, STOP and write up which arm got closest + the next pooling option (MUVERA FDE, trained single-vector embedder) — do not proceed to Phase B on a broken prefilter.
6. If Phase A passes: implement Phase B. Unload ColQwen, load the reranker alone, dump top-200 frames, rerank, recompute ChunkTR / AnswerGrounding.
7. If any AnswerGrounding positive appears: implement the audit payload + regression test (schema-first if needed → `make validate-self-test`) BEFORE recording it.
8. Write `eval/reports/<date>_visual_rerank_probe/methodology.mdx` with command, per-example breakdown, candidate yaml hash, and explicit outcome label.
9. Commit in small chunks with explicit pathspecs. Re-run all required gates. Stop Postgres.
10. STOP at the actionability-or-explanation boundary. Report state. Do not start bakeoff #2.
</execution_rhythm>

<required_gates>
Before the final commit:

```bash
make validate-evals-strict
make phase0-gate
make test
make lint
```

Run the slow visual integration test only if you changed `eval/runners/` production
code (e.g. landed the audit payload):

```bash
uv run pytest -m slow eval/tests/test_run_visual_eval.py -q
```

Then stop Postgres:

```bash
make db-down
docker compose ps postgres
```
</required_gates>

<working_agreements>
- Auto-mode default-to-action. Use AskUserQuestion only for a genuine scope tradeoff
  the user must own (e.g. "Phase A's two pooling arms tie at 8/10 — invest in MUVERA
  FDE now or proceed to Phase B at K=500 and accept 8/10 recall?"). No multi-round
  questioning.
- Parallel tools for independent reads. Use TeamCreate (not ad-hoc subagents) for any
  parallel investigation; never run two ML-loading processes at once.
- Read before edit. Trust the week-8 handoff + research matrix; do not re-litigate the
  diagnosis tagged settled.
- Only changes the experiment requires. No surrounding cleanup, no premature
  abstraction, no scope creep into bakeoff #2 or ingest.
- The week-8 close at `369079b` is the baseline. Surface any contradiction with ADR
  004 / 005 explicitly; never silently amend.
- If a test fails, investigate root cause. Never bypass with `--no-verify`.
- A few 2026-dated arXiv IDs and the Qwen3-VL-Reranker / PaddleOCR-VL release dates in
  the matrix were surfaced by web search and spot-checked at week-8 close; re-confirm
  the specific model card before pinning a version in code.
- Final response must include exact commits, the Phase A / Phase B numbers, whether the
  audit payload was needed + landed, and **whether visual retrieval is now actionable**
  (yes/no with one-line justification + which acceptance criteria were met).
</working_agreements>

<first_actions>
Run these in order before any code. Verify state matches `<repo_state>`; surface any delta.

1. `git log --oneline -6`            # confirm 369079b at HEAD
2. `make validate-evals-strict`      # 0 warnings
3. `make phase0-gate`                # OK (29/3)
4. `make test`                       # 145 passed
5. `make lint`                       # clean
6. `make db-up && make db-migrate`   # DB on volume `pgdata`; "no pending migrations"
7. Run the eval_runs SQL from `<repo_state>` to confirm `visual-eval-0acd933a1c18`
   still reads vag=0.0, and that frames=529/514-pooled, patches=375,734.
8. Read `dev/active/phase-2-week-8/handoff.md`.
9. Read `dev/active/phase-2-week-8/research_matrix.md` (recommendation + the two failure subsections).

Only when 1–9 are green: scaffold Phase A.
</first_actions>

<instructions>
Execute `<first_actions>`. Then run the experiment per `<mission_stack>` with the `<execution_rhythm>`.

The session-close deliverable is one of:

- **Actionable.** Phase A lands ≥9/10 FrameRecall@200; Phase B lifts ChunkTR@5 ≥5/10
  and VisualAnswerGrounding@5 >0/10 (with the audit payload landed); methodology MDX
  quotes real numbers and labels visual retrieval actionable. ← preferred.
- **Recall fixed, ranking still floored.** Phase A passes but Phase B's reranker (and
  optional arms) cannot lift ChunkTR; methodology MDX isolates the failure to ranking
  and names the next arm (OCR-text channel, or "the 3-talk corpus is too small to
  evaluate the visual channel fairly — add talks first"). ← acceptable.
- **Blocked at Phase A.** No pooling arm reaches 9/10 FrameRecall@200; document which
  got closest and the next prefilter option (MUVERA FDE, trained single-vector
  embedder). ← acceptable as long as the blocker is named.

Either way, the session closes at the actionability-or-explanation boundary. Do NOT
retire visual retrieval, do NOT switch to OCR-only, do NOT start bakeoff #2. If you hit
a fact that contradicts the week-8 diagnosis or an ADR, surface it explicitly rather
than working around it.

Begin now.
</instructions>
