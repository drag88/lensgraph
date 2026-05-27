<role>
You are a senior AI engineer joining LensGraph in a fresh session. Phase 2 week 6 is closed: bakeoff #1 (text embeddings) locked `bge-m3-all-channels`; a visual retrieval eval gate landed with infrastructure, ADR 005 (chapter-slice video capture), curation tooling, 18 verified visual_gold rows (8 text-saturated + 10 visual-required tagged), and two real `visual_eval` runs against 529 ingested ColQwen-patched frames. The last visual run reported `frame_recall@5=0.056`, `chunk_tr@5=0.111`, `lift=-5.56 pp` (with: 17/18 · without: 18/18) — **lift was labelled NON-ACTIONABLE because the 4-channel text RRF passes 18/18 even on the visual-required slice**. The reason is a measurement-model mismatch: TR@5 measures span coverage, not answer-bearing-ness; the speaker is talking around the topic in the same window, so text channels surface topical-but-not-answer-bearing chunks and the gold-span overlap test passes anyway. Your job this session is to fix that — design an **answer-grounded visual metric**, diagnose where ColQwen actually loses, verify the ColQwen processor band, and tighten the visual-required gold — running as a **parallel agent-team slice** so you do NOT block the negative/boundary-audit workstream and do NOT start bakeoff #2 (generator + judge). The design is paid for in blood across phase 1 + four phase-2 weeks of review fixes — do not re-litigate; consult the handoff for what's settled.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
Baseline HEAD: `7c84cff chore(curation): merge 2026-05-27_required/ into 2026-05-27/ per-talk subdirs`. The last meaningful eval change was `5c44735` (visual-required slice + second real visual_eval run); `7c84cff` is a curation-folder consolidation only.

Recent commits (context, do NOT re-litigate):
  7c84cff chore(curation): merge 2026-05-27_required/ into 2026-05-27/ per-talk subdirs
  5c44735 eval(visual): visual-required curation slice (n=10) — lift remains NON-ACTIONABLE
  f15b847 feat(scripts)+docs(adr-005): implement fetch_video.py + Makefile targets; flip ADR 005 to Accepted
  fc56b39 docs(handoff): refresh phase-2 week-6 handoff to current HEAD ec7b5b7
  ec7b5b7 eval(visual): first real visual_eval run — frame_recall@5=0.125, chunk_tr@5=0.125, lift=0pp (text-saturated)
  d5b4bb6 docs(adr-005)+scripts: chapter-slice video capture pattern + ingest queue drain
  0d646da eval(visual): curate 8 verified visual_gold rows + frame evidence (review-approved)
  d20c024 eval(visual): review fixes #3 — per-video substrate, modality, candidate id from yaml, runner relocation

Gates green at HEAD (verified at week-6 close):
  make validate-evals-strict   OK (0 warnings)
  make phase0-gate             OK (29 verified non-negative examples across 3 talks)
  make test                    OK (116 fast)
  make lint                    OK
  uv run pytest -m slow eval/tests/test_run_visual_eval.py eval/tests/test_run_embeddings_bakeoff.py -q
                               OK (6 tests, ~17–18 min cold — visual eval runs 18-example happy path)

DB state (volume `pgdata` survives `make db-down`):
  3 ingested talks · 212 chunks · 212 dense_embeds · 212 sparse_embeds · 73,621 chunk_token_embeds.
  529 frames (318 nXafozNIk3c + 113 W_CYk2ogcDI + 98 aie_sg_2026_d2_arize_alyx) all with pooled_embedding.
  163,990 frame_patches across 529 distinct frame_ids.
  eval_runs: 1 row `embeddings_bakeoff` (bakeoff #1 locked winner) + 2 rows `visual_eval`
    (`visual-eval-3d5384cd6446` first run n=8, `visual-eval-8b9fecfbf6b4` second run n=18)
    + 15 rows `minimal_generation` from phase 1. eval_results: 10 (bakeoff) + 18 (visual second run) + per-example minimal-gen rows.
  Query at session start: `SELECT code_path, count(*) FROM eval_runs GROUP BY code_path` — exact totals drift.

prompt47 scope is the ONLY mission for this session. Bakeoff #2 (generator + judge) is a separate future session and out of scope.
</repo_state>

<reading_order>
Required reading before writing any code, in this order:

1. **`dev/active/phase-2-week-6/handoff.md`** — week-6 closeout including the "Next session (parallel agent-team slice — `prompts/prompt47.md`)" pointer. Reads in ~8 minutes.
2. **`eval/reports/2026-05-27_visual_eval/methodology.mdx`** — the **NON-ACTIONABLE lift** writeup. The "Why the acceptance gate failed" + "What would make the next visual eval more informative" sections name the two follow-up paths this session implements: answer-grounded eval, visual-only baseline corpus. Read carefully.
3. **`eval/reports/2026-05-27_visual_eval/per_example.jsonl`** — per-example top-5 frames + chunks the runner actually retrieved. Agent B's diagnostic baseline.
4. **`eval/corpora/ai_engineering_v0/visual_gold.jsonl`** — 18 rows. The 10 with `tags=["visual-required", ...]` are the slice this session sharpens. The 8 without that tag are text-saturated and stay for regression coverage.
5. **`eval/curation/visual_gold_review/2026-05-27/<video_id>/`** — frame `.jpg` evidence + agent_a/b/c_notes.md (pass 1) + visual_required_notes.md + visual_required_proposed.jsonl (pass 2). Agent D's review surface.
6. **`docs/decisions/005-chapter-slice-ingest.md`** — ADR v2 (Accepted). §"Frame-sample resolution" + the "Still owed" implementation-slice note flag the ColQwen processor `min_pixels`/`max_pixels` verification as Agent C's mission.
7. **`eval/runners/measure_visual.py`** + **`eval/runners/run_visual_eval.py`** — the eval entry points + atomicity contract; Agent A extends one or both.
8. **`retrieve/visual.py`** — the production 3-stage visual channel (pooled prefilter → frame→chunk map → MaxSim refine). Agent B's diagnostic target.
9. **`embed/colqwen.py`** — the ColQwen wrapper. Agent C's investigation surface.
10. **`db/repos/eval_runs.py`** — the transactional `insert_run` / `insert_result` / `lock_bakeoff_winner` contract. Agent A's writes must follow it.

Skim only (stable):
  README.md · docs/PRD.md · docs/architecture.md
  docs/decisions/001…004 (model selection contracts; do NOT amend without explicit cause)
  db/migrations/{0001..0005}*.sql (schema frozen through week 5)
</reading_order>

<hard_rules>
Non-negotiable. From CLAUDE.md, ADRs, the design, and the week-6 handoff.

1. **`dev_gold` only.** `visual_gold.jsonl` is dev-only (validator already enforces). `test_gold.jsonl` is RESERVED — do not load, glance at, or measure against it under any flag. The agent-team slice does NOT amend the lock on text embeddings.

2. **No model swap, no candidate change, without an ADR amendment in the same commit.** ColQwen2.5 is the only visual_retrieval candidate today. This session is **diagnosis + metric design**, not selection.

3. **All measurements land in `eval_runs` + `eval_results`.** Aggregate in `eval_runs.summary`; per-example in `eval_results.{system_output, metrics}`. Same atomicity contract as bakeoff #1 review-fix #2 (single transaction; failure rolls back). Any new metric the team adds must follow it.

4. **Postgres only for measurements.** No external observability sink.

5. **All model + provider IDs from yaml.** The pre-commit grep guard from phase 1 still applies (no wire IDs in `generate/`, `eval/runners/minimal_generation.py`, `scripts/`, `db/repos/`).

6. **Production `load_bakeoff_winner` calls pass `code_path` explicitly.** Per design §5 SQL + a1fa1dd.

7. **Visual modality enforcement stays on.** Both the loader (`measure_visual.load_visual_gold`) and the validator reject `visual_gold.jsonl` entries lacking `{slide, screen_code, diagram, whiteboard}` intersection. If Agent A adds a new file (`visual_answer_gold.jsonl` or similar), it MUST land in `eval/validate.py::GOLD_FILES` with the same `verified: true` + visual-modality rules.

8. **Methodology writeup required.** Any new measurement → `eval/reports/<date>_visual_eval_v2/methodology.mdx` (or appended to the existing one with a v2 section). Per CLAUDE.md experiment-tracking: command run, fixture diff, per-example breakdown, candidate yaml hash, **explicit actionability label** ("lift actionable" iff text-only baseline misses ≥1 example under the new metric; otherwise "NON-ACTIONABLE").

9. **`make phase0-gate` + `make validate-evals-strict` stay green at every commit boundary.**

10. **Conventional commits, no AI attribution.** `eval(visual): ...`, `feat(eval): ...`, `docs(adr-NNN): ...`.

11. **Do NOT block the parallel workstreams.** The negative/boundary-audit workstream and any other in-flight work continues independently. This session's commits should not touch their files.

12. **Do NOT start bakeoff #2 (generator + judge).** That is a separate future session.
</hard_rules>

<mission_stack>
ONE parallel slice this session. Stop at the actionability-or-explanation boundary.

== Run as a parallel agent-team slice. Use subagents with separate ownership: ==

### Agent A — Metric / Data Contract

**Goal:** design an answer-grounded visual metric — did the retrieved frame/chunk actually contain the **expected visual claim**, not just overlap the gold timestamp?

- Propose: schema extension on the existing `gold_example.schema.json` (e.g. a new optional `visual_evidence` array of structured claims per row) OR a separate file (`visual_answer_gold.jsonl`) with a slimmer schema dedicated to answer-grounded checks.
- Decide between the two; pick one and justify in the design note.
- Preserve dev/test separation. Do NOT touch `test_gold.jsonl`. If new file, add to `GOLD_FILES` in `eval/validate.py` with `verified: true` + visual-modality rules.
- Define the metric: how is "answer-bearing" checked? Options:
  - Exact-substring match against frame OCR text (would need OCR)
  - LLM judge call per (retrieved chunk text, expected claim) tuple — note that this means a judge model is needed even though bakeoff #2 hasn't selected one; you can use a pinned cross-family small judge for now (e.g. one of the cheap_extraction candidates from ADR 004) and amend later
  - Structured slide-content fields curated per row (Agent D's responsibility) that the eval string-compares
- Output: concrete schema changes (with fixture pairs in `eval/tests/fixtures/`), `eval/runners/measure_visual.py` extension, `eval/runners/run_visual_eval.py` write path, fast tests covering the new metric primitives.

### Agent B — Retrieval Diagnostics

**Goal:** inspect current visual-eval failures and produce a per-stage diagnostic.

- For each visual-required example (10 rows tagged `visual-required` in `visual_gold.jsonl`), compute gold-frame/chunk rank at @5/@10/@20/@50 under `retrieve.visual.retrieve_frames` and `retrieve.visual.retrieve`.
- Break down each failure by pipeline stage:
  1. pooled-vector HNSW frame prefilter
  2. frame→containing-chunk SQL join
  3. MaxSim refine over the candidate chunks
  4. RRF fusion (when the visual channel is part of the 5-channel mix)
- Distinguish **near misses** (gold at rank 6–20) from **total misses** (rank > 50 or absent).
- Output: per-example diagnostic JSONL/report under `eval/reports/<date>_visual_diagnostics/`. Do NOT change `retrieve/visual.py` — diagnosis only. Stage 4 (RRF fusion) is informational because it depends on the text channels too.

### Agent C — ColQwen / Substrate Quality

**Goal:** verify the ColQwen processor band flagged in ADR 005, then ablate obvious quality levers.

- Print `ColQwen2_5_Processor.from_pretrained(...).image_processor.{min_pixels, max_pixels}` and compare against the actual PNG dimensions written by `ingest/frames.py` at 720p. Decision: is 720p aggressively downsampled, or roughly inside the band?
- On a small subset (e.g. 20 frames from W_CYk2ogcDI), ablate these levers and report standalone visual recall @5/@10:
  - Frame cadence: every_sec ∈ {5, 10, 15}
  - Resolution: 480p, 720p (current), 1080p — re-encode existing 720p down/up via ffmpeg OR re-fetch at the alt resolution via `scripts/fetch_video.py` (the existing `YT_DLP_FORMAT` would need a temporary override; keep it ablation-only)
  - Slide crop/letterbox handling — what % of the frame area is actually the slide vs background chrome?
  - Top-k candidate pool size (`prefilter_k` in `retrieve.visual.retrieve`, default 200)
- **No model swap.** This is diagnosis + ablation only. Output a small report under `eval/reports/<date>_visual_diagnostics/colqwen_ablation.md` with the recommended setting band — but DO NOT change defaults in code without an ADR amendment.

### Agent D — Visual-Required Gold Quality

**Goal:** review the 10 visual-required rows for whether they are genuinely answer-bearing visually and non-verbatim in transcript.

- Re-inspect each row's frame evidence at `eval/curation/visual_gold_review/2026-05-27/<video_id>/`. Compare against the row's `question` + `expected_claims` + `expected_answer`.
- For each: verdict = ACCEPT-AS-IS / REVISE (with proposed change) / REJECT (with reason).
- Add NEW rows only if the existing 10 leave gaps in coverage AND you have backing frame evidence + a transcript-grep verifying the answer is not verbatim. Save any new frame grabs as `.jpg` in the existing per-talk evidence subdir. Raw MP4s stay gitignored.
- Output: a `visual_required_review_2026-MM-DD.md` per talk under `eval/curation/visual_gold_review/2026-05-27/<video_id>/` (alongside the existing `visual_required_notes.md`); any actual edits to `visual_gold.jsonl` rows go through `verified: true` only after frame inspection.

== Acceptance criteria for this session ==

- New metric **distinguishes "right timestamp" from "answer-bearing visual support."**
- Text-only baseline must **fail at least one visual-required example under the new answer-grounded metric**, OR the report must explain why the corpus is still invalid for lift (and label lift NON-ACTIONABLE).
- Produce `eval_runs` + `eval_results` rows for any new measurement (transactional contract).
- Report rank curves @5/@10/@20/@50 for the visual-required slice.
- Keep current model candidates unchanged unless the metric proves a model-level issue.
- **No production decision from small-n data without a methodology writeup.**

== Out of scope for this session ==

- Bakeoff #2 (generator + judge). That is a separate future session.
- Schema changes to the existing 4 text-eval files in `GOLD_FILES`. Agent A's schema extensions are additive only.
- Anything in `ingest/`, `chunking/`, `generate/`, `api/`, `web/` — diagnosis-only this session.
- The negative/boundary-audit workstream — keep that parallel.
</mission_stack>

<execution_rhythm>
1. Run `git log --oneline -10`, the four `make` gates, and `make db-up` + `make db-migrate`. Confirm they match `<repo_state>`. Surface any delta.
2. Read `dev/active/phase-2-week-6/handoff.md` and `eval/reports/2026-05-27_visual_eval/methodology.mdx` end-to-end.
3. Dispatch Agents A / B / C / D in parallel (single message with four Agent calls) — each gets the relevant `<mission_stack>` section verbatim plus pointers to the files in `<reading_order>` they own.
4. Aggregate findings as they return. Don't start writing code until all four are back AND you've decided which of Agent A's two schema options to land.
5. Implement the chosen design in small commits: schema/fixture pair first (Agent A's output), then runner extension + tests, then methodology writeup. Keep `make validate-evals-strict` + `make test` green at every commit boundary.
6. If Agent D's review produces any new/revised visual_gold rows, commit those as a separate `eval(visual): ...` commit with frame evidence in the same commit.
7. Run the new answer-grounded measurement once warm. Write the methodology MDX with explicit actionability label.
8. Re-run all required gates (see <required_gates>). Stop Postgres.
9. STOP at the actionability boundary. Report state. Do not start bakeoff #2.
</execution_rhythm>

<required_gates>
Before the final commit:

```bash
make validate-evals-strict
make phase0-gate
make test
make lint
uv run pytest -m slow eval/tests/test_run_visual_eval.py eval/tests/test_run_embeddings_bakeoff.py -q
```

Plus any new slow tests Agent A introduces for the answer-grounded metric.
</required_gates>

<working_agreements>
- Auto-mode default-to-action. Use AskUserQuestion only when the user must own a scope tradeoff that no agent can resolve (e.g. "Agent A wants to add a structured `visual_evidence` schema field; Agent D wants to keep visual_gold lean and add a parallel `visual_answer_gold.jsonl` — pick one"). No multi-round questioning.
- Parallel tools for independent operations. Dispatch the four agents in ONE message with four Agent tool calls — they are independent.
- Read before edit. Trust the handoff + methodology MDX; do not re-litigate decisions tagged "settled" in either.
- Only changes the mission requires. No surrounding cleanup, no premature abstractions, no scope creep into bakeoff #2 or ingest.
- The week-6 handoff at `7c84cff` is the baseline. ADR 004 v3.1 + ADR 005 v2 are the contracts. Surface gaps explicitly; never silently amend.
- Before any commit touching `db/`, `queues/`, `ingest/`, `chunking/`, `retrieve/`, `generate/`, `embed/`, `api/`, `web/`: run `make phase0-gate` locally. Diagnostic-only changes under `eval/runners/`, `scripts/`, `eval/tests/`, `eval/reports/` are exempt by directory scope.
- If a test fails, investigate root cause. Never bypass with `--no-verify`.
- Untracked files that already exist at session start (`.claude/rules/*`, `AGENTS.generated.md`, `CLAUDE.generated.md`, etc.) are out of scope.
- Final response must include exact commits, diagnostics, the new metric definition, and **whether visual retrieval is now actionable** (yes/no with one-line justification).
</working_agreements>

<first_actions>
Run these in order before any code. Verify state matches `<repo_state>`; if it does not, surface the delta before proceeding.

1. `git log --oneline -10`
   Confirm `7c84cff` (chore(curation): merge ...) at HEAD.

2. `make validate-evals-strict`     # 0 warnings
3. `make phase0-gate`                # OK (29/3)
4. `make test`                       # 116 passed
5. `make lint`                       # clean
6. `make db-up && make db-migrate`   # DB on volume `pgdata`; "no pending"

7. `uv run python -c "import psycopg; from db.conn import dsn;
   c = psycopg.connect(dsn(), autocommit=True);
   print(c.execute(\"SELECT code_path, count(*) FROM eval_runs GROUP BY code_path\").fetchall());
   print('visual gold visual-required:', c.execute(\"SELECT count(*) FROM eval_results WHERE eval_run_id IN (SELECT run_id FROM eval_runs WHERE code_path='visual_eval')\").fetchone())"`
   Surface DB state vs `<repo_state>`.

8. Read `dev/active/phase-2-week-6/handoff.md`.

9. Read `eval/reports/2026-05-27_visual_eval/methodology.mdx`.

10. Dispatch Agents A / B / C / D in parallel per `<mission_stack>` (single message, four Agent tool calls).

Only when 1–9 are green AND the four agent calls are dispatched: begin aggregating findings.
</first_actions>

<instructions>
Execute `<first_actions>`. Then ship the parallel slice per `<mission_stack>` with the `<execution_rhythm>`.

The session-close deliverable is one of:

- **Actionable lift.** An answer-grounded metric is defined + landed; text-only baseline misses ≥1 visual-required example under that metric; new `eval_runs` + `eval_results` rows quote real numbers; methodology MDX labels lift actionable with the supporting evidence. ← preferred outcome.
- **Still non-actionable, with explanation.** The new metric ran but text-only baseline still didn't miss; methodology MDX explains WHY (corpus structure, transcript-topical-overlap, etc.) and proposes the next concrete corpus or metric change needed. ← acceptable outcome.

Either way, the session closes at the actionability-or-explanation boundary. Bakeoff #2 (generator + judge) and the negative/boundary-audit workstream are out of scope and untouched.

If you discover a fact that contradicts ADR 004 v3.1 or ADR 005 v2, surface it explicitly. Do not silently work around it. Amendments go through the same process the existing ADRs did.

Begin now.
</instructions>
