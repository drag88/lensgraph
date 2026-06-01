<role>
You are a senior AI engineer joining LensGraph in a fresh session. Phase 2 week 7 closed with visual eval v3: a frame-side `VisualAnswerGrounding@k` metric (OCR via pytesseract) shipped, reported `0/10` on the visual-required slice, and identified the window gate as the upstream failure. Between then and now, an uncommitted v4 follow-up was attempted in the previous session — 720p re-ingest, an `ingest/handlers.py` NaN/inf skip patch, a re-run visual_eval, and a `prefilter_k ∈ {200, 500, 1000}` ablation. The v4 work is **uncommitted and unverified**; the previous session also got stuck before doing the research step the user explicitly asked for. Your job this session is to (1) review the v4 working tree against the actual repo + DB state, separate verified facts from claims, and (2) do a thorough web-research-backed visual retrieval redesign matrix before recommending the next eval experiment. The user does NOT want visual retrieval retired and does NOT want to blindly choose OCR. Research first. Bakeoff #2 (generator + judge) is out of scope.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
Baseline HEAD: `1767704 docs(handoff): close phase-2 week-7 (visual eval v3)`. Codex reviewed this stack and accepted it; treat HEAD as the verified baseline.

Recent commits at HEAD (verified by Codex):
  1767704 docs(handoff): close phase-2 week-7 (visual eval v3)
  c850338 eval(visual): stage prefilter_k ablation script (deferred)
  b79f95d docs(eval): visual eval v3 report — VisualAnswerGrounding@5 = 0/10
  f5d59e2 eval(visual): add VisualAnswerGrounding@k frame-side metric (OCR)
  e7a0871 feat(scripts)+docs(adr-005): yt-dlp 720p height guard
  2754167 docs(eval): record visual eval v2 diagnostics and handoff

Gates green at HEAD (verified at v3 close):
  make validate-evals-strict     OK
  make phase0-gate               OK (29 verified / 3 talks)
  make test                      OK (142 fast)
  make lint                      OK
  uv run pytest -m slow eval/tests/test_run_visual_eval.py    OK (2 passed in 769s)

Working tree (UNCOMMITTED — your first job is to triage):
  M ingest/handlers.py                            # claimed NaN/inf skip in embed_frames_handler — UNREVIEWED
  M dev/active/phase-2-week-7/handoff.md          # mid-session refresh — UNREVIEWED
  ?? eval/reports/2026-05-28_visual_eval_v4/      # methodology.mdx + summary.json + per_example.jsonl
  ?? eval/reports/2026-05-28_visual_ablations/    # ablation_data.jsonl + report.md (run_ablations.py is committed)
  ?? dev/active/phase-2-week-8/handoff.md         # this session's handoff — read first

Plus a lot of pre-existing dirty files that are OUT OF SCOPE (do not stage):
  .claude/CLAUDE.md (deletion), .claude/rules/*, .claude/skills/*, .cursor/*
  .cursorignore, .cursorindexingignore, .vscode/settings.json
  AGENTS.md, CLAUDE.md (relocations), docs/codebase-explorer.html, prompts/*
  generate/*, eval/runners/{providers,minimal_generation,measure_embeddings}.py
  scripts/run_{embeddings,bakeoff_prep_sweep}_bakeoff.py
  retrieve/visual.py, embed/colqwen.py, db/repos/frames.py, ingest/cli_all.py
  and a long list of test_*.py formatter churn

DB state (volume `pgdata` survives `make db-down`):
  3 ingested talks. The previous session re-ingested at 720p. If you trust the v4 claim:
    529 frames total (318 nXafozNIk3c + 113 W_CYk2ogcDI + 98 aie_sg_2026_d2_arize_alyx)
    514 of 529 frames carry pooled_embedding (15 NaN-skipped)
    375,734 frame_patches (was 163,990 at 360p) — verify via SQL
  eval_runs: 4 visual_eval rows (3 from v2/v3 + 1 claimed v4 at run_id `visual-eval-0acd933a1c18`)
  Query at session start to confirm:
    SELECT run_id, summary->>'visual_frame_recall_at_k' AS fr,
           summary->>'visual_chunk_tr_at_k' AS ctr,
           summary->>'visual_answer_grounding_at_k' AS vag,
           created_at::date
    FROM eval_runs WHERE code_path='visual_eval' ORDER BY created_at DESC LIMIT 5;

This session's mission is REVIEW + RESEARCH. Do not jump to implementation until the research matrix is in place.
</repo_state>

<reading_order>
Required reading before any tool work, in this order:

1. **`dev/active/phase-2-week-8/handoff.md`** — this session's brief. Lays out everything below in shorter form.
2. **`AGENTS.md`** (repo root) — current agent guidelines for this codebase.
3. **`dev/active/phase-2-week-7/handoff.md`** — the v3 close + the uncommitted v4 mid-session refresh. Note this file is one of the unreviewed v4 changes; read it as Claude's draft, not gospel.
4. **`eval/reports/2026-05-28_visual_eval_v3/methodology.mdx`** — committed v3 report. This is the trusted prior baseline.
5. **`eval/reports/2026-05-28_visual_eval_v3/integrity_check.md`** — executed v3 audit. Includes the OCR-payload TODO that still applies.
6. **`eval/reports/2026-05-28_visual_eval_v4/methodology.mdx`** — uncommitted v4 claims. Read but verify each number against the DB before quoting.
7. **`eval/reports/2026-05-28_visual_ablations/report.md`** — uncommitted ablation narrative. There is a known inconsistency: this file currently states the substrate is 360p (the deferred-state language) but the v4 handoff says it was run after 720p re-ingest. Resolve before trusting the conclusion.
8. **`eval/reports/2026-05-28_visual_ablations/ablation_data.jsonl`** — uncommitted raw per-row ablation data. The truth source for whether the K=500 prefilter rescue is real.
9. **`ingest/handlers.py`** — diff vs HEAD to see the uncommitted NaN/inf skip. Decide whether it's correct and whether a test belongs in `eval/tests/test_handlers_*.py`.
10. **`eval/runners/measure_visual.py`** + **`eval/runners/run_visual_eval.py`** — committed v3 implementation. The TODO(v4) markers reference the missing OCR audit payload.
11. **`retrieve/visual.py`** — the production 3-stage visual channel. Subject of the MaxSim-replacement option in the recommendation matrix.
12. **`docs/decisions/005-chapter-slice-ingest.md`** v3 — committed by `e7a0871`. Operator floor `yt-dlp >= 2026.3.17`.

Optional but useful:
- `eval/reports/2026-05-28_visual_diagnostics/report.md` (committed) — original per-stage attribution that motivated this whole arc.
- `eval/reports/2026-05-28_visual_diagnostics/colqwen_band_check.md` (committed) — ColQwen processor band finding.
</reading_order>

<first_actions>
Run these in order before reading anything. Verify state matches `<repo_state>`; surface any delta.

1. `git status --short --untracked-files=all`
2. `git log --oneline -12`
3. `git diff --stat`
4. `git diff HEAD -- dev/active/phase-2-week-7/handoff.md ingest/handlers.py`
5. `ls eval/reports/2026-05-28_visual_eval_v4/`
6. `ls eval/reports/2026-05-28_visual_ablations/`
7. `make db-up && make db-migrate`
8. Run the eval_runs SQL from `<repo_state>` above to confirm `visual-eval-0acd933a1c18` exists and its `summary->>'visual_answer_grounding_at_k'` matches the claimed `0.0`.
9. Sample 3 PNGs from `frames/<video_id>/` and confirm they are 1280×720 (or 640×360 — the claim is 720p but verify).
10. Read the files in `<reading_order>` in order. Do NOT spawn agents until the review is done.
</first_actions>

<hard_rules>
Non-negotiable. From CLAUDE.md, AGENTS.md, the v3 review, and the user's latest direction.

1. **`test_gold.jsonl` is untouched.** Do not load, glance at, or measure against it. The validator already enforces dev-only access.

2. **No generator + judge bakeoff (#2).** Out of scope for this session. The eval-as-product story depends on visual retrieval resolution first.

3. **The user does NOT want visual retrieval retired** and **does NOT want to blindly choose OCR**. Both are valid future paths but only after the research matrix names them with evidence.

4. **Research before implementation.** Cover at minimum: late-interaction visual document retrieval (ColPali, ColQwen/ColQwen2.5, ViDoRe, colpali-engine, vidore-benchmark); screenshot/slide/document embedding models (CLIP/SigLIP baselines, Google `multimodalembedding@001`, Gemini multimodal embeddings if available, Jina/Voyage multimodal if current); OCR/document parsing (Tesseract baseline, PaddleOCR/PaddleOCR-VL, Surya, Docling/SmolDocling, GOT-OCR style); visual rerankers / VLM judges (Qwen-VL, InternVL); hybrid patterns (frame embedding → OCR rerank, OCR-first slide indexing, image embedding + visual judge, alternative aggregators to MaxSim). Record dates and links.

5. **TeamCreate, not subagents.** When parallelizing work, use TeamCreate + named teammates. The user has been explicit about this for two sessions.

6. **Memory safety.** Mac is 24 GB unified RAM. ColQwen 2.5 loads ~7-8 GB on MPS per Python process. NEVER run more than one ColQwen-loading process at a time — the previous session demonstrated this crashes Cursor when two ML processes overlap. Sequence ColQwen loads serially. Recommended: `sudo purge` between heavy loads if swap > 7 GB.

7. **Keep unrelated working-tree changes out of any commit stack.** The full exclude list is in the week-8 handoff. Use explicit `git add <paths>`; never `git add -A`.

8. **Any future positive `VisualAnswerGrounding` result requires the structured audit payload landed first.** Minimum fields: `evaluated_frame_ids`, `evaluated_image_paths`, `ocr_excerpts` (or `visual_judge_outputs`), `matched_term`, `failure_reason`. `TODO(v4)` markers already in `measure_visual.py:562` and `run_visual_eval.py:127`.

9. **All measurements land in `eval_runs` + `eval_results`.** Aggregate in `eval_runs.summary`; per-example in `eval_results.{system_output, metrics}`. Single transaction per run, same atomicity contract as bakeoff #1.

10. **Schema first.** If a new metric requires a schema change, update `eval/schemas/`, add a fixture pair under `eval/tests/fixtures/`, run `make validate-self-test` before touching consumer code.

11. **No AI attribution in commits or docs.** Conventional commits only.

12. **Postgres stopped before final.** Always end with `make db-down`.
</hard_rules>

<mission_stack>
This session has TWO parallel concerns. Drive both via TeamCreate; sequence the work so research lands before implementation decisions.

== Concern 1 — Verify or discard the uncommitted v4 work ==

Goal: separate verified v4 facts from unreviewed claims. Decide per-file whether to commit, revise, or revert.

### Agent A — V4 review

Owns:
- `ingest/handlers.py` (uncommitted NaN/inf skip)
- `eval/reports/2026-05-28_visual_eval_v4/{methodology.mdx,summary.json,per_example.jsonl}` (uncommitted)
- `eval/reports/2026-05-28_visual_ablations/{ablation_data.jsonl,report.md}` (uncommitted)
- `dev/active/phase-2-week-7/handoff.md` (uncommitted mid-session refresh)

Tasks:
1. Verify each claimed v4 number against the DB. The `visual-eval-0acd933a1c18` row should exist with `summary->>'visual_frame_recall_at_k' = 0.0556`, `chunk_tr_at_k = 0.0556`, `answer_term_hit_at_k = 0.0`, `visual_answer_grounding_at_k = 0.0`, `visual_answer_grounding_n_evaluable = 10`. Surface any mismatch.
2. Verify frame PNG dimensions on disk (`frames/<video_id>/frame_000001.png`). Claim is 1280×720; confirm.
3. Verify pooled-frame counts per video. Claim: 306/110/98 (15 NaN-skips). Confirm with SQL.
4. **Resolve the 720p vs 360p inconsistency in `eval/reports/2026-05-28_visual_ablations/report.md`.** The file's preamble still says "Substrate: 529 frames... Frames are 360p PNGs." This is the deferred-state text that should have been updated when the script actually ran. Decision: update the report preamble + executive summary to reflect the 720p substrate the script actually queried, OR if the report is wrong about the substrate state at run time, document the discrepancy.
5. Review the `ingest/handlers.py` NaN/inf skip patch:
   - Is `np.isfinite(patches).all()` the right predicate? (Yes; isfinite catches both NaN and ±inf.)
   - Should the skip ALSO mark the frame's `ingest_step_status` row as `failed` instead of leaving it `pending`? Currently the return-without-update path silently ACKs the PGMQ message; the row may be left in an inconsistent state.
   - Is there an existing test in `eval/tests/test_handlers_*.py` that covers the embed_frames path? If yes, extend it; if no, add a minimal one with a fixture that monkeypatches `colqwen.encode_image_patches` to return a NaN matrix.
6. Decide commit-or-discard per file. Recommend a 2-3 chunk commit plan (or "discard all of it").

### Agent B — Research matrix (the user's explicit ask)

Owns: new file at `dev/active/phase-2-week-8/research_matrix.md`.

Tasks:
1. Web search across the categories listed in `<hard_rules>` §4. Record:
   - Model name + provider/repo URL
   - Last updated date (model card / release notes)
   - License
   - Hardware footprint (RAM/VRAM, MPS support yes/no)
   - Cost shape (free local / hosted per-call / hosted per-token)
   - Known performance on slide/document benchmarks (ViDoRe v1/v2, DocVQA, etc.)
   - Integration cost for LensGraph (new candidate yaml + handler vs full pipeline rewrite)
2. Be specific about ColPali/ColQwen alternatives that fix the "MaxSim flatness" problem the v4 ablation identified. Candidates to evaluate:
   - ColPali v1.3 (PaliGemma backbone) — does it have the same MaxSim issue?
   - ColQwen2.5 with a different aggregator (top-K mean, IDF-weighted sum, cross-encoder rerank over top-100)
   - Single-vector slide-tuned embedding models
   - Voyage `voyage-multimodal-3` (if current and usable on this account)
   - Gemini multimodal embeddings (if available)
   - PaddleOCR-VL / SmolDocling as OCR-rerank candidates
   - Hybrid: frame embedding for candidate gen + VLM judge (e.g. Qwen2-VL-7B locally) for top-K rerank
3. Produce a comparison matrix with: option name | model+link | LensGraph integration cost | expected eval impact | runtime cost | risk | next concrete experiment.
4. Recommend the smallest reversible experiment to run first. Cite which row of the matrix it implements.

### Agent C — Audit-payload preparation

Owns: design only (no implementation this session).

Tasks:
1. Read the TODO(v4) markers in `measure_visual.py:562` + `run_visual_eval.py:127`.
2. Sketch the schema delta for the structured audit payload in `eval/reports/2026-05-28_visual_eval_v3/integrity_check.md` §B. Required fields per `<hard_rules>` §8.
3. Decide: do these fields live in `eval_results.metrics.visual_answer_grounding` (jsonb sub-object) or in `eval_results.system_output.ocr_audit` (alongside top_k_frames)? Both are jsonb; pick one and justify.
4. Write a 1-page design doc at `dev/active/phase-2-week-8/audit_payload_design.md` ready for the agent that implements it next session.

### Team-lead (you) responsibilities

- Dispatch A/B/C in parallel via TeamCreate.
- After all three return: write `dev/active/phase-2-week-8/handoff.md` refresh + a v4 disposition section (committed / discarded / partially salvaged).
- If Agent A recommends committing any v4 files, commit them in small chunks with explicit pathspec — never `git add -A`. Suggested order if all approved:
  1. `chore(ingest): defensive NaN/inf skip in embed_frames_handler`
  2. `eval(visual): visual eval v4 report — 720p re-ingest, VisualAnswerGrounding@5 = 0/10`
  3. `eval(visual): execute prefilter_k ablation against 720p substrate`
  4. `docs(handoff): close phase-2 week-8 (visual review + research matrix)`
- If Agent A recommends discarding, `git checkout HEAD -- <paths>` for the rejected files and document why in the handoff.

== Concern 2 — Plan the smallest reversible next experiment ==

After the research matrix is in place, the team-lead picks ONE recommendation from Agent B's matrix to scope as the next implementation session. Do NOT implement it this session.

Output: a `prompts/prompt49.md` kickoff for the next session that includes:
- The chosen approach + the matrix row + the link/citations
- The exact files likely to change
- The exact commands the next session will run
- Acceptance criteria tied to the existing metrics (`VisualFrameRecall@k`, `VisualChunkTR@k`, `AnswerTermHit@k`, `VisualAnswerGrounding@k`)
- A rollback plan (every change must be reversible without losing the v3 baseline)

== Acceptance criteria for this session ==

- Verified v4 disposition documented (commit / discard / partial).
- 720p vs 360p inconsistency in ablation report resolved.
- NaN/inf skip patch reviewed and either committed with a test or rejected with documented reason.
- Research matrix shipped with at least 8 candidate approaches, each with model+link + cost + LensGraph integration cost + expected eval impact.
- `prompts/prompt49.md` kickoff for the next implementation session.
- All gates green; Postgres stopped.

== Out of scope ==

- Implementing any of the research-matrix candidates.
- Bakeoff #2 (generator + judge).
- Schema changes to `gold_example.schema.json` (the visual_evidence field already exists; new metric work waits for the next session).
- Touching the 8 text-saturated rows of `visual_gold.jsonl`.
- Re-ingest (it was done in the previous session).
</mission_stack>

<required_gates>
Before final:

```bash
make validate-evals-strict
make phase0-gate
make test
make lint
```

Slow tests are NOT required this session unless you commit code changes to `eval/runners/` (only Agent A's NaN/inf review touches non-`eval/runners/` code). If you commit anything, re-run:

```bash
uv run pytest -m slow eval/tests/test_run_visual_eval.py -q
```

Postgres stopped at end (`make db-down`).
</required_gates>

<working_agreements>
- Auto-mode default-to-action. AskUserQuestion only for genuine scope tradeoffs (e.g., "Agent A recommends committing the NaN/inf skip but says the right design is to also mark the ingest_step_status row failed — implement now or defer to next session?").
- Parallel TeamCreate dispatch for the three agents.
- Read before edit. The v3 stack is verified; don't re-litigate.
- Working-tree narrow: explicit `git add <paths>` only. Pre-existing dirty files stay unstaged.
- If you choose to do the work yourself (e.g., A/B/C agents fail or hit memory limits), document the team-vs-main-context decision explicitly in the handoff.
- Final response must include: A/B/C outcomes, v4 disposition (per file), research matrix summary, prompt49 path, commits made (if any), gates run, Postgres stopped confirmation, and the recommended next experiment.
</working_agreements>

<instructions>
Execute `<first_actions>`. Then dispatch the TeamCreate slice per `<mission_stack>` with the `<working_agreements>` constraints.

The session-close deliverable is one of:

- **Verified + research complete.** v4 disposition documented; research matrix shipped; `prompts/prompt49.md` kickoff written; all gates green. ← preferred.

- **Verified but research incomplete.** v4 disposition is done; research matrix has fewer than 8 rows because some candidate models couldn't be evaluated this session; explicitly document what's missing and why. ← acceptable.

- **Blocked.** Something in the v4 review is irreconcilable (e.g., DB rows don't match claimed numbers, suggesting the previous session's claims are wrong). Document the blocker and stop. ← acceptable as long as the blocker is clearly named.

The user's strongest signal: do NOT skip the research step. The previous session got stuck before doing it. Don't reproduce that mistake.

Bakeoff #2 stays untouched until visual retrieval is resolved. Schema changes wait for the implementation session. Begin now.
</instructions>
