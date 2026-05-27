<role>
You are a senior AI engineer joining LensGraph in a fresh session. Phase 1's text retrieval, visual stage 1+2 (pooled prefilter + per-patch MaxSim), RRF fusion, frame-sampling worker, embed_frames pipeline, and BGE reranker are all shipped. Your job this session is the last slice of design §8 week 5: the LangGraph generation loop, ending with the `make answer` exit-gate artifact. The design is paid for in blood across 4 revisions + 3 plan-review passes — do not re-litigate; consult the handoff for what's settled.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
Implementation head: 57c0c9e feat(retrieve): visual stage 2 + BGE reranker — slice 2B
  Trailing docs(handoff) / docs/test closeout commits may sit on top —
  confirm 57c0c9e is in `git log`, don't pin to an exact HEAD.

Recent commits (context, do not re-litigate):
  57c0c9e feat(retrieve): visual stage 2 + BGE reranker — slice 2B
  3fc5ab4 feat(embed,ingest): ColQwen patches + embed_frames pipeline — slice 2A
  6ee5719 docs(handoff): soften remaining implementation-head check
  c55327d docs(handoff),test(cli_all): slice-1 closeout
  fd335a3 docs(handoff): close slice 1 + slot slice 2 as next
  645c37e chore: ruff format sweep across the tree
  45575df fix(ingest,readiness): slice-1 review hardening
  3c47a4c feat(ingest): frame-sampling worker — slice 1 of phase-1 week-5

Gates green at HEAD (verified after slice 2 + review fixes):
  make validate-evals-strict   OK (0 warnings, 11 verified gold)
  make phase0-gate             OK (11/3 across 3 talks)
  make test                    OK (60 fast tests)
  make lint                    OK
  slice-2 slow suite           51 passed (cli_all_slow + full slice-2 suites)

DB state (resumes from named volume pgdata after make db-up):
  3 talks ingested (ai_engineering_v0 corpus)
  212 chunks + 212 dense + 212 sparse + 73,621 chunk_token_embeds rows
  frames + frame_patches populated only once .mp4 files are staged under
    videos/ai_engineering_v0/ (none staged as of this writing — the
    cli_all required-videos guard skips frames + embed_frames drains
    cleanly in that case)

Slice 0 (visual stage 1) — CLOSED 2026-05-25.
Slice 1 (frame-sampling worker) — CLOSED 2026-05-26.
Slice 2 (visual stage 2 + reranker) — CLOSED 2026-05-26.
Slice 3 (LangGraph generation loop + bakeoff harness) is the only thing
left this week and the week-5 EXIT GATE.

One deviation from the design worth flagging: the BGE reranker uses
`sentence_transformers.CrossEncoder`, NOT FlagEmbedding's FlagReranker —
FlagEmbedding 1.4.0 calls `tokenizer.prepare_for_model` which was removed
in transformers 5.x (we're pinned at 5.9.0). Same checkpoint + scoring;
only the loader changes. See 57c0c9e commit body.
</repo_state>

<reading_order>
Required reading before writing any code, in this order:

1. dev/active/phase-1-week-5/handoff.md — the operational guide for this week. Reads in ~5 minutes; covers shipped surface, gates, gotchas, mission stack.

2. docs/phase-1-design.md — focus on:
     §4 (retrieval API + RRF semantics)
     §5 (LangGraph loop — node-by-node contract, the slice-3 north star)
     §8 weeks 5 (steps 22–35a — your three slices)

3. docs/decisions/004-model-selection.md v3.1 — no-silent-defaults; generator/judge candidate sets.

4. .claude/CLAUDE.md hard rules — every word. Especially: Postgres-only, LangGraph-only, no silent model defaults, conventional commits.

5. eval/config/model_candidates.yaml — bakeoff candidate IDs that GENERATOR / JUDGE / PLANNER values must match.

Skim only (stable):
  docs/PRD.md, docs/eval-methodology.md, docs/architecture.md, docs/roadmap.md
  docs/decisions/001-langgraph-not-llamaindex.md, 002-postgres-only.md, 003-conference-talks-corpus.md
  eval/curation/* (mission-1 territory)
  db/migrations/*.sql (schema is frozen through week 5)
</reading_order>

<hard_rules>
Non-negotiable. From CLAUDE.md, the ADRs, and the design's binding constraints.

1. Postgres only. PGMQ for queues, pgvector for embeddings. No Redis / Celery / RabbitMQ / second store (ADR 002).
2. LangGraph only. No LangChain core, no LlamaIndex (ADR 001).
3. No silent model defaults. `make answer` MUST require explicit GENERATOR= and JUDGE= flags pre-bakeoff (ADR 004 v3.1).
4. Cross-family judge discipline. Curation family ≠ judge family ≠ generator family.
5. `test_gold.jsonl` is locked. SHA256 in README is a contract.
6. `make phase0-gate` stays green at every commit boundary. Re-run before each commit.
7. Conventional commits — feat(scope): / fix(scope): / refactor(scope): / docs: / eval: / chore:. No AI attribution.
8. No new framework / datastore / provider / model library without an ADR amendment in the same commit.
9. The 1-based-on-wire / 0-based-on-Python pgvector sparsevec contract is documented in db/repos/embeds.py — don't shift in our wrappers.
10. `pgmq.send_batch` returns SETOF bigint (use fetchall + unwrap), not bigint[]. Documented in queues/pgmq_client.py.
11. BGE-M3 colbert_vecs strips trailing EOS but keeps leading CLS — multi[i].shape[0] == len(input_ids) - 1. Documented in eval/tests/test_bge_m3.py.
12. Use TeamCreate for substantive slices. Two parallel teammates per slice (disjoint files); orchestrator verifies + commits.
</hard_rules>

<mission_stack>
Three slices, take them in order. Each verifies before the next begins. Stop after each slice and report. Do not preempt the next slice without explicit user signal.

== Slice 0 — CLOSED 2026-05-25 ==

`eval/tests/test_visual_prefilter.py` 4/4 slow tests passed in 38.08s. ColQwen2.5 v0.2 resolved from `eval/config/model_candidates.yaml`.

== Slice 1 — CLOSED 2026-05-26 ==

Shipped across `3c47a4c feat(ingest)` + `45575df fix(ingest,readiness)` + `645c37e chore`. Frames substrate is end-to-end:

- `ingest/frames.py` — `sample(video_path, *, video_id, every_sec, start_sec, duration_sec, out_dir)` via ffmpeg subprocess, PNG + bitexact for stable sha256. `frame_sec` is RELATIVE to talk zero (chapter slices use parent `source_video_id` window). `default_video_path_for_talk()` handles BOTH relative and absolute transcript paths via `Path(...).parent.name`.
- `db/repos/frames.py` — idempotent upsert on `UNIQUE(video_id, frame_sec)`; `pooled_embedding` preserved under re-run (slice 2 owns that column).
- `ingest/handlers.py::frames_handler` — persists frames + fans out `embed_frames` status rows + `ingest_embed_frames` messages.
- `ingest/cli_all.py` — drains `ingest_frames` after `ingest_chunk`+`ingest_embed_text`. Required-videos guard enumerates expected `.mp4` paths (deduped on chapter-slice parents) and branches: no visual talks → skip exit 0; zero staged → skip exit 0; partial staged → skip + force exit 1 + print missing; all staged → drain.
- `ingest/readiness.py` — `VideoReadiness.frame_sample_status` + `frames_n` informational fields (NOT part of `complete`); `scripts/bakeoff_prep.py` shows them under the new `frames` column.

Slow suite: `test_frames.py` 15, `test_handlers_slow.py::*frames*` 3, `test_cli_all_slow.py` 6, `test_readiness.py` 11. All green.

== Slice 2 — CLOSED 2026-05-26 ==

Shipped across `3fc5ab4 feat(embed,ingest)` + `57c0c9e feat(retrieve)`.
ColQwen full patches + embed_frames pipeline (Teammate A) + visual stage 2
chunk-level `retrieve()` + BGE reranker via `sentence_transformers.CrossEncoder`
(Teammate B; FlagEmbedding 1.4.0 broken under transformers 5.x — same
checkpoint, different loader). Catastrophic-regression smoke against
`tengyu-rag-library-analogy` passes pre and post rerank.

Slow suite: `test_colqwen_patches.py` 3, `test_frames.py` 23,
`test_handlers_slow.py` 11, `test_visual_prefilter.py` 7,
`test_rerank_functional.py` 6, `test_rerank_smoke.py` 1 = 51 slow passing
in ~55s warm.

== Slice 3 — LangGraph generation loop + bakeoff harness (design §8 steps 28-35a) ==

The week-5 EXIT GATE. Three sub-pieces, all in this slice:

1. **Minimal generation harness + shared parser** (steps 28-29):
   - `eval/runners/providers.py` — DeepInfra HTTP client (exponential backoff retry; env-only auth)
   - `eval/runners/minimal_generation.py` — internal bakeoff harness (NOT product path)
   - `generate/parser.py` — `parse_generation_output(raw_str) -> GenerationOutput` (shared with the LangGraph generate node in step 31)
   - RED + slow tests with mocked provider per design §8 step 28

2. **LangGraph nodes + state + tracing** (steps 30-31):
   - `generate/state.py` — `AgentState` TypedDict (total=False so adding fields is non-breaking)
   - `generate/nodes/{plan,retrieve,rerank,verify,generate,cite}.py`
   - `generate/trace.py` — span emission to `traces` + `trace_spans`
   - `db/repos/traces.py` — repo for both tables
   - The deterministic Plan path (no-LLM classify-and-degenerate) is the default; LLM Plan via `cheap_extraction` is the opt-in (per ADR 004 candidate set).

3. **Graph + answer API** (steps 32-33):
   - `generate/graph.py` — build_graph() assembling nodes + conditional edges (Verify → Retrieve loop, max 2 iterations)
   - `generate/api.py` — public `answer(query, *, corpus_id="ai_engineering_v0", generator_candidate=None)`. `load_bakeoff_winner_or_raise()` raises `BakeoffNotYetRunError` when GENERATOR/JUDGE omitted AND no winner_locked row exists. `CrossFamilyViolationError` raised when generator+judge share a family.
   - `make answer QUERY=... GENERATOR=... JUDGE=... [PLANNER=...]`

End-of-slice artifact (the week-5 exit gate):
  make answer QUERY="how does Tengyu Ma compare long context, fine-tuning, and RAG" \
              GENERATOR=qwen3-235b-a22b-instruct \
              JUDGE=deepseek-v3.2
  # Returns cited answer + trace_id; SELECT * FROM trace_spans WHERE trace_id=... shows Plan→Retrieve→Rerank→Verify→Generate→Cite

Plus step 35: one-off bakeoff harness run via `minimal_generation.generate_for_bakeoff` on all 11 dev_gold examples with one generator candidate (arbitrarily selected, NOT "the winner") — produces `eval_runs` row with code_path='minimal_generation'. This UNLOCKS phase 2.

Plus step 35a: scaffold `scripts/run_embeddings_bakeoff.py` per the embeddings-bakeoff forward contract (RED test asserts the lock_bakeoff_winner call; GREEN is ~50 lines).
</mission_stack>

<cdf_workflow>
Three slices, each a /cdf:tdd cycle.

  /cdf:tdd — RED → GREEN → REFACTOR per slice. The slow tests requiring real model loads (ColQwen patches, BGE-reranker, BGE-M3) are the GREEN gates.
  /cdf:verify — before each commit. validate-evals-strict + phase0-gate + test + lint.
  /cdf:git — atomic commits per concern within a slice. Conventional commits format.
  /cdf:troubleshoot — if a slow test fails, surface the failure cause before patching.

Avoid this week:
  /cdf:design — the design at e950583 is the contract; don't re-design.
  /cdf:plan-review — 3 passes are settled; surface true design gaps, don't silently amend.
</cdf_workflow>

<working_agreements>
- Team-from-the-top: spawn TeamCreate at slice start with 2 teammates on disjoint files. Orchestrator (you, in main context) verifies end-to-end and commits.
- Auto-mode default-to-action. Use AskUserQuestion only when the user must own a scope tradeoff. No multi-round questioning.
- Parallel tools for independent operations. Never serialize independent file reads.
- Read before edit. No speculation on files you haven't opened.
- Only changes the mission requires. No surrounding cleanup, no premature abstractions, no scope creep.
- The design at e950583 is the contract. Surface gaps; never silently amend.
- Before any commit touching db/, queues/, ingest/, chunking/, retrieve/, generate/, embed/, api/, web/: run `make phase0-gate` locally.
- If a hook or test fails, investigate root cause. Never bypass with --no-verify.
- Untracked files that already exist at session start (`.claude/rules/*`, `AGENTS.generated.md`, `CLAUDE.generated.md`, `prompts/codebase-explorer.md`, `docs/codebase-explorer.html`, `.cursor/`) are out of scope. Don't stage them.
</working_agreements>

<first_actions>
Run these in order before any code. Verify state matches <repo_state>; if it does not, surface the delta before proceeding.

1. git log --oneline -10
   Confirm 57c0c9e (feat(retrieve): visual stage 2 + BGE reranker) appears.
   Trailing docs(handoff) / docs/test closeout commits may sit on top —
   that's expected; don't pin to an exact HEAD.

2. make validate-evals-strict     # 0 warnings, 11 verified examples
3. make phase0-gate                # OK (11/3)
4. make test                       # 58 passed
5. make lint                       # clean

6. Read dev/active/phase-1-week-5/handoff.md (the operational guide).

7. Skim docs/phase-1-design.md §5 (LangGraph loop — node-by-node contract,
   the slice-3 north star) and §8 weeks 5 steps 28-35a.

Slices 0, 1, AND 2 are all CLOSED — no need to re-run their slow suites
unless you've changed code under `embed/`, `db/migrations/`, `ingest/`,
`retrieve/visual.py`, or `retrieve/rerank.py`.

Only when 1-5 are green: begin slice 3 (LangGraph loop + bakeoff harness).
</first_actions>

<instructions>
Execute <first_actions>. Then ship the remaining week-5 slices (2 and 3) in order per <mission_stack>.

The execution rhythm per slice:
  1. Spawn a TeamCreate team with 2 teammates on disjoint files.
  2. Wait for both teammate completion messages.
  3. Verify end-to-end against the live container (make ingest-all where relevant).
  4. Commit atomically per the slice's concerns (multiple commits OK within a slice).
  5. Run make phase0-gate + make test + make lint after each commit — all must stay green.
  6. Stop at slice boundary. Report state. Wait for explicit user signal before starting the next slice.

The week-5 EXIT GATE is `make answer QUERY=...` returning a cited answer + trace_id at the end of slice 3 — that's when phase 1 hits the design's stated artifact. After that, the next session moves into phase 2 (bakeoff #1 prep).

If you discover a fact that contradicts the design at e950583, surface it explicitly. Do not silently work around it. ADR amendments go through the same process the design did.

Begin now.
</instructions>
