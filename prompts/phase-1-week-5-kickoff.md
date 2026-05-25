<role>
You are a senior AI engineer joining LensGraph in a fresh session. Phase 1's text retrieval + visual prefilter stage 1 + RRF fusion are shipped. Your job this week is design §8 weeks 5: frames worker → visual stage 2 + reranker → LangGraph generation loop. Three slices, sized one-per-session, in that order. The design is paid for in blood across 4 revisions + 3 plan-review passes — do not re-litigate; consult the handoff for what's settled.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
HEAD: f2fd3a2 docs(handoff): phase-1 week-5 session handoff + kickoff prompt
  (or later — a fix(embed) ColQwen-config commit may land on top before slice 0)

Last 10 commits (context, do not re-litigate):
  f2fd3a2 docs(handoff): phase-1 week-5 session handoff + kickoff prompt
  c199a2f feat(retrieve): visual prefilter stage 1 — ColQwen pooled + cosine HNSW
  baffa55 fix(retrieve): RRF returns FusedResult with channel_ranks provenance
  3f3f1c9 feat(retrieve): RRF fusion module + closed-form synthetic tests
  82c0d47 fix(retrieve): BM25 OR-fallback for natural-language queries
  5806e6f feat(retrieve): four text retrieval channels (bm25 + dense + sparse + multivec)
  9c716e5 fix(ingest): per-chunk readiness + cli_all exit codes + Makefile CORPUS
  5016064 feat(ingest): chunk + embed_text handlers + ingest-all CLI + bakeoff tokens
  8036813 feat(db,queues): chunks + embeds repos + pgmq.send_batch
  6b2f921 fix(embed): config-driven model id + sparse normalization + lazy-load test + decouple migration test
  37fd978 feat(embed): BGE-M3 lazy singleton with three-channel encode

Gates green at HEAD (verified end of last session):
  make validate-evals-strict   OK (0 warnings, 11 verified gold)
  make phase0-gate             OK (11/3 across 3 talks)
  make test                    OK (36 fast tests)
  make lint                    OK

DB state (resumes from named volume pgdata after make db-up):
  3 talks ingested (ai_engineering_v0 corpus)
  212 chunks + 212 dense + 212 sparse + 73,621 chunk_token_embeds rows
  frames table EMPTY (frame-sampling worker is your slice 1 deliverable)

One open verification gate from the prior session: `eval/tests/test_visual_prefilter.py` (4 slow tests) was committed but the end-to-end run timed out mid-ColQwen download. Run it FIRST this session per <first_actions>.
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

== Slice 0 — close the visual stage-1 verification gate ==

Before any week-5 work, exercise the deferred slow test from last session:

  make db-up
  make db-migrate
  uv run pytest eval/tests/test_visual_prefilter.py -q -m slow   # ~10-15 min first time (model dl + cold load)
  make db-down

If the 4 tests pass, proceed to Slice 1. If any fails, triage `embed/colqwen.py` (most likely colpali-engine API drift since 0.3.16 — adjust import path; or POOLED_DIM mismatch, in which case the runtime assertion raises with the actual dim). Fix in a focused commit `fix(embed): ...` and re-run. Do NOT proceed until visual stage 1 is green.

== Slice 1 — Frame sampler worker (design §8 steps 22-23) ==

Functional surface:
- `ingest/frames.py` — public `sample(video_path, *, every_sec=10.0) -> list[Frame]`. Subprocess to ffmpeg; writes JPEG/PNG to a stable per-video directory; computes sha256 per frame; returns Frame dataclasses (matches the frames table FK shape).
- `ingest/handlers.py::frames_handler(conn, payload)` — mirrors `embed_text_handler`'s shape. Payload `{video_id, step:"frame_sample", entity_id:0}`. Reads talk's video path (you'll need to extend talks.yaml or use the transcript_path directory + a sibling .mp4 convention — your call, document in commit body), runs `frames.sample()`, inserts frames rows. **Does NOT compute pooled embeddings yet** — that fans out to a separate step (see slice 2).
- `ingest/cli_all.py` — drain the ingest_frames queue after ingest_chunk and ingest_embed_text.

RED: `eval/tests/test_frames.py` — synthetic short video (use ffmpeg `lavfi` source: `ffmpeg -f lavfi -i color=c=red:s=320x240:d=60` to generate a 60s test asset, or bundle a fixture). Sample at 10s intervals → 6 frames. sha256 stable across runs.

Verification artifact:
  make ingest VIDEO_ID=W_CYk2ogcDI                # already works
  # …new: frames are sampled + inserted; ingest_step_status flips fetch=done, frame_sample=pending → drained → completed
  make bakeoff-prep                                # now shows frame_sample status per video

Out of scope this slice: pooled embedding population (the embeds.upsert_pooled call), patches (the encode_image_patches function).

== Slice 2 — Visual stage 2 (patches + MaxSim) + reranker (design §8 steps 24-27) ==

Three concerns, one slice:

- `embed/colqwen.py::encode_image_patches(images) -> np.ndarray (N, num_patches, 128)`. Per-design §4: per-token storage, MaxSim aggregation. Slot-fit with the existing `encode_image_pooled` signature.
- `embed/colqwen.py::pool_patches(patches) -> np.ndarray (128,)`. Internal helper consumed by encode_image_pooled to maintain consistency between stages.
- A new ingest step `embed_frames` that fans out per frame_id: claims, runs `encode_image_patches` for one frame, writes `frame_patches` rows + computes pooled by mean-of-patches and updates `frames.pooled_embedding`. Handler: `ingest/handlers.py::embed_frames_handler`.
- `retrieve/visual.py` extended with stage 2: take stage-1 top-K (default 30) prefilter, fetch each frame's patches, run MaxSim against the query token embeddings (you'll need ColQwen text-patches too, not just pooled — extend `encode_text_query_patches`).
- `retrieve/rerank.py` — BGE-reranker-v2-m3 (CPU local per tech-stack). Public `rerank(conn, query, candidates: list[FusedResult], *, top_k=8) -> list[RerankedResult]`. Score is a 0-1 cross-encoder confidence.

Functional tests only — no quality assertions yet. The catastrophic-regression smoke gate per design §8 step 26 (`test_rerank_smoke.py::test_known_gold_chunk_in_top_5` for the tengyu-rag-library-analogy) IS in scope.

Out of scope: quality metric logging (`scripts/log_retrieval_quality.py`) — that's an artifact step, not a build step.

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

1. git log --oneline -5
   Top should be: c199a2f feat(retrieve): visual prefilter stage 1 — ColQwen pooled + cosine HNSW

2. make validate-evals-strict     # 0 warnings, 11 verified examples
3. make phase0-gate                # OK (11/3)
4. make test                       # 36 passed (was 33 before ColQwen-config lazy tests)
5. make lint                       # clean

6. Read dev/active/phase-1-week-5/handoff.md (the operational guide).

7. Close the open visual gate (slice 0):
     make db-up
     make db-migrate
     uv run pytest eval/tests/test_visual_prefilter.py -q -m slow
   ~10-15 min on first run (ColQwen2.5 v0.2 ~14GB download + cold load).
   4 passes required before slice 1.

8. Skim docs/phase-1-design.md §4, §5, §8 weeks 5.

Only when 1-7 are green: begin slice 1 (frames worker).
</first_actions>

<instructions>
Execute <first_actions>. Verify everything green INCLUDING the deferred visual slow suite. Then ship the three week-5 slices in order per <mission_stack>.

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
