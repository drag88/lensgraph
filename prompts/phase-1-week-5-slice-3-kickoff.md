<role>
You are a senior AI engineer joining LensGraph in a fresh session. Phase 1's text retrieval (BM25 + BGE-M3 dense/sparse/multivec), visual stage 1+2 (ColQwen pooled prefilter + per-patch MaxSim), RRF fusion, frame-sampling worker, embed_frames pipeline, and BGE-reranker-v2-m3 cross-encoder are all shipped. Your job this session is the LAST slice of design §8 week 5: the LangGraph generation loop + bakeoff harness, ending with the `make answer` exit-gate artifact. The design is paid for in blood across 4 revisions + 3 plan-review passes — do not re-litigate; consult the handoff for what's settled. Stop at the slice-3 boundary; do NOT start phase 2.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
Implementation head: 4c7ea94 test(cli_all): expect 4-queue drain order after slice-2A embed_frames wiring
  Trailing docs(handoff) closeout commits may sit on top (e.g. a7bff07) —
  confirm 4c7ea94 is in `git log`, don't pin to an exact HEAD.

Recent commits (context, do NOT re-litigate):
  4c7ea94 test(cli_all): expect 4-queue drain order after slice-2A embed_frames wiring
  9d2c851 fix(embed): ColQwen mixed-batch pooling averages only real patch rows
  7eab5af fix(retrieve): reranker single-sigmoid — bypass CrossEncoder default activation
  ed3c795 docs(handoff): close slice 2 + slot slice 3 as next
  57c0c9e feat(retrieve): visual stage 2 + BGE reranker — slice 2B
  3fc5ab4 feat(embed,ingest): ColQwen patches + embed_frames pipeline — slice 2A
  645c37e chore: ruff format sweep across the tree
  45575df fix(ingest,readiness): slice-1 review hardening
  3c47a4c feat(ingest): frame-sampling worker — slice 1 of phase-1 week-5

Gates green at HEAD (verified after slice 2 + review fixes):
  make validate-evals-strict   OK (0 warnings, 11 verified gold)
  make phase0-gate             OK (11/3 across 3 talks)
  make test                    OK (60 fast tests)
  make lint                    OK
  slice-2 slow suite           51 passed (test_colqwen_patches + test_frames +
                                          test_handlers_slow + test_visual_prefilter +
                                          test_rerank_functional + test_rerank_smoke)

DB state (resumes from named volume pgdata after make db-up):
  3 talks ingested (ai_engineering_v0 corpus)
  212 chunks + 212 dense + 212 sparse + 73,621 chunk_token_embeds rows
  frames + frame_patches populated only after .mp4 files are staged under
    videos/ai_engineering_v0/<source_video_id or video_id>.mp4
  (none staged today — text-only retrieval works against the populated
  text channels, which is sufficient for the slice-3 exit gate because
  the tengyu-rag-library-analogy example is transcript-only by modality).

Slices 0, 1, 2 — all CLOSED. Slice 3 (LangGraph generation loop + bakeoff
harness) is the only thing left this week and the week-5 EXIT GATE.

Slice-2 review deviations worth knowing (don't regress these):
  - Reranker uses sentence_transformers.CrossEncoder (NOT FlagEmbedding —
    FlagEmbedding 1.4.0 calls tokenizer.prepare_for_model removed in
    transformers 5.x). Activation pipeline is activation_fn=Identity()
    + one torch.sigmoid. See retrieve/rerank.py + commits 57c0c9e/7eab5af.
  - encode_image_pooled averages only real patch rows via the shared
    _encode_images_with_true_counts helper. See embed/colqwen.py + 9d2c851.
</repo_state>

<reading_order>
Required reading before writing any slice-3 code, in this order:

1. dev/active/phase-1-week-5/handoff.md — slice-3 operational guide. Reads
   in ~10 minutes; covers shipped surface, what's already wired for slice 3
   to consume, hard rules, and the teammate split.

2. docs/phase-1-design.md §5 (LangGraph loop — node-by-node contract,
   AgentState shape, Verify→Retrieve loop, serialize_state_snapshot helper).
   This is the slice-3 north star.

3. docs/phase-1-design.md §8 weeks 5 steps 28-35a (implementation order +
   per-step RED/GREEN).

4. docs/decisions/004-model-selection.md v3.1 — no-silent-defaults;
   generator/judge candidate sets; cross-family rule.

5. .claude/CLAUDE.md hard rules — every word. Especially: Postgres-only,
   LangGraph-only, no silent model defaults, model IDs from yaml.

6. eval/config/model_candidates.yaml — candidates.generator,
   candidates.judge (cross_family_required: true), candidates.cheap_extraction
   (for the LLM Plan path). The GENERATOR / JUDGE / PLANNER values used at
   call time MUST match candidate ids here.

7. db/migrations/0005_traces.sql — the schema your db/repos/traces.py
   writes into. trace_spans.input/output/metadata are jsonb; spans store
   chunk_ids: list[int], not chunk text.

8. retrieve/rerank.py + eval/tests/test_rerank_lazy.py — the closest
   parallel for the yaml-resolver + mocked-fast-test pattern you'll mirror
   for providers.py.

Skim only (stable):
  docs/PRD.md, docs/eval-methodology.md, docs/architecture.md
  docs/decisions/001-langgraph-not-llamaindex.md, 002-postgres-only.md
  db/migrations/{0001..0004}*.sql (schema frozen through week 5)
</reading_order>

<hard_rules>
Non-negotiable. From CLAUDE.md, ADRs, and the design's binding constraints.
Slice-3-relevant rules promoted to top of mind:

1. LangGraph only. No LangChain core, no LlamaIndex (ADR 001). The graph
   is langgraph.StateGraph(AgentState).
2. Postgres only for traces. traces + trace_spans tables in the existing
   DB; no Langfuse, no external tracing backend, no Redis (ADR 002).
3. No silent model defaults. `make answer` MUST require explicit
   GENERATOR= AND JUDGE= flags until bakeoff winners are locked
   (ADR 004 v3.1). load_bakeoff_winner_or_raise raises
   BakeoffNotYetRunError when both are omitted AND no winner_locked row
   exists in eval_runs.
4. Cross-family judge enforced from
   model_candidates.yaml::candidates.judge.minimums.cross_family_required: true
   plus per-candidate `family` field. CrossFamilyViolationError raises
   when generator and judge share a family.
5. ALL model + provider IDs from eval/config/model_candidates.yaml. No
   hardcoded "deepinfra", no hardcoded "google/gemma-4-31B-it", no
   hardcoded "deepseek-ai/DeepSeek-V3.2", no hardcoded "Qwen/Qwen3-235B-A22B-Instruct".
   Resolver pattern: mirror embed.bge_m3._resolve_model_id /
   embed.colqwen._resolve_model_id / retrieve.rerank._resolve_model_id.
   Grep check before commit: `grep -r "gemma-4-31B\|qwen3-235b\|deepseek-v3.2"
   generate/ eval/runners/` should return zero lines.
6. Provider tests mocked. Pytest NEVER makes real DeepInfra calls. The
   operator-run `make answer` exit gate is the one place real network
   traffic happens — and only after explicit operator invocation.
7. make phase0-gate stays green at every commit boundary.
8. Conventional commits — feat(scope): / fix(scope): / refactor(scope): /
   docs: / eval: / chore:. No AI attribution.
9. No new framework / datastore / provider / model library without an
   ADR amendment in the same commit.
10. Use TeamCreate for substantive slices. Two parallel teammates on
    disjoint files; orchestrator verifies + commits.
11. The 1-based-on-wire / 0-based-on-Python pgvector sparsevec contract
    is documented in db/repos/embeds.py — don't shift in our wrappers.
12. pgmq.send_batch returns SETOF bigint (use fetchall + unwrap), not
    bigint[]. Documented in queues/pgmq_client.py. Same shape applies if
    you batch-flush trace_spans.
</hard_rules>

<mission_stack>
ONE slice this session. Stop at the slice boundary; do NOT preempt phase 2.

== Slice 3 — LangGraph generation loop + bakeoff harness (design §8 steps 28-35a) ==

Two parallel teammates on disjoint files (team-from-the-top). Orchestrator
(you, in main context) verifies end-to-end against the live `make answer`
exit gate and commits atomically per concern.

--- Teammate A — bakeoff harness + shared parser ---

Files owned (exclusive):

* eval/runners/providers.py — DeepInfra HTTP client.
    - Env-only auth: DEEPINFRA_API_KEY read from os.environ; raise a clear
      error when missing or empty. Never log the key.
    - Exponential backoff retry on 429 and 5xx; capped at ~3 retries with
      jittered sleep. Permanent failures (4xx other than 429, malformed
      response) raise immediately.
    - Per-call timeout. Returns a ProviderResponse dataclass with
      raw_text + latency_ms + provider_model_id echo.
    - Provider model id resolved from model_candidates.yaml — never
      hardcoded.
    - No global state. Small dataclass constructed per call site (or a
      tiny factory) so tests just monkeypatch the factory or the HTTP
      transport.

* generate/parser.py — parse_generation_output(raw_str) -> GenerationOutput.
    - GenerationOutput is a Pydantic BaseModel (design §5: all non-primitive
      AgentState values are Pydantic). Fields per design spec: answer,
      claims, citations (with answer_claim_index referencing the model's
      own claims), parse_ok, raw_response, error.
    - Malformed JSON sets parse_ok=False; raw_response is always present
      even on failure.
    - SHARED between minimal_generation.py and the LangGraph generate node
      — single parser path, no drift.

* eval/runners/minimal_generation.py — internal bakeoff harness.
    - generate_for_bakeoff(example, generator_candidate, *, conn) — runs
      the prompt against one provider call, parses, writes one eval_runs
      row with code_path='minimal_generation'.
    - NOT the product path. The LangGraph graph (Teammate B's surface)
      is the product path. This harness is what step 35 uses for the
      dev_gold sweep that unlocks phase 2.

Tests:
* eval/tests/test_providers.py (fast, mocked HTTP — e.g. monkeypatch the
  factory or use httpx.MockTransport). Cover retry-on-429,
  retry-on-5xx, give-up after N, env-var-missing raises, env-var-empty
  raises. NO real network calls in pytest.
* eval/tests/test_parser.py (fast). Happy path well-formed output;
  malformed JSON sets parse_ok=False; raw_response always present;
  citation answer_claim_index references model's own claims.
* eval/tests/test_minimal_generation.py (slow per design step 28, but
  with mocked provider — the "slow" marker is about real DB writes via
  eval_runs, not real network). Exercise the harness with a mocked
  provider returning canned text; assert eval_runs row written with
  code_path='minimal_generation', parse_ok bubbled up, latency captured.

--- Teammate B — LangGraph nodes + state + tracing + graph + answer API ---

Files owned (exclusive):

* db/repos/traces.py — repo for traces + trace_spans.
    - start_trace(trace_id, query, corpus_id, generator_model_id) -> row insert.
    - append_span(trace_id, span) buffered in memory; flushed at graph
      completion via one INSERT ... VALUES (...), (...) per trace
      (design §5 "Spans batch-flush at graph completion").
    - end_trace(trace_id, status, final_answer, final_citations,
      iterations, latency_ms) -> row update + span flush.
    - Spans store chunk_ids: list[int], NOT chunk text. The phase-4
      viewer joins chunks on read.
    - Honor the psycopg list[Jsonb] gotcha: serialise per-span jsonb
      payloads with json.dumps to list[str] and cast ::jsonb[]
      server-side for the batched INSERT (mirrors queues.pgmq_client.send_batch).

* generate/state.py — AgentState(TypedDict, total=False) per design §5.
    - All non-primitive values are Pydantic BaseModel (per design §5
      "All AgentState values that aren't primitives MUST be Pydantic
      BaseModel"): RetrievedChunk, GenerationOutput, Citation,
      AnswerClaim, GeneratorCandidate, JudgeCandidate,
      CheapExtractionCandidate, AnswerResult.
    - serialize_state_snapshot(state) -> dict for trace_spans.input/output
      jsonb columns. Replaces non-serialisable values with their model_dump().

* generate/trace.py — span emission helpers; integrates with db/repos/traces.py.

* generate/nodes/plan.py — DETERMINISTIC Plan path is the DEFAULT
  (no-LLM classify-and-degenerate per design + ADR 004 v3.1 finding #3).
  LLM Plan via cheap_extraction candidate is opt-in via PLANNER= flag.
  Either way, no silent LLM default.

* generate/nodes/retrieve.py — calls retrieve.{bm25,dense,sparse,multivec,visual}.retrieve()
  then retrieve.rrf.fuse(). All five channels are available; the visual
  channel returns [] gracefully if frames/frame_patches are empty (today's
  state without staged .mp4s).

* generate/nodes/rerank.py — calls retrieve.rerank.rerank() on the
  fused candidates. RerankedResult carries rerank_score (0-1 sigmoid)
  + rrf_score (provenance).

* generate/nodes/verify.py — cross-family judge. Reads candidate from
  model_candidates.yaml::candidates.judge. Confidence threshold from env
  var (design §5 — name it VERIFY_CONFIDENCE_THRESHOLD or similar).
  Returns "loop" or "exit" decision.

* generate/nodes/generate.py — calls Teammate A's providers.deepinfra +
  parser.parse_generation_output. NO direct HTTP here; all provider
  traffic goes through providers.py.

* generate/nodes/cite.py — VALIDATION, not snapping (design §5 finding #9).
  Emits final.valid_citations + final.invalid_citations; phase-2
  CitationAccuracy = len(valid) / len(valid + invalid).

* generate/graph.py — build_graph() assembling nodes + conditional
  edges. Verify → Retrieve loop with MAX 2 ITERATIONS. Iteration cap
  is non-negotiable (design §5 + cost reminder).

* generate/api.py — public answer(query, *, corpus_id="ai_engineering_v0",
  generator_candidate=None, judge_candidate=None, planner_candidate=None)
  -> AnswerResult (with trace_id).
    - load_bakeoff_winner_or_raise() raises BakeoffNotYetRunError when
      both generator and judge omitted AND no winner_locked row exists
      in eval_runs.
    - CrossFamilyViolationError raises when generator and judge share a
      family.

* Makefile `answer` target — passes QUERY / GENERATOR / JUDGE / optional
  PLANNER env vars through to `python -m generate.api`. Exits nonzero on
  BakeoffNotYetRunError or CrossFamilyViolationError so CI sees the gap.

Tests:
* eval/tests/test_traces.py (slow). start_trace / append_span / end_trace
  round-trip against the live DB; batched-flush INSERT semantics
  exercised with multiple spans per trace.
* eval/tests/test_generate_nodes.py (slow). Each node tested in isolation
  with stub AgentState inputs; state transitions correct; trace spans
  emitted; spans store chunk_ids: list[int], not text.
* eval/tests/test_plan_deterministic.py (fast). No-LLM Plan path
  classify-and-degenerate behavior; covers single_clip + synthesis +
  negative-scope branches.
* eval/tests/test_generate_graph.py (slow). End-to-end on one dev_gold
  example with mocked generator + judge. Assert:
    (a) trace + spans persisted (Plan→Retrieve→Rerank→Verify→Generate→Cite).
    (b) Verify→Retrieve loop fires when judge returns low confidence.
    (c) iteration cap stops at 2.
    (d) abstention path produces final.abstain=True.
    (e) BakeoffNotYetRunError raised when generator/judge omitted AND
        no winner_locked row exists.
    (f) CrossFamilyViolationError raised when generator+judge share a
        family.

--- Step 35a (Teammate A OR B — your call; tiny) ---

* scripts/run_embeddings_bakeoff.py forward-contract scaffold.
* eval/tests/test_run_embeddings_bakeoff.py::test_lock_writes_winner_locked_true_row
  (RED) — asserts the script reads channel measurements, applies the
  ADR 004 selection rule (cheapest within 3pp meeting minimums), and
  calls db.repos.eval_runs.lock_bakeoff_winner('text_embeddings',
  'bge-m3', run_id=...). GREEN is ~50 lines. Phase 2 consumes; phase 1
  ships the scaffolding.

--- End-of-slice EXIT GATE artifact (the design step 34 deliverable) ---

  make db-up
  make db-migrate
  make answer QUERY="how does Tengyu Ma compare long context, fine-tuning, and RAG" \
              GENERATOR=qwen3-235b-a22b-instruct \
              JUDGE=deepseek-v3.2

Required behavior:
  - Returns a cited answer + trace_id on stdout.
  - SELECT * FROM trace_spans WHERE trace_id = '<id>' ORDER BY started_at
    shows the full Plan → Retrieve → Rerank → Verify → Generate → Cite
    path (6 spans across 1 iteration in the happy case; 12 spans across
    2 iterations if Verify loops once).

Plus step 35: one-off run of minimal_generation.generate_for_bakeoff on
all 11 dev_gold examples with ONE generator candidate (arbitrarily
selected — NOT "the winner"). Produces eval_runs rows with
code_path='minimal_generation'. UNLOCKS phase 2.
</mission_stack>

<execution_rhythm>
1. Spawn TWO teammates in parallel (Agent calls in a single message) on the
   disjoint file sets above. Brief each agent fully — they don't see your
   conversation history; the prompt must be self-contained.
2. Each teammate runs `make lint` (NOT `make fmt` — that cascades formatter
   ripples across the tree and confuses the other agent's diff) and the
   targeted fast tests for their slice. Each REPORTS without committing.
3. Wait for BOTH teammate completion notifications. No polling — the
   harness notifies you.
4. Orchestrator verifies end-to-end:
     - read the touched files
     - make test + make lint + make phase0-gate (fast gates)
     - bring DB up if down: make db-up + make db-migrate
     - run the targeted slow tests for each teammate's surface
     - run the EXIT GATE: make answer QUERY=... GENERATOR=qwen3-235b-a22b-instruct
       JUDGE=deepseek-v3.2 (real DeepInfra call; ~$0.001-0.002)
     - verify trace_spans shows the full path
5. Commit atomically per concern (Teammate A's harness; Teammate B's nodes
   + graph; step 35a scaffold; docs(handoff) refresh). Conventional commits;
   no AI attribution.
6. STOP at the slice-3 boundary. Report state. Wait for explicit user
   signal before starting phase 2.
</execution_rhythm>

<working_agreements>
- Auto-mode default-to-action. Use AskUserQuestion only when the user
  must own a scope tradeoff (e.g. "found a design gap; amend ADR or work
  around?"). No multi-round questioning.
- Parallel tools for independent operations. Never serialize independent
  file reads.
- Read before edit. No speculation on files you haven't opened.
- Only changes the mission requires. No surrounding cleanup, no premature
  abstractions, no scope creep.
- The design at e950583 is the contract. Surface gaps explicitly; never
  silently amend.
- Before any commit touching db/, queues/, ingest/, chunking/, retrieve/,
  generate/, embed/, api/, web/: run `make phase0-gate` locally.
- If a hook or test fails, investigate root cause. Never bypass with
  --no-verify.
- Untracked files that already exist at session start (`.claude/rules/*`,
  `AGENTS.generated.md`, `CLAUDE.generated.md`, `prompts/codebase-explorer.md`,
  `docs/codebase-explorer.html`, `.cursor/`) are out of scope. Don't stage them.
- One opportunistic comment fix: embed/colqwen.py::encode_image_pooled's
  zero-patch defensive branch comment claims "the HNSW NULL-pooled WHERE
  clause filter it downstream" — wrong, a zero vector is NOT NULL. Fix
  the comment ONLY if a slice-3 reason brings you into embed/colqwen.py.
  Do NOT open the file just for the comment.
</working_agreements>

<first_actions>
Run these in order before any code. Verify state matches <repo_state>;
if it does not, surface the delta before proceeding.

1. git log --oneline -10
   Confirm 4c7ea94 (test(cli_all): expect 4-queue drain order) appears.
   Trailing docs(handoff) closeout commits may sit on top — that's
   expected; don't pin to an exact HEAD.

2. make validate-evals-strict     # 0 warnings, 11 verified examples
3. make phase0-gate                # OK (11/3)
4. make test                       # 60 passed
5. make lint                       # clean

6. Read dev/active/phase-1-week-5/handoff.md (the slice-3 operational guide).

7. Skim docs/phase-1-design.md §5 + §8 steps 28-35a.

Only when 1-5 are green: spawn the two teammates per <mission_stack>.
</first_actions>

<instructions>
Execute <first_actions>. Then ship Slice 3 per <mission_stack> with the
<execution_rhythm>.

The week-5 EXIT GATE is `make answer QUERY=...` returning a cited answer +
trace_id at the end of this slice. That artifact closes phase 1 week 5
and is the hand-off point to phase 2 (bakeoff #1 prep).

If you discover a fact that contradicts the design at e950583, surface it
explicitly. Do not silently work around it. ADR amendments go through the
same process the design did.

Begin now.
</instructions>
