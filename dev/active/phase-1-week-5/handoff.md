# Handoff — Phase 1, Week 5, Slice 3 (LangGraph generation loop + bakeoff harness)

**Written:** 2026-05-25 (last updated 2026-05-27 after slice-2 review-fix slice)
**Read by:** the slice-3 session, before any implementation
**Authoritative design:** `docs/phase-1-design.md` rev 4 (commit `e950583`)
**Implementation head:** `4c7ea94 test(cli_all): expect 4-queue drain order after slice-2A embed_frames wiring`. A trailing `docs(handoff)` closeout commit (e.g. `a7bff07`) may sit on top of that — don't pin a fresh session to an exact HEAD; just confirm `4c7ea94` is in `git log` and gates are green.

**Slice 1 commits (closed 2026-05-26):**
- `3c47a4c feat(ingest): frame-sampling worker — slice 1 of phase-1 week-5`
- `45575df fix(ingest,readiness): slice-1 review hardening`
- `645c37e chore: ruff format sweep across the tree`

**Slice 2 commits (closed 2026-05-26, review fixes 2026-05-27):**
- `3fc5ab4 feat(embed,ingest): ColQwen patches + embed_frames pipeline — slice 2A`
- `57c0c9e feat(retrieve): visual stage 2 + BGE reranker — slice 2B`
- `7eab5af fix(retrieve): reranker single-sigmoid — bypass CrossEncoder default activation`
- `9d2c851 fix(embed): ColQwen mixed-batch pooling averages only real patch rows`
- `4c7ea94 test(cli_all): expect 4-queue drain order after slice-2A embed_frames wiring`

---

## Where we are

Phase 1 is through week 4 of the design's build sequence (§8) plus slices 1 and 2 of week 5. Retrieval is feature complete: 4 text channels (BM25, BGE-M3 dense, BGE-M3 sparse, BGE-M3 multivec) + visual stage 1+2 (ColQwen pooled HNSW prefilter → frame→chunk JOIN → MaxSim refine), RRF fuses all five with per-channel rank provenance, BGE-reranker-v2-m3 cross-encoder lifts the top with single-sigmoid 0-1 confidence scores. The ingest substrate produces every input the LangGraph loop reads.

Slice 3 is the last piece of week 5: the LangGraph generation loop ending with `make answer` returning a cited answer + trace_id.

### Gates green at HEAD

```
make test                60 fast tests pass
make lint                clean
make phase0-gate         OK (11/3 verified gold across distinct talks)
make validate-evals-strict  clean
slice-2 slow suite       51 passed warm (~55s)
```

### Per-suite counts (verified against the live container)

```
test_db_migrations         12  (slow)
test_talks_repo            19  (slow)
test_pgmq_smoke            13  (slow)
test_chunks_repo            5  (slow)
test_embeds_repo            7  (slow)
test_handlers_slow         11  (slow — 5 chunk/embed_text + 3 frames + 3 embed_frames)
test_readiness             11  (slow — 6 original + 5 frame_sample exposure)
test_cli_all_slow           7  (slow — 3 fetch/readiness + 4 required-videos guard branches)
test_channels_smoke        20  (slow — 4 channels × 5 test shapes)
test_bge_m3                 6  (slow)
test_visual_prefilter       7  (slow — 4 stage-1 + 3 stage-2 channel retrieve)
test_frames                23  (slow — 13 sampler/repo + 2 absolute-path + 8 patches lifecycle)
test_colqwen_patches        3  (slow — 5 of the 8 in this file are fast)
test_rerank_functional      6  (slow — top_k monotonic, preserve-set, edge cases, provenance carry-through)
test_rerank_smoke           1  (slow — catastrophic-regression against tengyu-rag-library-analogy)
test_handlers_fast         11  (fast — 3 chunk + 5 frames + 3 embed_frames)
test_bge_m3_lazy            1  (fast)
test_rrf                   14  (fast)
test_colqwen_lazy           5  (fast)
test_colqwen_patches (fast) 6  (fast — patches/text-patches/pool resolver + lazy + mixed-batch pooling)
test_rerank_lazy            5  (fast — resolver + empty-candidates lazy + double-sigmoid regression)
```

### DB state

Persists across `make db-down`/`up` in the named volume `pgdata`.
- 3 talks ingested (`ai_engineering_v0` corpus): nXafozNIk3c, aie_sg_2026_d2_arize_alyx (chapter slice of m12vGjfbNlo), W_CYk2ogcDI.
- 212 chunks + 212 dense + 212 sparse + 73,621 chunk_token_embeds rows.
- `frames` + `frame_patches` populated only after `.mp4` files are staged under `videos/ai_engineering_v0/<source_video_id or video_id>.mp4`. None are staged today; the cli_all required-videos guard skips frames + embed_frames drains cleanly in that case, so text-only retrieval works against the populated text channels.

### `tengyu-rag-library-analogy` is the canonical exit-gate example

That `dev_gold.jsonl` example (id verbatim) is what the rerank smoke test pins against AND what the week-5 exit gate `make answer` is supposed to answer. Its question is:

> *"How does Tengyu Ma use a library analogy to compare long-context, fine-tuning, and retrieval-augmented generation?"*

Gold span: [168, 286) on video `W_CYk2ogcDI`. The span overlaps 6 chunks (7, 8, 9, 10, 11, 12) — the smoke test's "at least one of the gold-overlapping set" framing is the catastrophic-regression detector. Retrieval consistently puts 3 of the 6 in the top-5 today.

---

## What's shipped vs what's deferred

### Shipped (text + visual + reranker)

| Surface | State |
|---|---|
| `chunking/` | `fixed_window` strategy + registry; v0 only |
| `embed/bge_m3.py` | 3-channel encode (dense/sparse/multi); config-driven model id |
| `embed/colqwen.py` | full encode surface — pooled + patches + text-pooled + text-patches (one encoder path); mixed-batch pooling averages only real patch rows; config-driven model id |
| `queues/{pgmq_client,workers}.py` | send / send_batch / process_one with full transactional contract |
| `ingest/{fetch,quality,pipeline,handlers,cli,cli_all,frames,readiness}.py` | fetch + chunk + embed_text + frame_sample + embed_frames handlers; CORPUS-driven ingest-all with required-videos guard |
| `db/repos/{talks,chunks,embeds,frames,ingest_step_status}.py` | full CRUD; 1-based sparsevec; idempotent (frames upsert preserves pooled_embedding; replace_patches DELETE+INSERT) |
| `retrieve/{bm25,dense,sparse,multivec,visual,rrf,rerank}.py` | 5 channels (4 text + visual stage 1+2 chunk-level) + RRF + BGE-reranker (single-sigmoid CrossEncoder backend) |
| `scripts/bakeoff_prep.py` | per-chunk strict text readiness with N/M counts + visual `frames` column (status + count) |

### Deferred (per design §8 weeks 5+) — slice 3 owns all of it

| Step | Surface |
|---|---|
| 22-23 | ✅ **CLOSED slice 1** — `ingest/frames.py` + `db/repos/frames.py` + `frames_handler` + cli_all drain + required-videos guard |
| 24-25 | ✅ **CLOSED slice 2A** — `embed/colqwen.py` full patches + text-patches + pool helper; `db/repos/frames.py` `update_pooled` + `replace_patches`; `embed_frames_handler` + drain wiring |
| 26-27 | ✅ **CLOSED slice 2B** — `retrieve/visual.py` channel-level `retrieve()` (pooled prefilter → frame→chunk JOIN → MaxSim); `retrieve/rerank.py` (sentence-transformers CrossEncoder backend, single-sigmoid) |
| 28-29 | `eval/runners/providers.py` + `eval/runners/minimal_generation.py` + `generate/parser.py` (shared parser) — bakeoff harness |
| 30-31 | `generate/state.py` + `generate/nodes/{plan,retrieve,rerank,verify,generate,cite}.py` + `generate/trace.py` + `db/repos/traces.py` |
| 32-33 | `generate/graph.py` + `generate/api.py` + Makefile `answer` target |
| 34 | **Week-5 EXIT GATE:** `make answer QUERY=... GENERATOR=... JUDGE=...` returns a cited answer + trace_id |
| 35 | Bakeoff-prep one-off harness via `minimal_generation.generate_for_bakeoff` on all 11 dev_gold — UNLOCKS phase 2 |
| 35a | `scripts/run_embeddings_bakeoff.py` forward-contract scaffold (RED test + ~50 GREEN lines) |

Out-of-phase deferrals (not until phase 2+): worker daemon CLI, ASR worker (WhisperX), `retrieve/api.py` public surface, web UI.

---

## What's already wired for slice 3 to consume

- **`db/migrations/0005_traces.sql`** — `traces` + `trace_spans` tables already migrated. Schema in the file; `trace_spans.input/output/metadata` are `jsonb`; spans are partitioned conceptually by `(trace_id, node_name, iteration)`. Per design §5: spans store `chunk_ids: list[int]`, NOT chunk text — the phase-4 viewer joins `chunks` on read.
- **`db/migrations/0004_eval_runs.sql`** — `eval_runs` already migrated; the bakeoff harness writes a row with `code_path='minimal_generation'`.
- **`eval/config/model_candidates.yaml`** — generator candidates (gemma-4-31b, qwen3-235b-a22b-instruct, deepseek-v3.2), judge candidates (same three but cross-family enforced), cheap_extraction candidates (gemma-4-e4b, qwen3-8b local; fallback-to-chosen-generator). `candidates.judge.minimums.cross_family_required: true` is the contract `CrossFamilyViolationError` enforces.
- **`docs/phase-1-design.md` §5 (LangGraph loop)** — node-by-node contract, `AgentState` TypedDict shape, Verify→Retrieve loop with max 2 iterations, `serialize_state_snapshot` helper, deterministic Plan path as default.
- **`retrieve/{bm25,dense,sparse,multivec,visual,rrf,rerank}.py`** — all callable from the `retrieve` and `rerank` nodes with no further glue. `FusedResult` from `rrf.fuse()` is the input shape for `rerank()`; `RerankedResult` carries `rerank_score` (0-1) + `rrf_score` (provenance).

---

## First commands the new session MUST run

In order. Stop if any is red.

```bash
git log --oneline -10
# Confirm 4c7ea94 (test(cli_all): expect 4-queue drain order) appears.
# Trailing docs(handoff) closeout commits may sit on top — that's expected.

make validate-evals-strict
make phase0-gate
make test
make lint
# All four must pass (60 fast tests).
```

Slices 0, 1, 2 are all CLOSED. No prerequisite slow-suite re-run unless you've changed code under `embed/`, `db/migrations/`, `ingest/`, `retrieve/visual.py`, or `retrieve/rerank.py`.

---

## Reading order — load before writing slice-3 code

1. **`docs/phase-1-design.md` §5** — LangGraph loop, node-by-node contract. This is the slice-3 north star.
2. **`docs/phase-1-design.md` §8 steps 28-35a** — implementation order + per-step RED/GREEN.
3. **`docs/decisions/004-model-selection.md` v3.1** — no-silent-defaults; candidate sets; cross-family rule.
4. **`.claude/CLAUDE.md` hard rules** — every word.
5. **`eval/config/model_candidates.yaml`** — bakeoff candidate IDs that `GENERATOR=` / `JUDGE=` / `PLANNER=` values must match.
6. **`db/migrations/0005_traces.sql`** — the schema your `db/repos/traces.py` writes into.
7. **`retrieve/rerank.py`** — the closest existing parallel for a config-driven yaml-resolver pattern wrapping a provider/library.
8. **`eval/tests/test_rerank_lazy.py`** — the fast-test pattern for monkeypatched models + sigmoid-pipeline regression assertion.

Skim only (stable):
- `db/migrations/{0001..0004}*.sql` — schema is frozen through week 5
- `eval/corpora/ai_engineering_v0/*.jsonl` — corpus is locked

---

## Slice 3 — LangGraph generation loop + bakeoff harness (design §8 steps 28-35a)

Two parallel teammates on disjoint files (team-from-the-top per hard rule 12). Orchestrator (you, in main context) verifies end-to-end against the live `make answer` exit gate and commits.

### Teammate A — bakeoff harness + shared parser

Files owned (exclusive):
- `eval/runners/providers.py` — DeepInfra HTTP client. Env-only auth (`DEEPINFRA_API_KEY` from `os.environ`, never logged). Exponential backoff retry on 429/5xx; capped at ~3 retries; timeout per call. Returns `ProviderResponse` (raw_text + latency_ms + provider_model_id echo). No global state; the client is a small dataclass constructed per call site (or per process via a tiny factory) so mocking in tests just monkeypatches the factory.
- `generate/parser.py` — `parse_generation_output(raw_str) -> GenerationOutput`. `GenerationOutput` is a Pydantic model (per design §5 "All AgentState values that aren't primitives MUST be Pydantic `BaseModel`") with fields per the design spec (answer, claims, citations, parse_ok, raw_response, error). Shared between `minimal_generation.py` and the LangGraph `generate` node.
- `eval/runners/minimal_generation.py` — internal bakeoff harness. NOT the product path (the LangGraph graph is). One function `generate_for_bakeoff(example, generator_candidate, *, conn) -> EvalRunRow` (or similar). Writes one `eval_runs` row with `code_path='minimal_generation'`. Used by step 35 for the dev_gold sweep.

Tests:
- `eval/tests/test_providers.py` — fast tests with mocked HTTP (e.g. `httpx.MockTransport` or `monkeypatch.setattr` on `requests.post`); cover retry-on-429, retry-on-5xx, give-up after N, env-var-missing raises, env-var-present-but-empty raises. NO real network calls in pytest. Provider model id resolution must read from `eval/config/model_candidates.yaml`.
- `eval/tests/test_parser.py` — happy-path well-formed output, malformed JSON sets `parse_ok=False`, raw_response always present even on failure, citation `answer_claim_index` references model's own claims.
- `eval/tests/test_minimal_generation.py` (slow gate per design step 28) — exercise the harness with a mocked provider returning canned text; assert `eval_runs` row written with `code_path='minimal_generation'`, parse_ok bubbled up, latency captured.

### Teammate B — LangGraph nodes + state + tracing + graph + answer API

Files owned (exclusive):
- `db/repos/traces.py` — repo for `traces` + `trace_spans` tables. `start_trace`, `end_trace`, `append_span` (batched flush at graph completion per design §5 "Spans batch-flush ... one `INSERT ... VALUES (...), (...)` per trace"). Spans store `chunk_ids: list[int]`, not chunk text.
- `generate/state.py` — `AgentState(TypedDict, total=False)` per design §5. All non-primitive values are Pydantic `BaseModel`. `serialize_state_snapshot(state) -> dict` for `trace_spans.input/output` jsonb columns.
- `generate/trace.py` — span emission helpers; integrates with `db/repos/traces.py`.
- `generate/nodes/plan.py` — deterministic Plan path is DEFAULT (no-LLM classify-and-degenerate). LLM Plan via `cheap_extraction` candidate is opt-in (env or arg). Per design §5 + ADR 004 v3.1.
- `generate/nodes/retrieve.py` — calls `retrieve.{bm25,dense,sparse,multivec,visual}.retrieve()` then `retrieve.rrf.fuse()`.
- `generate/nodes/rerank.py` — calls `retrieve.rerank.rerank()` on the fused candidates.
- `generate/nodes/verify.py` — cross-family judge; reads candidate from `model_candidates.yaml::candidates.judge`; confidence threshold from env var per design §5. Returns "loop" or "exit" decision.
- `generate/nodes/generate.py` — calls Teammate A's `providers.deepinfra` + `parser.parse_generation_output`.
- `generate/nodes/cite.py` — VALIDATION, not snapping (per design §5 finding #9). Emits `final.valid_citations` + `final.invalid_citations`; phase-2 CitationAccuracy = `len(valid) / len(valid + invalid)`.
- `generate/graph.py` — `build_graph()` assembling nodes + conditional edges (Verify → Retrieve loop, MAX 2 ITERATIONS).
- `generate/api.py` — public `answer(query, *, corpus_id="ai_engineering_v0", generator_candidate=None, judge_candidate=None, planner_candidate=None)`. `load_bakeoff_winner_or_raise()` raises `BakeoffNotYetRunError` when both generator and judge omitted AND no `winner_locked` row exists. `CrossFamilyViolationError` raises when generator and judge share a family.
- `Makefile` `answer` target — passes `QUERY` / `GENERATOR` / `JUDGE` / optional `PLANNER` env vars through to `python -m generate.api`.

Tests:
- `eval/tests/test_traces.py` (slow) — `start_trace` / `append_span` / `end_trace` round-trip against the populated DB.
- `eval/tests/test_generate_nodes.py` (slow) — each node tested in isolation with stub `AgentState` inputs; state transitions correct; trace spans emitted; spans store `chunk_ids: list[int]`, not text.
- `eval/tests/test_plan_deterministic.py` (fast) — no-LLM Plan path classifies + degenerates correctly without touching a model.
- `eval/tests/test_generate_graph.py` (slow) — end-to-end on one dev_gold example with mocked generator + judge. Assert: (a) trace + spans persisted, (b) Verify→Retrieve loop fires when judge returns low confidence, (c) iteration cap stops at 2, (d) abstention path produces `final.abstain=True`, (e) `BakeoffNotYetRunError` raised when generator/judge omitted AND no winner_locked row, (f) `CrossFamilyViolationError` raised when generator+judge share a family.

---

## Hard rules — non-negotiable

From CLAUDE.md + ADRs + the design's binding constraints. Slice-3-relevant rules promoted to top of mind:

1. **LangGraph only.** No LangChain core, no LlamaIndex (ADR 001). The graph is `langgraph.StateGraph(AgentState)`.
2. **Postgres only for traces.** `traces` + `trace_spans` tables in the existing DB; no Langfuse, no external tracing backend, no Redis (ADR 002).
3. **No silent model defaults.** `make answer` MUST require explicit `GENERATOR=` and `JUDGE=` flags until bakeoff winners are locked (ADR 004 v3.1).
4. **Cross-family judge.** Enforced from `model_candidates.yaml::candidates.judge.minimums.cross_family_required: true` plus per-candidate `family` field. `CrossFamilyViolationError` when generator and judge share a family.
5. **All model + provider IDs from `eval/config/model_candidates.yaml`.** No hardcoded `"deepinfra"`, no hardcoded `"google/gemma-4-31B-it"`, no hardcoded `"deepseek-ai/DeepSeek-V3.2"`. Resolver pattern: mirror `embed.bge_m3._resolve_model_id` / `embed.colqwen._resolve_model_id` / `retrieve.rerank._resolve_model_id`. Grep test in CI: `grep -r "gemma-4-31B\|qwen3-235b\|deepseek-v3.2" generate/ eval/runners/` should return zero lines.
6. **Provider tests mocked unless running the final `make answer` exit gate.** Pytest never makes real DeepInfra calls. The exit gate (operator-run) is the one place real network traffic happens.
7. **`make phase0-gate` stays green at every commit boundary.** Re-run before each commit.
8. **Conventional commits.** `feat(scope):` / `fix(scope):` / `refactor(scope):` / `docs:` / `eval:` / `chore:`. No AI attribution lines.
9. **No new framework / datastore / provider / model library without an ADR amendment in the same commit.**

---

## Known gotchas (still load-bearing for slice 3)

- **pgvector sparsevec is 1-based on wire, 0-based on the SparseVector Python API.** Don't shift in our wrappers.
- **`pgmq.send_batch` returns SETOF bigint** (a row per id), NOT a single `bigint[]`. Use `fetchall()` + unwrap `r[0]`.
- **`psycopg` adapts `list[Jsonb]` as a TEXT-array literal of JSON strings**, not `jsonb[]`. Pre-serialise with `json.dumps` to `list[str]` and cast `::jsonb[]` server-side. This will matter for batched `trace_spans` inserts.
- **`websearch_to_tsquery` uses AND-of-all-terms.** `retrieve/bm25.py` already has the OR fallback; don't re-introduce websearch.
- **Reranker uses `sentence_transformers.CrossEncoder`, not FlagEmbedding.** And the activation pipeline is `activation_fn=torch.nn.Identity()` + one `torch.sigmoid`. If a future fix touches `retrieve/rerank.py`, keep both invariants.
- **ColQwen `encode_image_pooled` averages only real patch rows** (true_counts from the attention_mask). Don't regress to `patches.mean(axis=1)` on the padded tensor — that bug shipped briefly in slice 2 and was caught in review.

### One stale comment to fix opportunistically

`embed/colqwen.py::encode_image_pooled`'s zero-patch defensive branch comment claims "the HNSW NULL-pooled WHERE clause filter it downstream" — wrong. A zero vector is NOT NULL; `WHERE pooled_embedding IS NOT NULL` won't catch it (an all-zeros row would still appear in the prefilter and produce undefined cosine distance against any non-zero query). Fix the comment ONLY if you touch `embed/colqwen.py` for some unrelated slice-3 reason. Do NOT open the file just for the comment.

---

## Cost reminder (for `make answer` once it lands)

Per design §9 risk 8: each `make answer` query is ~$0.001–0.002 of DeepInfra cost (Plan + Verify×N + Generate). The ADR 004 budget of $18-33 for 12 weeks counts bakeoff traffic only. If `make answer` enters routine use (>50 queries/day sustained), rate-limit or cache-by-query-hash before the demo runs the bill up.

---

## Where to push back if asked to deviate

Don't silently bend the design. If a slice plan needs to violate any of:
- Postgres-only invariant
- LangGraph-only invariant
- No-silent-defaults rule
- Cross-family judge discipline
- Model-IDs-from-yaml rule

Surface explicitly, propose an ADR amendment, and wait for the user's "approve amendment" signal. The 4-rev'd design + 3 plan-review passes are the contract; deviations move through the same process.
