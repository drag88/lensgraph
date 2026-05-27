# Handoff — Phase 1 Week 5 CLOSED · Phase 2 on deck (Bakeoff #1 embeddings)

**Closed:** 2026-05-27 (slice 3 + live EXIT GATE + step-35 sweep all green)
**Status:** ✅ All week-5 slices and the design step-34 EXIT GATE artifact shipped. Step 35 wrote 11/11 eval_runs rows with `code_path='minimal_generation'`, which is the contract that **UNLOCKS phase 2**.
**Read by:** the phase-2 session, before any bakeoff code.
**Authoritative design:** `docs/phase-1-design.md` rev 4 (commit `e950583`).

---

## TL;DR

Phase 1 is done. The LangGraph generation loop is live, the trace store is populated, the bakeoff harness is wired, and a real `make answer` call against DeepInfra returned a cited answer routed through the full Plan → Retrieve → Rerank → Verify → Generate → Cite path (6 spans, status='completed', 7 valid citations). The minimal_generation sweep wrote one `eval_runs` row per dev_gold + synthesis example (11 total, 11/11 parse_ok, 1/11 abstain) — phase-2's generator+judge bakeoff has the substrate it needs.

Phase 2 week 6 starts with the embeddings bakeoff. Slice 35a shipped a scaffold (`scripts/run_embeddings_bakeoff.py`); phase 2 swaps the synthetic measurements for a real TimestampRecall@5 sweep per channel, applies the ADR 004 selection rule, and calls `db.repos.eval_runs.lock_bakeoff_winner('text_embeddings', ...)`.

---

## Phase-1 week-5 commits (this session)

Slice 3, atomic per concern:

- `feat(eval,generate,db): bakeoff harness + shared parser + eval_runs repo (slice 3, Teammate A)`
- `feat(generate,db): LangGraph nodes + graph + answer() + traces repo (slice 3, Teammate B)`
- `feat(scripts): embeddings-bakeoff scaffold (step 35a) + dev_gold sweep driver (step 35)`
- `fix(eval): refresh Qwen3-235B-A22B-Instruct provider_model_id to -2507 suffix`
- `docs(handoff): close phase-1 week-5 — slice 3 shipped, EXIT GATE passed, step 35 unlocks phase 2`

Slices 1 + 2 closeout commits live in earlier handoffs.

---

## Gates green at HEAD

```
make test                 84 fast tests pass (+24 from slice 3)
make lint                 clean
make phase0-gate          OK (11 verified non-negative examples across 3 talks)
make validate-evals-strict 0 warnings
slice-3 slow suite        27 passed (~3 min warm)
  - test_eval_runs_repo        5 slow
  - test_traces                4 slow
  - test_minimal_generation    3 slow
  - test_generate_nodes        9 slow
  - test_generate_graph        6 slow (a–f sub-assertions)
  - test_run_embeddings_bakeoff 1 slow
```

Per-suite counts (verified live against the container):

```
Slice 3 NEW fast suites (in addition to all prior slice-1/2 fast tests):
  test_providers              11  (fast — yaml resolver pair, mocked httpx, env-key guard, 429/5xx retry)
  test_parser                  6  (fast — happy path, malformed, code-fence salvage, abstain shape)
  test_plan_deterministic      7  (fast — synthesis cues, speaker mentions, deterministic-only)

Slice 3 NEW slow suites:
  test_eval_runs_repo          5  (slow — insert_run, insert_result, lock_bakeoff_winner idempotent, load_bakeoff_winner MRU)
  test_traces                  4  (slow — start/end/flush round-trip; batched jsonb[] cast)
  test_minimal_generation      3  (slow — mocked provider; eval_runs row written; ProviderError propagates)
  test_generate_nodes          9  (slow — plan/retrieve/rerank/verify/generate/cite isolated)
  test_generate_graph          6  (slow — (a) trace+spans persisted (b) loop fires (c) iter cap stops at 2
                                          (d) abstention path (e) BakeoffNotYetRunError (f) CrossFamilyViolationError)
  test_run_embeddings_bakeoff  1  (slow — winner_locked=true row written for text_embeddings)
```

---

## EXIT GATE artifact (design step 34)

```
$ make answer QUERY="how does Tengyu Ma compare long context, fine-tuning, and RAG" \
              GENERATOR=qwen3-235b-a22b-instruct \
              JUDGE=deepseek-v3.2

trace_id: 019e6757-2e52-7af9-947a-c242c2b8f224
abstain: False
iterations: 0
latency_ms: 89877

Tengyu Ma compares long context, fine-tuning, and RAG by drawing analogies to
human learning ... He views long context as inefficient, likening it to scanning
an entire library for every question ... Fine-tuning, in his view, is like memorizing
an entire library ... In contrast, he favors RAG, seeing it as a more efficient
and practical approach.

valid_citations (7):
  - W_CYk2ogcDI @ 174.6-204.9 → chunk 8
  - W_CYk2ogcDI @ 149.6-180.7 → chunk 7
  - nXafozNIk3c @ 2775.5-2804.0 → chunk 157
  - (4 more)
```

`SELECT ... FROM trace_spans WHERE trace_id = '019e6757-2e52-7af9-947a-c242c2b8f224'`:

```
 n | node_name | iteration | status | sec
---+-----------+-----------+--------+-----
 1 | plan      |         0 | ok     |   0
 2 | retrieve  |         0 | ok     |  32
 3 | rerank    |         0 | ok     |  13
 4 | verify    |         0 | ok     |   7
 5 | generate  |         0 | ok     |  38
 6 | cite      |         0 | ok     |   0
```

6 spans across 1 iteration — exactly the design's "6 spans / 1 iteration in the happy case" target. Loop didn't fire because judge returned `confidence=0.8` on the first verify ("Chunks contain direct comparisons of long context, fine-tuning, and RAG, including personal opinions and analogies."). Reranker landed gold chunks 7 + 8 (in the gold-overlapping set [7, 8, 9, 10, 11, 12]) in the top-8 — matches the rerank smoke test's catastrophic-regression detector.

---

## Step 35 sweep — minimal_generation across all 11 phase-0 examples

```
sweep_id=sweep-20260527T025945  generator=qwen3-235b-a22b-instruct
n_rows | n_parse_ok | n_abstain | p50_ms | p95_ms
--------+------------+-----------+--------+--------
    11 |         11 |         1 |  49939 | 87584
```

11/11 `parse_ok=True`. One abstain (`synth-scaffolding-around-imperfect-llms` — model couldn't synthesize across two speakers, which is legitimate signal, NOT a code bug). Per-example claim+citation counts: 3-7 each.

Eval-runs row shape: `code_path='minimal_generation'`, `chunking_strategy='fixed_window'`, `embedding_model_id='bge-m3-all-channels'`, `generator_model_id='qwen3-235b-a22b-instruct'`, `summary={component, candidate_id, parse_ok, latency_ms, abstain}`. Eval-results row shape: `system_output=<full GenerationOutput>`, `metrics={parse_ok, latency_ms, abstain, claim_count, citation_count}`.

This is the contract phase 2's generator+judge bakeoff iterates on. The "winner" is NOT this run — it's whatever phase-2 picks via the ADR 004 selection rule.

Cost: ~$0.011 against DeepInfra (11 calls × ~$0.001). Wall-clock: ~13 min (10 dev_gold sweep batched + 1 synthesis one-off).

Total live-DeepInfra spend this session: ~$0.015 (3 EXIT GATE attempts at ~$0.001 each plus the sweep). All under the ADR 004 $18–33 12-week budget.

---

## What's shipped vs deferred

### Shipped (week 5 closed)

| Surface | State |
|---|---|
| `eval/runners/providers.py` | DeepInfra HTTP client; env-only auth; httpx.MockTransport-tested; exp backoff retry on 429/5xx; ProviderError on 4xx (non-429) and malformed-200 |
| `eval/runners/minimal_generation.py` | Bakeoff harness; one provider call → one eval_runs row + one eval_results row; shared parser; 90s timeout |
| `generate/parser.py` | Pydantic models + parse_generation_output; code-fence salvage; parse_ok=False on any failure (raw_response always populated) |
| `db/repos/eval_runs.py` | insert_run, insert_result, lock_bakeoff_winner (idempotent), load_bakeoff_winner (MRU) |
| `db/repos/traces.py` | start_trace, end_trace, flush_spans (batched ::jsonb[] cast idiom from queues/pgmq_client.py) |
| `generate/state.py` | AgentState TypedDict + all Pydantic models (RetrievedChunk, GeneratorCandidate, JudgeCandidate, AnswerResult, ValidatedCitation, InvalidCitation, ...) |
| `generate/trace.py` | new_uuid7 (RFC 9562 inline), serialize_state_snapshot, span context manager |
| `generate/nodes/{plan,retrieve,rerank,verify,generate,cite}.py` | Six nodes; plan defaults to deterministic regex+speaker heuristic (no silent LLM); verify is cross-family judge; cite VALIDATES (not snaps) |
| `generate/graph.py` | LangGraph StateGraph wiring; per-node span wrapping; verify_router gates on (low conf + refined_query set); cap in verify (current_iter < 2) — single source of truth |
| `generate/api.py` | answer() coroutine + CLI; BakeoffNotYetRunError; CrossFamilyViolationError; resolver pair calls into providers._resolve_candidate_{provider,family,model_id}; copy-pasteable error messages |
| `Makefile` | `answer` target (env-passthrough: QUERY, GENERATOR, JUDGE, optional PLANNER) |
| `scripts/run_embeddings_bakeoff.py` | Step-35a scaffold; reads JSON measurements; applies ADR 004 selection rule; writes winner_locked row |
| `scripts/run_bakeoff_prep_sweep.py` | Step-35 driver; walks dev_gold.jsonl + synthesis.jsonl; runs 5-channel retrieve + RRF + rerank + minimal_generation per example |

### Deferred — explicitly out of phase 1

- `eval/runners/judges/` versioned prompts (phase-2 generator+judge bakeoff will need them; not slice-3 scope).
- Async LangGraph (`ainvoke`) — slice 3 uses `.invoke` per Makefile-target simplicity. Phase-4 streaming will revisit.
- OpenRouter failover wiring (design §6 "engages only when `LENSGRAPH_PROVIDER_FAILOVER=1`" — not slice 3).
- `retrieve/api.py` public surface — generate/nodes/retrieve.py calls channels + RRF directly; the dedicated module is phase-4 UI work.
- Per-component code_path filter on `load_bakeoff_winner`: design §5 SQL filters `code_path='minimal_generation'` for generator/judge winners. Today's behaviour is component-keyed via `summary->>'component'` which is unique enough across the existing code_paths; phase-2 can add the code_path filter as a kwarg if a future code_path conflict surfaces. Not load-bearing today (no `winner_locked` rows exist yet).

---

## YAML drift fix (this session)

The live EXIT GATE surfaced one drift: DeepInfra now publishes `Qwen/Qwen3-235B-A22B-Instruct` only under the dated `-2507` suffix. The un-suffixed id 404s. Fixed in `eval/config/model_candidates.yaml` for BOTH generator AND judge entries; `last_updated` bumped to 2026-05-27. `prices_checked_at` left at 2026-05-24 (no price re-check this session — re-verify before phase-2 bakeoff #2).

Working pattern: when a model 404s at runtime, query `https://api.deepinfra.com/v1/openai/models` with the API key and match the candidate's canonical name against the published list. Updating `provider_model_id` is a yaml-only fix (NOT an ADR amendment) — the candidate `id` is the ADR 004-locked contract, only the wire routing string can drift.

---

## Substrate state (resumes across `make db-down`/`up` via named volume `pgdata`)

- 3 talks ingested in `ai_engineering_v0`: nXafozNIk3c, aie_sg_2026_d2_arize_alyx (chapter slice of m12vGjfbNlo), W_CYk2ogcDI.
- 212 chunks · 212 dense_embeds · 212 sparse_embeds · 73,621 chunk_token_embeds.
- `frames` / `frame_patches` empty (no `.mp4` files staged today). Visual retrieval still works at the API level — gracefully returns `[]` when no frames exist.
- `traces`: 4 rows (3 EXIT GATE attempts; 1 successful with 6 spans).
- `eval_runs`: 13 rows total (2 smoke runs + 11 step-35 sweep, all `code_path='minimal_generation'`).
- `eval_results`: 13 rows (one per eval_runs).

---

## Hard rules — still load-bearing for phase 2

1. **LangGraph only.** (ADR 001.) `langgraph.graph.StateGraph` is the graph contract. `langchain-core` is a transitive dep — that's accepted; do NOT author LangChain LCEL chains directly.
2. **Postgres only.** (ADR 002.) No Langfuse, no Redis. The trace + trace_spans tables are the observability backend.
3. **No silent model defaults.** (ADR 004 v3.1.) `answer()` raises `BakeoffNotYetRunError` when caller omits generator OR judge AND no `winner_locked` row exists. Until phase-2's bakeoffs land winners, callers MUST pass `GENERATOR=` and `JUDGE=`.
4. **Cross-family judge enforced at runtime.** `CrossFamilyViolationError` when `generator.family == judge.family`. Sourced from `candidates.<component>.options[*].family` in `model_candidates.yaml`.
5. **All model + provider IDs from `eval/config/model_candidates.yaml`.** Resolver pair in `eval.runners.providers`: `_resolve_provider_model_id`, `_resolve_candidate_family`, `_resolve_candidate_provider`. Zero hardcoded literals in `generate/`, `eval/runners/`, `scripts/`, `db/repos/`. Pre-commit grep guard:
   `grep -rE "qwen3-235|deepseek-v3|gemma-4-31" generate/ eval/runners/ scripts/ db/repos/ | grep -v ".pyc"`
   MUST return zero matches.
6. **Provider tests mocked.** `httpx.MockTransport` only in pytest. The operator-run `make answer` is the only place real network traffic happens.
7. **`make phase0-gate` stays green at every commit.**
8. **Iteration cap = 2 enforced inside verify.** The router checks `(low conf + refined_query is not None)` only — `refined_query=None` is the verify-side signal that the cap fired. Single source of truth.

---

## Phase 2 week 6 — what's next

The week-6 deliverable is **Bakeoff #1: text embeddings** (ADR 004 v3.1 + design §8 step 35a).

Substrate already in place:
- `scripts/run_embeddings_bakeoff.py` scaffolds the runner (synthetic measurements today; phase 2 swaps for real TimestampRecall@5).
- `eval_runs` + `eval_results` tables ready.
- `db.repos.eval_runs.lock_bakeoff_winner('text_embeddings', 'bge-m3-all-channels', run_id=...)` is the lock contract; `load_bakeoff_winner(component='text_embeddings')` reads it.
- BGE-M3 is the only realistic v0 candidate today (Voyage / Gemini Embedding 2 are fallbacks gated on BGE-M3 failing the 0.75 TimestampRecall@5 minimum).

Phase 2 week 6 tasks (in order):
1. Replace `scripts/run_embeddings_bakeoff.py` synthetic measurements with a real per-channel TimestampRecall@5 measurement against `dev_gold.jsonl`.
2. Run it; lock the winner.
3. Phase 2 week 6 task 2: build the generator+judge bakeoff runner that:
   - Reuses `minimal_generation.generate_for_bakeoff` (no new product surface).
   - For each generator candidate ∈ {gemma-4-31b, qwen3-235b-a22b-instruct, deepseek-v3.2}: run all dev_gold; for each, score via cross-family judges; lock the winner per ADR 004 selection rule.
   - Adds judge prompts in `eval/runners/judges/` (content-hashed; logged in summary.json per CLAUDE.md experiment-tracking rules).
4. Re-check DeepInfra pricing + provider_model_ids before any bakeoff run (the Qwen3 drift fix is a sample of what tends to break).

Cost ceiling for bakeoff #1 (embeddings): $0 (BGE-M3 is local). Bakeoff #2 (generator+judge): ~$2 per ADR 004's estimate.

---

## Stale comment to fix opportunistically

`embed/colqwen.py::encode_image_pooled`'s zero-patch defensive branch comment claims "the HNSW NULL-pooled WHERE clause filter it downstream" — wrong (a zero vector is NOT NULL; `WHERE pooled_embedding IS NOT NULL` won't catch it). Fix only when another reason brings you into that file — don't open it just for the comment.

---

## First commands the phase-2 session MUST run

```bash
git log --oneline -10
# Confirm the slice-3 commits appear at the top (look for the feat(eval,generate,db)
# Teammate-A commit and the feat(generate,db) Teammate-B commit).

make validate-evals-strict
make phase0-gate
make test
make lint
# All four must pass (84 fast tests).

make db-up
make db-migrate
```

Then read `docs/decisions/004-model-selection.md` v3.1 selection rule + `scripts/run_embeddings_bakeoff.py` scaffold + this handoff's "Phase 2 week 6" section. After that, start replacing the synthetic measurements with a real per-channel TimestampRecall@5 sweep.
