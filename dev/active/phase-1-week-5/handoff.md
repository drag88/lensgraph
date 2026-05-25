# Handoff — Phase 1, Week 5 (frames worker + reranker + LangGraph)

**Written:** 2026-05-25, end of week-4 retrieval session
**Read by:** the next session, before any implementation
**Authoritative design:** `docs/phase-1-design.md` rev 4 (commit `e950583`)
**Branch head:** `c199a2f` `feat(retrieve): visual prefilter stage 1 — ColQwen pooled + cosine HNSW`

---

## Where we are

Phase 1 is **deep through week 4** of the design's build sequence (§8). The text side is end-to-end: 3 talks ingested, 212 chunks chunked, 212 dense + 212 sparse + 73,621 multi-vector token rows persisted, four text retrieval channels functionally smoke-tested against the populated corpus, RRF fusion lands with per-channel rank provenance.

Two pieces ship in the most recent slice:

- **`baffa55 fix(retrieve): RRF returns FusedResult with channel_ranks provenance`** — RRF now emits `FusedResult` (chunk_id + video_id + start_sec + end_sec + text + score + rank + `channel_ranks: dict[str, int]`). Trace logging, Verify-node confidence checks, and per-channel ablation downstream all depend on this provenance.

- **`c199a2f feat(retrieve): visual prefilter stage 1 — ColQwen pooled + cosine HNSW`** — `embed/colqwen.py` exposes `encode_image_pooled` + `encode_text_query`; `retrieve/visual.py::retrieve_frames` queries `frames.pooled_embedding` via pgvector cosine HNSW. **STAGE 1 ONLY** — per-patch embeddings + MaxSim stage 2 land in step 24-25 with the frame-sampling worker.

### Gates green at HEAD

```
make test                33 fast tests pass (was 22 last week; +11 RRF)
make lint                clean
make phase0-gate         OK (11/3 verified gold across distinct talks)
make validate-evals-strict  clean
```

Per-suite slow counts (verified against the live container during the slice; ~1-2 min total when warm):
```
test_db_migrations      12
test_talks_repo         19
test_pgmq_smoke         13
test_handlers_slow       5
test_chunks_repo         5
test_embeds_repo         7
test_readiness           6
test_cli_all_slow        3
test_channels_smoke     20  (4 channels × 5 test shapes)
test_bge_m3              6
test_handlers_fast       3  (fast, picked up by make test)
test_bge_m3_lazy         1  (fast)
test_rrf                14  (fast, +3 provenance)
test_visual_prefilter    4  (slow, DEFERRED — see below)
```

### One open verification gate — visual slow test

The visual prefilter test (`eval/tests/test_visual_prefilter.py`) was committed with a deferred end-to-end run because the ColQwen2.5 v0.2 model is ~14 GB and the session ran out of download budget after pulling only 244 MB. The wire is straightforward (same pgvector cosine HNSW pattern as `dense.py`, which IS fully exercised), but the next session must close this gate before relying on visual retrieval downstream.

**Run before any week-5 work:**

```bash
make db-up
make db-migrate
uv run pytest eval/tests/test_visual_prefilter.py -q -m slow   # ~10-15 min first time (model dl + cold load)
make db-down
```

If any of the 4 visual tests fail, fix before proceeding. The failure mode most likely worth checking: colpali-engine API path. The teammate probed `colpali_engine.models.ColQwen2_5` + `ColQwen2_5_Processor` on 0.3.16 and confirmed those exports exist; if they shift in a future release, `_model()` in `embed/colqwen.py` needs adjustment.

---

## What's shipped vs what's deferred

### Shipped (text + visual prefilter)

| Surface | State |
|---|---|
| `chunking/` | `fixed_window` strategy + registry; v0 only |
| `embed/bge_m3.py` | 3-channel encode (dense/sparse/multi); config-driven model id |
| `embed/colqwen.py` | `encode_image_pooled` + `encode_text_query` (stage 1); patches deferred |
| `db/repos/{talks,chunks,embeds,ingest_step_status}.py` | full CRUD; 1-based sparsevec; idempotent |
| `queues/{pgmq_client,workers}.py` | send / send_batch / process_one with full transactional contract |
| `ingest/{fetch,quality,pipeline,handlers,cli,cli_all}.py` | fetch + chunk + embed_text handlers; CORPUS-driven ingest-all |
| `retrieve/{bm25,dense,sparse,multivec,visual,rrf}.py` | 4 text channels + visual stage 1 + RRF fusion with channel_ranks |
| `scripts/bakeoff_prep.py` | per-chunk strict readiness with N/M counts |

### Deferred (per design §8 weeks 5+)

| Step | Surface |
|---|---|
| 22-23 | `ingest/frames.py` — ffmpeg frame sampler + frames-table writer (the **producer** for visual.retrieve_frames) |
| 24-25 | `embed/colqwen.py::encode_image_patches()` — per-patch embeddings + MaxSim stage 2 in `retrieve/visual.py` |
| 26-27 | `retrieve/rerank.py` — BGE-reranker-v2-m3 on the RRF top-K |
| 28-31 | `eval/runners/minimal_generation.py` + `generate/` LangGraph nodes (plan/retrieve/rerank/verify/generate/cite) + `generate/state.py` + `generate/trace.py` + `db/repos/traces.py` |
| 32-33 | `generate/graph.py` + `generate/api.py` (`make answer`) |
| 34 | **Week-5 EXIT GATE:** `make answer QUERY=...` returns a cited answer + trace_id |
| 35 | Bakeoff-prep one-off harness via `minimal_generation.generate_for_bakeoff` on all 11 dev_gold |
| 35a | `scripts/run_embeddings_bakeoff.py` forward-contract scaffold |

Out-of-phase deferrals (not until phase 2+): worker daemon CLI, ASR worker (WhisperX), `retrieve/api.py` public surface, generation, agent loop, API, web.

---

## First commands the new session MUST run

In order. Stop if any is red.

```bash
git log --oneline -5
# Top should be: c199a2f feat(retrieve): visual prefilter stage 1 ...

make validate-evals-strict
make phase0-gate
make test
make lint
# All four must pass.

# Then close the open visual gate:
make db-up
make db-migrate
uv run pytest eval/tests/test_visual_prefilter.py -q -m slow
# 4 passed. Heavy first run (~10-15 min) due to ColQwen download.
make db-down
```

If the visual suite fails: triage in `embed/colqwen.py`. The pgvector cosine query is a copy of `retrieve/dense.py` (proven to work). The likely failure is either (a) colpali-engine API drift since 0.3.16 — adjust the import path, OR (b) `POOLED_DIM` mismatch — the runtime assertion in `encode_image_pooled` will raise loudly with the actual dim, then update both `POOLED_DIM` and `db/migrations/0003_frames.sql` accordingly (and add a new migration if you've already deployed elsewhere).

---

## Reading order — load before writing code

1. `docs/phase-1-design.md` §4 (retrieval API), §5 (LangGraph loop), §8 weeks 5 (frames + reranker + LangGraph + bakeoff harness)
2. `docs/decisions/004-model-selection.md` v3.1 — selection rule + no-silent-defaults discipline
3. `.claude/CLAUDE.md` hard rules
4. `eval/config/model_candidates.yaml` — bakeoff candidate set (generator/judge/reranker pinned here)
5. `retrieve/multivec.py` — the closest existing parallel for the stage-2 MaxSim work landing in step 25
6. `eval/tests/test_handlers_slow.py` + `eval/tests/test_pgmq_smoke.py` — the template for new slow tests that exercise the PGMQ worker contract

Skim only (stable for the next few sessions):
- `db/migrations/*.sql` — schema is frozen; no migration changes expected through week 5
- `eval/corpora/ai_engineering_v0/*.jsonl` — corpus is locked

---

## Mission stack — what week 5 ships

Three slices, sized for ~one session each. Take them in order; each verifies before the next begins.

### Slice 1 — Frames worker (design §8 steps 22-23)

- `ingest/frames.py` — ffmpeg subprocess sampling frames every N seconds; writes frames rows + computes sha256
- ingest_frames PGMQ handler in `ingest/handlers.py::frames_handler` mirroring `embed_text_handler`'s shape
- Slow test: 60s synthetic video (use ffmpeg to generate test asset) → handler runs → frames table populated; sha256 stable across re-runs
- `make ingest-all` drains the ingest_frames queue after ingest_chunk

### Slice 2 — Visual stage 2 + reranker (design §8 steps 24-27)

- `embed/colqwen.py::encode_image_patches()` — per-patch embeddings (deferred from this session)
- `retrieve/visual.py` extended with stage 2 MaxSim over the prefiltered frame set
- `retrieve/rerank.py` — BGE-reranker-v2-m3 on the RRF top-K (CPU local per design)
- Functional smoke tests for both (no quality assertions; that's the phase-2 bakeoff's job)

### Slice 3 — LangGraph generation loop (design §8 steps 28-34)

This is the big one — the week-5 EXIT GATE. Three sub-pieces:
- `eval/runners/minimal_generation.py` + `eval/runners/providers.py` + `generate/parser.py` (shared with the LangGraph generate node)
- `generate/{state,nodes/*,trace}.py` + `db/repos/traces.py`
- `generate/{graph,api}.py` — `make answer QUERY=...` returns a cited answer + trace_id; SELECT * FROM trace_spans shows the full Plan→Retrieve→Rerank→Verify→Generate→Cite path

Verification artifact at the end of week 5:
```bash
make answer QUERY="how does Tengyu Ma compare long context, fine-tuning, and RAG" \
            GENERATOR=qwen3-235b-a22b-instruct \
            JUDGE=deepseek-v3.2
# returns answer + trace_id; trace_spans table shows 6 spans across 1 iteration
```

---

## Hard rules — non-negotiable (from CLAUDE.md + ADRs)

1. **`make phase0-gate` stays green at every commit boundary.** Re-run before each commit.
2. **Postgres only.** No Redis/Celery/RabbitMQ. PGMQ for queues, pgvector for embeddings, no second store (ADR 002).
3. **LangGraph only.** No LangChain core, no LlamaIndex (ADR 001).
4. **No silent model defaults.** `make answer` must require explicit `GENERATOR=` and `JUDGE=` until bakeoff #1 winners are locked.
5. **Cross-family judge.** Curation family ≠ judge family ≠ generator family per ADR 004.
6. **`test_gold.jsonl` is locked.** Its SHA256 is in the README; modifying it requires a methodology revision note.
7. **One model library per concern.** BGE-M3 + ColQwen2.5 + WhisperX (when ASR lands) — no new heavy model packages without an ADR amendment.
8. **No worker daemon CLI yet.** `make ingest-all` is the one-shot driver; the long-running `python -m queues.workers --queues ...` daemon waits until step 36+.
9. **Conventional commits.** `feat:` `fix:` `refactor:` `docs:` `eval:` `chore:`. No AI attribution lines.
10. **Use TeamCreate for substantive slices.** Two parallel teammates per slice (disjoint files), orchestrator verifies + commits.

---

## Known gotchas (collected across week-3 + week-4)

These are the empirical contracts that tripped people up. Don't relearn them.

- **pgvector sparsevec is 1-based on wire, 0-based on the SparseVector Python API.** `pgvector.SparseVector({0: 0.5}, 250002)` serialises as `{1:0.5}/250002`. Don't shift in our wrapper.
- **`pgmq.send_batch` returns SETOF bigint** (a row per id), NOT a single `bigint[]`. Use `fetchall()` + unwrap `r[0]`.
- **psycopg adapts `list[Jsonb]` as a TEXT-array literal of JSON strings**, not `jsonb[]`. Pre-serialise with `json.dumps` to `list[str]` and cast `::jsonb[]` server-side.
- **BGE-M3 `colbert_vecs` strips the trailing EOS** but keeps the leading CLS, so `multi[i].shape[0] == len(input_ids) - 1`. Downstream MaxSim must account for this.
- **`websearch_to_tsquery` uses AND-of-all-terms semantics.** Natural-language questions match nothing under AND. `retrieve/bm25.py` falls back to OR-expanded `to_tsquery` via a Python stopword-filtered tokenizer.
- **`make ingest-all` returns nonzero on fetch failure OR incomplete per-chunk readiness.** `scripts/bakeoff_prep.py` shows N/M counts so partial coverage reads as ✗.
- **`make db-reset` ignores shell `POSTGRES_DSN`** (pinned to local compose DSN inside the recipe). `make db-migrate` still respects it.
- **`make db-build` first run on arm64** falls back to `quay.io/tembo/pg17-pgmq` because `postgresql-16-pgmq` isn't in PGDG apt. Documented in `Dockerfile.postgres`.

---

## Cost reminder (for `make answer` once it lands)

Per design §9 risk 8: each `make answer` query is ~$0.001–0.002 of DeepInfra cost (Plan + Verify×N + Generate). The ADR 004 budget of $18-33 for 12 weeks counts bakeoff traffic only. If `make answer` enters routine use (>50 queries/day sustained), rate-limit or cache-by-query-hash before the demo runs the bill up.

---

## Where to push back if asked to deviate

Don't silently bend the design for convenience. If a slice plan needs to violate any of:
- Postgres-only invariant
- LangGraph-only invariant
- No-silent-defaults rule
- The 1-based-on-wire sparsevec contract
- The cross-family judge discipline

Surface explicitly, propose an ADR amendment, and wait for the user's "approve amendment" signal. The 4-rev'd design + 3 plan-review passes are the contract; deviations move through the same process.
