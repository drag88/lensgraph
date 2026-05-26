# Handoff — Phase 1, Week 5 (frames worker + reranker + LangGraph)

**Written:** 2026-05-25, end of week-4 retrieval session (last updated 2026-05-26 after slice-1 fix slice)
**Read by:** the next session, before any implementation
**Authoritative design:** `docs/phase-1-design.md` rev 4 (commit `e950583`)
**Implementation head:** `57c0c9e` `feat(retrieve): visual stage 2 + BGE reranker — slice 2B`. A trailing `docs(handoff)` closeout commit may sit on top of that — don't pin a fresh session to an exact HEAD, just confirm `57c0c9e` is in `git log` and gates are green.
**Slice 1 commits:** `3c47a4c feat(ingest): frame-sampling worker`, `45575df fix(ingest,readiness): slice-1 review hardening`, `645c37e chore: ruff format sweep`.
**Slice 2 commits:** `3fc5ab4 feat(embed,ingest): ColQwen patches + embed_frames pipeline — slice 2A`, `57c0c9e feat(retrieve): visual stage 2 + BGE reranker — slice 2B`.

---

## Where we are

Phase 1 is **deep through week 4** of the design's build sequence (§8). The text side is end-to-end: 3 talks ingested, 212 chunks chunked, 212 dense + 212 sparse + 73,621 multi-vector token rows persisted, four text retrieval channels functionally smoke-tested against the populated corpus, RRF fusion lands with per-channel rank provenance.

Two pieces ship in the most recent slice:

- **`baffa55 fix(retrieve): RRF returns FusedResult with channel_ranks provenance`** — RRF now emits `FusedResult` (chunk_id + video_id + start_sec + end_sec + text + score + rank + `channel_ranks: dict[str, int]`). Trace logging, Verify-node confidence checks, and per-channel ablation downstream all depend on this provenance.

- **`c199a2f feat(retrieve): visual prefilter stage 1 — ColQwen pooled + cosine HNSW`** — `embed/colqwen.py` exposes `encode_image_pooled` + `encode_text_query`; `retrieve/visual.py::retrieve_frames` queries `frames.pooled_embedding` via pgvector cosine HNSW. **STAGE 1 ONLY** — per-patch embeddings + MaxSim stage 2 land in step 24-25 with the frame-sampling worker.

### Gates green at HEAD

```
make test                58 fast tests pass (46 at slice-1 close + 12 slice-2 fast)
make lint                clean
make phase0-gate         OK (11/3 verified gold across distinct talks)
make validate-evals-strict  clean
```

Per-suite counts (verified against the live container; slice-2 slow ~55s warm):
```
test_db_migrations         12  (slow)
test_talks_repo            19  (slow)
test_pgmq_smoke            13  (slow)
test_chunks_repo            5  (slow)
test_embeds_repo            7  (slow)
test_handlers_slow         11  (slow — 5 chunk/embed_text + 3 frames_handler + 3 embed_frames_handler)
test_readiness             11  (slow — 6 original + 5 frame_sample exposure)
test_cli_all_slow           7  (slow — 3 fetch/readiness + 4 required-videos guard branches)
test_channels_smoke        20  (slow — 4 channels × 5 test shapes)
test_bge_m3                 6  (slow)
test_visual_prefilter       7  (slow — 4 stage-1 + 3 stage-2 channel retrieve)
test_frames                23  (slow — 13 sampler/repo + 2 absolute-path + 8 patches/pooled lifecycle)
test_colqwen_patches        3  (slow — 5 of the 8 in this file are fast)
test_rerank_functional      6  (slow — top_k monotonic, preserve-set, empty, smaller-than-k, rrf_score / channel_ranks carry-through)
test_rerank_smoke           1  (slow — catastrophic-regression against tengyu-rag-library-analogy dev_gold example)
test_handlers_fast         11  (fast — 3 chunk + 5 frames_handler + 3 embed_frames_handler)
test_bge_m3_lazy            1  (fast)
test_rrf                   14  (fast)
test_colqwen_lazy           5  (fast)
test_colqwen_patches (fast)  5  (fast — patches/text-patches/pool resolver + lazy short-circuits)
test_rerank_lazy            4  (fast — resolver + empty-candidates lazy short-circuit)
```

### Slice 0 / visual stage-1 verification gate — CLOSED

**Status: CLOSED 2026-05-25.** All 4 slow tests in `eval/tests/test_visual_prefilter.py` passed in 38.08s against the live container. The ColQwen2.5 v0.2 weights were already cached locally from the prior session's partial pull, so the cold-start ~14 GB download did not need to repeat.

The pre-flight `2e28dce fix(embed)` commit pulled the model id from `eval/config/model_candidates.yaml::candidates.visual_retrieval[colqwen2.5]` (provider validated as `local`) — no string is hardcoded in `embed/colqwen.py` anymore.

If a future colpali-engine release shifts the export path away from `colpali_engine.models.ColQwen2_5` + `ColQwen2_5_Processor`, `_model()` in `embed/colqwen.py` needs adjustment. Probed on 0.3.16 and confirmed those exports exist.

### Slice 1 / frame-sampling worker — CLOSED

**Status: CLOSED 2026-05-26 (review fixes folded in).** `3c47a4c feat(ingest)` shipped the frame-sampling worker end-to-end. `45575df fix(ingest,readiness)` hardened the path resolver (handles absolute paths now), the cli_all required-videos guard (enumerates required .mp4s, three-branch decision), and exposed frame_sample status in readiness/bakeoff-prep without coupling text readiness to visual.

### Slice 2 / visual stage 2 + reranker — CLOSED

**Status: CLOSED 2026-05-26.** Two commits:

- `3fc5ab4 feat(embed,ingest)` — ColQwen patches + embed_frames pipeline.
  `embed/colqwen.py` exposes `encode_image_patches`, `encode_text_query_patches`, `pool_patches`; pooled encoders refactored to delegate (one encoder path). `db/repos/frames.py` adds `get`, `update_pooled`, `replace_patches`. `ingest/handlers.py::embed_frames_handler` is the terminal step: lookup → PIL load → encode → write `frame_patches` + `frames.pooled_embedding` from the SAME patch matrix. `ingest/cli_all.py` drains `ingest_embed_frames` after `ingest_frames` (same required-videos guard).

- `57c0c9e feat(retrieve)` — visual stage 2 + reranker.
  `retrieve/visual.py::retrieve()` (channel-level) does pooled HNSW prefilter → frame→chunk JOIN → MaxSim over concatenated per-chunk patches. `retrieve/rerank.py` is a NEW module wrapping `sentence_transformers.CrossEncoder` on `BAAI/bge-reranker-v2-m3` (resolved from `model_candidates.yaml::candidates.reranker.options[0]` — no hardcoded model strings). Raw logits sigmoid'd to 0-1 confidence.

  **Reranker backend deviation worth flagging:** the design says "BGE-reranker-v2-m3 via FlagEmbedding". FlagEmbedding 1.4.0 calls `tokenizer.prepare_for_model`, which was removed in transformers 5.x (we're pinned at 5.9.0). Every `FlagReranker.compute_score` call crashes with `AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model`. Switched to `sentence_transformers.CrossEncoder` — same model checkpoint, modern tokenizer API. If FlagEmbedding catches up to transformers 5.x, the switch back is one import + one constructor call. Not worth an ADR amendment because the model + scoring semantics are unchanged; only the loader differs.

LangGraph loop + `make answer` exit gate (slice 3) remains deferred.

---

## What's shipped vs what's deferred

### Shipped (text + visual prefilter)

| Surface | State |
|---|---|
| `chunking/` | `fixed_window` strategy + registry; v0 only |
| `embed/bge_m3.py` | 3-channel encode (dense/sparse/multi); config-driven model id |
| `embed/colqwen.py` | full encode surface — pooled + patches + text-pooled + text-patches (one encoder path); config-driven model id |
| `queues/{pgmq_client,workers}.py` | send / send_batch / process_one with full transactional contract |
| `ingest/{fetch,quality,pipeline,handlers,cli,cli_all,frames,readiness}.py` | fetch + chunk + embed_text + frame_sample + embed_frames handlers; CORPUS-driven ingest-all with required-videos guard |
| `db/repos/{talks,chunks,embeds,frames,ingest_step_status}.py` | full CRUD; 1-based sparsevec; idempotent (frames upsert preserves pooled_embedding; replace_patches DELETE+INSERT) |
| `retrieve/{bm25,dense,sparse,multivec,visual,rrf,rerank}.py` | 5 channels (4 text + visual stage 1+2 chunk-level) + RRF + BGE-reranker (CrossEncoder backend) |
| `scripts/bakeoff_prep.py` | per-chunk strict text readiness with N/M counts + visual `frames` column (status + count) |

### Deferred (per design §8 weeks 5+)

| Step | Surface |
|---|---|
| 22-23 | ✅ **CLOSED slice 1** — `ingest/frames.py` + `db/repos/frames.py` + `frames_handler` + cli_all drain + required-videos guard |
| 24-25 | ✅ **CLOSED slice 2A** — `embed/colqwen.py` full patches + text-patches + pool helper; `db/repos/frames.py` `update_pooled` + `replace_patches`; `embed_frames_handler` + drain wiring |
| 26-27 | ✅ **CLOSED slice 2B** — `retrieve/visual.py` channel-level `retrieve()` (pooled prefilter → frame→chunk JOIN → MaxSim); `retrieve/rerank.py` (sentence-transformers CrossEncoder backend) |
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
# Confirm 57c0c9e appears in git log; trailing docs/test closeout commits may sit on top.

make validate-evals-strict
make phase0-gate
make test
make lint
# All four must pass (58 fast tests).
```

Slices 0, 1, and 2 are all CLOSED. Slice 3 (LangGraph generation loop + `make answer` exit gate) is next. No prerequisite slow-suite re-run needed unless you've changed code under `embed/`, `db/migrations/`, `ingest/frames.py`, `retrieve/visual.py`, or `retrieve/rerank.py`.

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

### Slice 1 — Frames worker (design §8 steps 22-23) — ✅ CLOSED 2026-05-26

Shipped across `3c47a4c feat(ingest)` + `45575df fix(ingest,readiness)` + `645c37e chore`. Frames substrate is end-to-end: ffmpeg sampler, idempotent repo, fan-out handler, cli_all drain with strict required-videos guard. Visual readiness is now visible in bakeoff-prep without coupling to text readiness.

### Slice 2 — Visual stage 2 + reranker (design §8 steps 24-27) — ✅ CLOSED 2026-05-26

Shipped across `3fc5ab4 feat(embed,ingest)` + `57c0c9e feat(retrieve)`. ColQwen exposes full patches (image + text); embed_frames worker populates `frame_patches` + `frames.pooled_embedding` from the same matrix; `retrieve/visual.py::retrieve()` is the channel-level entry doing pooled prefilter → frame→chunk JOIN → MaxSim; `retrieve/rerank.py` wraps `sentence_transformers.CrossEncoder` (FlagEmbedding 1.4.0 broken under transformers 5.x — documented in commit body). Model IDs config-driven from `model_candidates.yaml`.

### Slice 3 — LangGraph generation loop (design §8 steps 28-34) — NEXT

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
