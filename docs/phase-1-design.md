# Phase 1 Design — LensGraph Ingest, Retrieve, LangGraph Loop

**Status:** rev 3, pending `/cdf:plan-review` re-run
**Date:** 2026-05-24
**Scope:** weeks 3–5 of `docs/roadmap.md`.
**Out of scope:** phase-4 surface (FastAPI + Next.js), auth, anything in §10.

## Changelog

- **rev 3 (2026-05-24).** Tightens ADR 004 v3.1's "no silent model defaults" rule across **every** model-using node in the LangGraph loop:
  1. `load_bakeoff_winner_or_default()` → `load_bakeoff_winner_or_raise()`. No "first YAML candidate" fallback anywhere. Pre-bakeoff, `answer()` and `make answer` require an explicit `GENERATOR=...`; post-bakeoff, they may auto-load the locked winner from `eval_runs`.
  2. Same rule applied to Verify. `judge_candidate` added to `AgentState` and `answer()`. Pre-judge-bakeoff, `make answer` requires `JUDGE=...`; post-bakeoff, auto-load. When both generator and judge are explicitly passed, cross-family is **enforced** (raises `CrossFamilyViolationError`).
  3. Plan node gains a **deterministic non-LLM mode**, which is the default before the cheap-extraction component is bakeoff-locked. Explicit `PLANNER=...` overrides to use an LLM cheap-extraction candidate. Either way, no silent LLM default.
  4. `BakeoffNotYetRunError` documented as the public failure mode; `make answer` surfaces it with a copy-pasteable fix message ("re-run with GENERATOR=qwen3-235b-a22b-instruct JUDGE=deepseek-v3.2 …").
  5. Other rev-3 amendments folded in from the second `/cdf:plan-review` pass: fan-out step completion semantics (PGMQ DAG), shared `parse_generation_output()` between bakeoff harness and LangGraph generate node, `Dockerfile.postgres` build-verification gate in week 3, visual-channel sampling-density note, catastrophic-regression rerank smoke test, trace spans store `chunk_ids` not chunk text, verify confidence threshold from env var, `AsrJob.timeout_sec` default raised to 1800, caption quality probe heuristic specified, CI runs `make test`, AgentState serialization helpers, honest 30–35h budget acknowledged.
- **rev 2 (2026-05-24).** LangGraph loop (Plan → Retrieve → Rerank → Verify → Generate → Cite with Verify→Retrieve back-edge) moved from old phase 3 into phase 1. `eval/runners/minimal_generation.py` repositioned as internal bakeoff harness only. Postgres-native tracing added. Five plan-review findings closed (scalar status, pgmq image, citation leakage, visual-channel dependence, rerank-as-test). Earlier round of `/cdf:plan-review` amendments (sparsevec dim, Docker image, FOR UPDATE, ffmpeg dependency, provider retry, latency definition, overlap comment, models-warm/replay-dlq/logs targets, slow-test gating, claim_index correction, batched chunk_token_embeds SELECT, HNSW note) all applied.
- **rev 1 (2026-05-24).** Initial design. Deferred LangGraph to phase 3. Superseded same day.

## Binding constraints

- ADR 004 v3.1. No model is "the default." Candidate sets and selection rule live in `eval/config/model_candidates.yaml`; that file is source of truth. DeepInfra primary, OpenRouter optional failover, Groq excluded.
- ADR 002. Postgres only. PGMQ for queues, pgvector (incl. sparsevec ≥ 0.7) for embeddings, in-database `traces` table for observability. No Redis, Celery, RabbitMQ, second vector store, external tracing backend.
- ADR 001. LangGraph is the agent framework — **in v1**, not deferred. Phase 1 implements the typed-state graph with Plan/Retrieve/Rerank/Verify/Generate/Cite nodes.
- BGE-M3 all three channels (dense + sparse + multi-vector). Voyage / Gemini Embedding 2 are fallbacks only, gated on BGE-M3 failing the embeddings minimum at the phase-2 week-5 bakeoff.
- ColQwen2.5 via `colpali-engine`. Gemini Embedding 2 single-vector is a writeup-only baseline.
- BGE-reranker-v2-m3 local CPU. Swap only if p95 > 80ms.
- WhisperX large-v3 local MPS.
- `transcript_sha256` in `talks.yaml` is a contract. Ingest never overwrites a transcript without updating the SHA atomically (temp → verify → swap → SHA update; the existing validator drift check catches misses).

---

## 1. Module map

Plain Python ≥ 3.11. Functions over classes; classes only where state genuinely warrants (model handles on GPU, DB pools). Files use `from __future__ import annotations` and `ruff` line-length 100. Type-hint everything; no `Any` escape hatches except at provider HTTP boundaries.

```
ingest/
  __init__.py
  fetch.py              # video + caption acquisition
  asr.py                # WhisperX fallback (timeout-bounded)
  frames.py             # frame sampling (ffmpeg)
  pipeline.py           # PGMQ job orchestration + per-step status reconciler
  cli.py                # `python -m ingest <video_id>` entry point
  quality.py            # caption quality probe (decides ASR yes/no)

chunking/
  __init__.py           # strategy registry — `get_chunker(name)`
  types.py              # Chunk dataclass
  fixed_window.py       # v0 strategy

retrieve/
  __init__.py
  types.py              # RetrievedChunk dataclass
  bm25.py               # Postgres FTS channel
  dense.py              # BGE-M3 dense / pgvector HNSW
  sparse.py             # BGE-M3 sparse / pgvector sparsevec HNSW
  multivec.py           # BGE-M3 multi-vector / MaxSim (batched)
  visual.py             # ColQwen2.5 — true 5th channel via pooled-prefilter then MaxSim
  rrf.py                # Reciprocal Rank Fusion
  rerank.py             # BGE-reranker-v2-m3
  api.py                # public `retrieve()` and `rerank()` entry points

embed/
  __init__.py
  bge_m3.py             # singleton wrapper, lazy MPS load, batched encode (dense+sparse+token)
  colqwen.py            # singleton wrapper, MPS first, Modal fallback config
                        # exposes encode_image_patches() AND encode_image_pooled() AND encode_text_query()

generate/                       # ← new in rev 2 (LangGraph loop, the product path)
  __init__.py
  state.py              # AgentState (TypedDict + Pydantic helpers for serialization)
  graph.py              # build_graph() — assembles nodes + conditional edges
  trace.py              # span emission to traces / trace_spans tables
  api.py                # public `answer(query, corpus_id) -> AnswerResult`
  nodes/
    __init__.py
    plan.py             # decompose query; classify single_clip vs synthesis
    retrieve.py         # wraps retrieve.api.retrieve
    rerank.py           # wraps retrieve.api.rerank
    verify.py           # confidence check → loop back to retrieve (max 2 iters)
    generate.py         # constrained generation over retrieved chunks
    cite.py             # attach [video_id, start_sec, end_sec] to each model-emitted claim

db/
  __init__.py
  conn.py               # psycopg pool, env-driven DSN
  migrate.py            # raw-SQL migration runner
  migrations/
    0001_init.sql
    0002_chunks_embeds.sql
    0003_frames.sql
    0004_eval_runs.sql
    0005_traces.sql     # ← new in rev 2
  repos/
    talks.py
    chunks.py
    embeds.py
    frames.py
    ingest_step_status.py     # ← per-step/per-entity (was scalar in rev 1)
    eval_runs.py
    traces.py                  # ← new in rev 2

queues/
  __init__.py
  pgmq_client.py        # thin psycopg + pgmq SQL wrapper
  jobs.py               # job-payload dataclasses + JSON (de)serialization
  workers.py            # `python -m queues.workers --queues fetch,asr,...`

eval/runners/
  minimal_generation.py   # internal bakeoff harness only (NOT product path)
  providers.py            # DeepInfra primary + OpenRouter failover HTTP client
  judges/                 # populated phase 2
```

Key public signatures:

```python
# ingest/pipeline.py
def enqueue_video(video_id: str, *, force: bool = False) -> None: ...
def next_step(video_id: str) -> StepDescriptor | None:   # returns the next pending step

# chunking/fixed_window.py
@dataclass(frozen=True)
class Chunk:
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    chunking_strategy: str
    frame_secs: list[float]
    token_count: int

def chunk(transcript_vtt: str, frames: list[Frame], *,
          window_sec: float = 30.0, overlap_sec: float = 5.0,
          max_tokens: int = 512) -> list[Chunk]: ...

# retrieve/api.py
@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    channel_ranks: dict[str, int]   # channel → 1-indexed rank within its top-k (missing channels absent)
    fused_score: float
    rerank_score: float | None = None

def retrieve(query: str, *, top_k_per_channel: int = 30) -> list[RetrievedChunk]: ...
def rerank(query: str, candidates: list[RetrievedChunk], *, top_k: int = 8) -> list[RetrievedChunk]: ...

# generate/api.py — the product path
@dataclass(frozen=True)
class AnswerResult:
    answer: str | None             # None when abstain=True
    citations: list[Citation]
    abstain: bool
    trace_id: str                  # FK into traces table
    iterations: int                # how many Verify→Retrieve cycles
    latency_ms: int

def answer(query: str, *, corpus_id: str = "ai_engineering_v0",
           generator_candidate: GeneratorCandidate | None = None) -> AnswerResult: ...

# eval/runners/minimal_generation.py — internal harness, not product
def generate_for_bakeoff(query: str, chunks: list[RetrievedChunk],
                         candidate: GeneratorCandidate) -> GenerationResult: ...
```

---

## 2. Postgres schema

**Migration approach: raw SQL via psycopg.** Justification: pgvector and pgmq are extensions with non-standard types (`vector`, `sparsevec`, `pgmq` schema); alembic autogenerate handles neither cleanly. Solo project, single database, ≤ ~10 migrations across phase 1. Raw SQL keeps the surface tight and the migration log readable. Applied migrations tracked in `schema_migrations(version text PK, applied_at timestamptz)`.

Required extensions (in `0001_init.sql`):

```sql
CREATE EXTENSION IF NOT EXISTS vector;        -- ≥ 0.7 for sparsevec
CREATE EXTENSION IF NOT EXISTS pgmq;          -- requires custom Postgres image (see §7)
```

### `0001_init.sql` — talks + per-step status

```sql
CREATE TABLE schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE talks (
    video_id            text PRIMARY KEY,
    source_video_id     text,
    source_start_sec    real,
    source_end_sec      real,
    title               text NOT NULL,
    speaker             text NOT NULL,
    url                 text NOT NULL,
    duration_sec        integer NOT NULL CHECK (duration_sec > 0),
    format_tags         text[] NOT NULL,
    license             text NOT NULL,
    captions_source     text NOT NULL,
    transcript_path     text NOT NULL,
    transcript_sha256   text NOT NULL CHECK (transcript_sha256 ~ '^[a-f0-9]{64}$'),
    accessed_at         timestamptz NOT NULL,
    notes               text,
    ingested_at         timestamptz
);

-- Per-step / per-entity status. Replaces rev 1's scalar last_completed_step,
-- which could not represent parallel branches (text vs frame embed) or
-- per-entity fan-out (one row per chunk for embed_text, per frame for
-- embed_frames). entity_id=0 means "whole-video step" (fetch, asr,
-- frame_sample, chunk).
CREATE TABLE ingest_step_status (
    video_id       text NOT NULL REFERENCES talks(video_id) ON DELETE CASCADE,
    step           text NOT NULL,             -- 'fetch','asr','frame_sample','chunk','embed_text','embed_frames'
    entity_id      bigint NOT NULL DEFAULT 0, -- chunk_id / frame_id; 0 for whole-video steps
    status         text NOT NULL,             -- 'pending','in_progress','completed','failed','skipped'
    attempts       integer NOT NULL DEFAULT 0,
    started_at     timestamptz,
    completed_at   timestamptz,
    error_payload  jsonb,
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (video_id, step, entity_id)
);
CREATE INDEX ingest_step_status_video_idx ON ingest_step_status (video_id, status);
CREATE INDEX ingest_step_status_pending_idx ON ingest_step_status (step, status)
    WHERE status IN ('pending', 'in_progress', 'failed');

-- PGMQ queues created at startup, not in migration (PGMQ API is functional):
--   SELECT pgmq.create('ingest_fetch');
--   SELECT pgmq.create('ingest_asr');
--   SELECT pgmq.create('ingest_frames');
--   SELECT pgmq.create('ingest_chunk');
--   SELECT pgmq.create('ingest_embed_text');
--   SELECT pgmq.create('ingest_embed_frames');
-- Plus matching DLQs: pgmq.create('<name>_dlq') for each.
```

### `0002_chunks_embeds.sql` — text retrieval substrate

```sql
CREATE TABLE chunks (
    chunk_id            bigserial PRIMARY KEY,
    video_id            text NOT NULL REFERENCES talks(video_id) ON DELETE CASCADE,
    chunking_strategy   text NOT NULL,    -- 'fixed_window' in v0
    start_sec           real NOT NULL,
    end_sec             real NOT NULL CHECK (end_sec > start_sec),
    text                text NOT NULL,
    token_count         integer NOT NULL CHECK (token_count <= 512),
    frame_secs          real[] NOT NULL DEFAULT '{}',
    tsv                 tsvector GENERATED ALWAYS AS
                          (to_tsvector('english', text)) STORED,
    created_at          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (video_id, chunking_strategy, start_sec, end_sec)
);
CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX chunks_video_strategy_idx ON chunks (video_id, chunking_strategy);

-- HNSW params (m=16, ef_construction=64) are pgvector defaults tuned for v1
-- corpus scale (~12.5K chunks). At v0 (~50 chunks) IVFFlat would be cheaper
-- but the parameter choice is dominated by v1 — don't re-tune for v0.
CREATE TABLE dense_embeds (
    chunk_id   bigint PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding  vector(1024) NOT NULL    -- BGE-M3 dense
);
CREATE INDEX dense_embeds_hnsw ON dense_embeds
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- 250002 = XLM-RoBERTa-large vocab size (BGE-M3's tokenizer). Verified at
-- ingest startup against model.tokenizer.vocab_size; a mismatch aborts.
CREATE TABLE sparse_embeds (
    chunk_id   bigint PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding  sparsevec(250002) NOT NULL
);
CREATE INDEX sparse_embeds_hnsw ON sparse_embeds
    USING hnsw (embedding sparsevec_ip_ops) WITH (m = 16, ef_construction = 64);

-- Per-token vectors live in a row-per-token table. HNSW on individual tokens
-- is not useful; MaxSim is computed in app code over the dense+sparse top-100
-- candidate union — never the full corpus. Rows scale with chunk count ×
-- tokens/chunk; chunk overlap multiplies storage. At default `overlap_sec=5.0`
-- on `window_sec=30.0` windows, ~17% overhead vs no-overlap. Tune via chunker
-- params, not schema.
CREATE TABLE chunk_token_embeds (
    chunk_id   bigint NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    position   integer NOT NULL,
    embedding  vector(1024) NOT NULL,
    PRIMARY KEY (chunk_id, position)
);
CREATE INDEX chunk_token_embeds_chunk_idx ON chunk_token_embeds (chunk_id);
```

### `0003_frames.sql` — visual retrieval substrate (now a true 5th channel)

```sql
CREATE TABLE frames (
    frame_id          bigserial PRIMARY KEY,
    video_id          text NOT NULL REFERENCES talks(video_id) ON DELETE CASCADE,
    frame_sec         real NOT NULL,
    image_path        text NOT NULL,        -- frames stored locally; gitignored
    sha256            text NOT NULL,
    pooled_embedding  vector(128),          -- ColQwen pooled per-frame embedding
                                            -- used by visual channel's coarse prefilter
                                            -- NULL until embed_frames step completes
    UNIQUE (video_id, frame_sec)
);
CREATE INDEX frames_video_idx ON frames (video_id, frame_sec);
CREATE INDEX frames_pooled_hnsw ON frames
    USING hnsw (pooled_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE pooled_embedding IS NOT NULL;

-- ColQwen2.5 patch embeddings (~128-dim). Exact dim verified at startup
-- against model output; mismatch aborts the migration apply.
CREATE TABLE frame_patches (
    frame_id     bigint NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
    patch_index  integer NOT NULL,
    embedding    vector(128) NOT NULL,
    PRIMARY KEY (frame_id, patch_index)
);
CREATE INDEX frame_patches_frame_idx ON frame_patches (frame_id);
```

### `0004_eval_runs.sql` — populated by phase 2

```sql
CREATE TABLE eval_runs (
    run_id              text PRIMARY KEY,           -- '<date>_<slug>'
    run_date            date NOT NULL,
    code_path           text NOT NULL,              -- 'minimal_generation' | 'langgraph'
    chunking_strategy   text NOT NULL,
    embedding_model_id  text NOT NULL,
    generator_model_id  text,
    judge_model_id      text,
    judge_prompt_hash   text,
    candidate_set_yaml  jsonb NOT NULL,             -- snapshot of model_candidates.yaml
    summary             jsonb,                      -- metrics, slices
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE eval_results (
    eval_run_id   text NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    example_id    text NOT NULL,
    system_output jsonb NOT NULL,
    metrics       jsonb NOT NULL,
    PRIMARY KEY (eval_run_id, example_id)
);
```

### `0005_traces.sql` — LangGraph observability (Postgres-native, ADR 002 honored)

```sql
CREATE TABLE traces (
    trace_id         text PRIMARY KEY,            -- UUIDv7 from app
    query            text NOT NULL,
    corpus_id        text NOT NULL,
    generator_model_id text,
    started_at       timestamptz NOT NULL DEFAULT now(),
    ended_at         timestamptz,
    status           text NOT NULL,               -- 'running','completed','failed','abstained'
    final_answer     text,
    final_citations  jsonb,
    iterations       integer NOT NULL DEFAULT 0,
    latency_ms       integer
);
CREATE INDEX traces_started_idx ON traces (started_at DESC);

CREATE TABLE trace_spans (
    span_id        text PRIMARY KEY,              -- UUIDv7
    trace_id       text NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
    parent_span_id text,                          -- self-ref for nested spans
    node_name      text NOT NULL,                 -- 'plan','retrieve','rerank','verify','generate','cite'
    iteration      integer NOT NULL DEFAULT 0,    -- which Verify→Retrieve cycle
    started_at     timestamptz NOT NULL,
    ended_at       timestamptz,
    input          jsonb,
    output         jsonb,
    metadata       jsonb,                         -- model_id, token_count, latency_ms, error
    status         text NOT NULL                  -- 'completed','failed'
);
CREATE INDEX trace_spans_trace_idx ON trace_spans (trace_id, started_at);
CREATE INDEX trace_spans_node_idx ON trace_spans (node_name, started_at);
```

Optional external Langfuse sink (re-emit spans via API) is a v2 add. The schema above is Langfuse-shaped so the migration is mechanical when we want it.

---

## 3. PGMQ job DAG

Per video, jobs flow:

```
fetch (always)
  └─→ asr (only if quality.probe(captions_vtt) fails)
  └─→ frame_sample (only if format_tags ∩ {slides_heavy, code_heavy, whiteboard, live_demo} ≠ ∅)
  └─→ chunk
        └─→ embed_text   (one job per chunk_id)
        └─→ embed_frames (one job per frame_id; only if frame_sample ran)
```

**Idempotency under per-step status.** Every job's first action, inside a transaction:

```sql
SELECT status FROM ingest_step_status
 WHERE video_id = $1 AND step = $2 AND entity_id = $3
   FOR UPDATE;
```

If status is `completed` or `skipped`, the job acks and exits. Otherwise the job sets `status = 'in_progress'` (incrementing `attempts`), does the work, then sets `status = 'completed'` in the same transaction that writes the artifact and deletes the PGMQ message. `FOR UPDATE` makes this safe even if concurrency lifts to multiple workers per queue later (v0 runs one worker per queue, but the lock is free insurance).

**Fan-out step semantics.** `embed_text` and `embed_frames` are fan-out steps — one job per `chunk_id` / `frame_id`. Two pieces of behavior MUST be explicit:

1. **Who enqueues children.** The `chunk` worker, in the same transaction that writes the chunks rows, calls `pgmq.send_batch('ingest_embed_text', [...one payload per chunk_id...])` and writes an `(video_id, 'embed_text', chunk_id, status='pending')` row per chunk into `ingest_step_status`. If the chunk worker crashes between insert and enqueue, transaction rollback un-writes both the chunks rows and the queue messages — the chunk job stays in-flight and the next dequeue retries cleanly (the chunks insert is idempotent via the `UNIQUE (video_id, chunking_strategy, start_sec, end_sec)` constraint). Same pattern for `frame_sample` worker enqueueing `embed_frames` children.

2. **When the fan-out is "done" for a video.** There is no `(video_id, 'embed_text', 0)` summary row; entity_id=0 is reserved for whole-video steps. `ingest.pipeline.next_step(video_id)` determines fan-out completion by counting:
   ```sql
   SELECT
     (SELECT count(*) FROM chunks WHERE video_id=$1) AS expected,
     (SELECT count(*) FROM ingest_step_status
        WHERE video_id=$1 AND step='embed_text' AND status='completed') AS done
   ```
   `embed_text` is complete for the video when `done == expected AND expected > 0`. Same shape for `embed_frames` against `frames`. The reconciler treats `done < expected` as "still pending" and surfaces stalled children (`status='failed'` or `attempts >= 3` without completion) so they can be replayed manually via `make replay-dlq`.

**Failure isolation.** PGMQ visibility timeout = 5 min. After `attempts >= 3`, the worker explicitly moves the message to `<queue>_dlq` and writes `status = 'failed'` with `error_payload` set. Surrounding jobs keep flowing. Manual replay: `make replay-dlq QUEUE=ingest_embed_text`.

**Concurrency.** One worker process per queue in v0 (1 video at a time end-to-end). `embed_text` and `embed_frames` can lift to 2 workers each if MPS contention allows — usually doesn't because BGE-M3 and ColQwen2.5 both want the GPU. The schema is concurrency-safe regardless via FOR UPDATE.

**Job payload schemas** (`queues/jobs.py`):

```python
@dataclass(frozen=True)
class FetchJob:        video_id: str
@dataclass(frozen=True)
class AsrJob:          video_id: str; audio_path: str; timeout_sec: int = 1800
@dataclass(frozen=True)
class FramesJob:       video_id: str; sample_every_sec: float = 10.0
@dataclass(frozen=True)
class ChunkJob:        video_id: str; strategy: str = "fixed_window"
@dataclass(frozen=True)
class EmbedTextJob:    video_id: str; chunk_id: int
@dataclass(frozen=True)
class EmbedFramesJob:  video_id: str; frame_id: int
```

**WhisperX timeout.** `ingest/asr.py` runs WhisperX inside a thread with `Thread.join(timeout_sec)`. Default `1800s` (30 min) covers a 60-min talk at ~5–10× real-time on MPS plus model-load slack. On timeout the thread is abandoned (best-effort; WhisperX MPS hangs are unrecoverable cleanly), the job records `error_payload={"reason":"asr_timeout"}`, status flips to `failed`, message moves to DLQ. Fallback path: re-enqueue with `captions_source` set back to `youtube_auto` and skip ASR (accept lower transcript quality with a `notes` flag on the talk).

**Caption quality probe (`ingest/quality.py`).** Decides whether to skip ASR and use the YouTube auto-captions directly:

```python
def probe(vtt_text: str, duration_sec: int) -> Literal["accept", "rerun"]:
    cues = parse_cues(vtt_text)
    words_per_sec = sum(len(c.text.split()) for c in cues) / max(1, duration_sec)
    placeholder_pat = re.compile(r"\[(music|applause|silence|inaudible)\]", re.I)
    placeholder_ratio = (
        sum(1 for c in cues if placeholder_pat.fullmatch(c.text.strip())) / max(1, len(cues))
    )
    if words_per_sec < 1.0 or placeholder_ratio > 0.30:
        return "rerun"
    return "accept"
```

Thresholds are heuristic and phase-2 may tune them; they capture the failure mode where YouTube emits long stretches of `[Music]` placeholders or extremely sparse cues. ASR runs are expensive — running unnecessarily wastes 10+ min per talk; skipping inappropriately corrupts every downstream chunk.

---

## 4. Retrieval API

### RRF

For document `d` with ranks `r_c(d)` in each channel `c` (1-indexed; missing channels contribute 0):

```
fused_score(d) = Σ over c in {bm25, dense, sparse, multivec, visual} : 1 / (k + r_c(d))
k = 60
```

`k = 60` is the published RRF default; not tuned in phase 1. Phase 2 chunking-ablation report will note if a sweep is justified.

### `retrieve()` flow (5 truly independent channels)

```
1. Fan out 5 channel queries in parallel (asyncio.gather):
     bm25      Postgres ts_query against chunks.tsv → top-30 chunks
     dense     pgvector HNSW over dense_embeds → top-30 chunks
     sparse    pgvector sparsevec HNSW over sparse_embeds → top-30 chunks
     multivec  Two-stage:
                 (a) coarse: dense+sparse top-100 union as candidate set
                 (b) MaxSim in Python over the candidates' chunk_token_embeds
                     (loaded in ONE batched SELECT: WHERE chunk_id = ANY($1))
                 → top-30 chunks
     visual    Two-stage and INDEPENDENT of text channels:
                 (a) coarse: pgvector HNSW over frames.pooled_embedding using
                     ColQwen text-encoder query embedding → top-200 frames
                 (b) map frames → containing chunks (frame_sec ∈ chunk span)
                 (c) MaxSim refine over those chunks' frame_patches against
                     query patches → top-30 chunks
2. Materialize chunk metadata for the union (single SELECT IN).
3. Compute fused_score per chunk; sort desc; return top (5 * top_k_per_channel)
   chunks with channel_ranks populated.
```

**Why visual is a true 5th channel now (rev 2 change).** Rev 1 had visual MaxSim operate over the dense+sparse top-100, making it effectively a reranker over transcript-selected chunks — not an independent retrieval signal. Rev 2 adds `frames.pooled_embedding` (single ColQwen-pooled vector per frame, HNSW-indexed) so the visual channel finds slide-bearing chunks regardless of whether the transcript channels picked them. Storage cost: ~6KB per frame × ~100 frames per slide-bearing talk × ~250 talks = ~150MB at v1 — negligible.

**Visual channel recall is bounded by frame sampling density.** Default `sample_every_sec=10.0`, so a 30s chunk has ~3 frames. A chunk with zero frames in its span (e.g., a transcript-only segment of a code_heavy talk where ffmpeg sampled nothing in the chunk's window) cannot be a visual-channel hit; RRF treats this as a missing-channel contribution (the chunk's `channel_ranks` dict simply omits `visual`). For v0 the 1-frame-per-10s density is sufficient; if phase-2 results show recall starved on slide-heavy talks, the lever is sampling density, not channel design.

**MaxSim sketch (batched):**

```python
def maxsim_batch(query_token_vecs: np.ndarray,    # (Tq, D), L2-normalized
                 chunks_tokens: dict[int, np.ndarray],  # chunk_id → (Tc, D)
                ) -> dict[int, float]:
    scores: dict[int, float] = {}
    for chunk_id, chunk_token_vecs in chunks_tokens.items():
        sims = query_token_vecs @ chunk_token_vecs.T   # (Tq, Tc)
        scores[chunk_id] = float(sims.max(axis=1).sum())
    return scores

# Caller:
candidate_ids: list[int] = ...                       # ≤100 from dense+sparse union
rows = await db.fetch(
    "SELECT chunk_id, position, embedding FROM chunk_token_embeds "
    "WHERE chunk_id = ANY($1) ORDER BY chunk_id, position",
    candidate_ids,
)
chunks_tokens = group_by_chunk(rows)                 # one SELECT, not N
scores = maxsim_batch(query_vecs, chunks_tokens)
```

Per-chunk MaxSim cost: (Tq ≤ 32) × (Tc ≤ 120) cosine matmul ≈ 4K ops. 100 candidates ≈ 400K ops. Dwarfed by the network/HNSW cost.

### `rerank()` flow

Single-process: load BGE-reranker-v2-m3 once, batch all candidates, sort by reranker score, return top-`top_k`. Cross-encoder pass on 150 chunks (5 channels × top-30) is ~80ms on M-series CPU per BGE benchmarks. If real-world p95 > 80ms, hosted alternative per `model_candidates.yaml` reranker minimums.

---

## 5. LangGraph loop — `generate/`

The product path. The user-facing `answer(query, corpus_id)` runs through this graph; the bakeoff harness in §6 does NOT.

### Typed state

```python
# generate/state.py
from typing import TypedDict, Literal

class AgentState(TypedDict, total=False):
    # Inputs (immutable through the graph run). All model candidates are
    # required at run start — answer() raises BakeoffNotYetRunError if any
    # is None and no locked bakeoff winner exists for that component.
    query: str
    corpus_id: str
    generator_candidate: GeneratorCandidate     # required; from model_candidates.yaml
    judge_candidate: JudgeCandidate             # required; from model_candidates.yaml
    planner_candidate: CheapExtractionCandidate | None
                                                # None → Plan uses deterministic fallback
                                                # not-None → Plan calls the LLM
    verify_confidence_threshold: float          # from env or default 0.6

    # Plan output
    question_type: Literal["single_clip", "synthesis"]
    sub_queries: list[str]                      # query decomposition

    # Retrieve output (mutable across iterations)
    retrieved: list[RetrievedChunk]
    iteration: int                              # which Verify→Retrieve cycle (0-indexed)
    refined_query: str | None                   # set by Verify when looping back

    # Rerank output
    reranked: list[RetrievedChunk]              # top-8 from rerank()

    # Verify output
    verify_confidence: float                    # 0..1, judged on top-3
    verify_reason: str                          # human-readable

    # Generate output
    raw_answer: str                             # JSON string from generator
    parsed: GenerationOutput | None             # None on parse failure
    parse_ok: bool

    # Cite output (terminal)
    final: AnswerResult

    # Tracing
    trace_id: str
```

`AgentState` is `TypedDict(total=False)` so phase 3 can add fields (e.g., verifier-tuning telemetry) without breaking phase-1 traces. The jsonb columns `trace_spans.input`/`output` query gracefully with missing keys (`->>` returns NULL). To minimize trace storage (see §9), spans store `chunk_ids: list[int]` rather than full chunk text — the phase-4 trace viewer joins `chunks` for display on read.

### Nodes

```python
# generate/nodes/plan.py
def plan(state: AgentState) -> AgentState:
    # Two paths, picked by whether the caller supplied a planner_candidate.
    #
    # 1. state["planner_candidate"] is None — DETERMINISTIC PATH (default
    #    pre-bakeoff). No LLM call. Heuristic:
    #      - question_type = "synthesis" if the query matches synthesis
    #        cues (e.g., r"\b(compare|contrast|both|and .+ (say|argue))\b"
    #        or multiple known speaker names from talks.yaml); else
    #        "single_clip".
    #      - sub_queries = [query]. Synthesis decomposition into multiple
    #        sub-queries requires an LLM; deterministic mode degrades to a
    #        single retrieval pass over the original query.
    #    Cheap, reproducible, free, and honest about its limits.
    #
    # 2. state["planner_candidate"] is set — LLM PATH. Calls
    #    providers.chat_completion(planner_candidate, ...) with a structured
    #    prompt that emits JSON {question_type, sub_queries[]}. On parse
    #    failure: degrades to deterministic path with a `notes` log entry.
    #
    # No silent LLM default. Caller must explicitly pass planner_candidate
    # via PLANNER=... (see §7), or accept the deterministic path.

# generate/nodes/retrieve.py
def retrieve(state: AgentState) -> AgentState:
    # Calls retrieve.api.retrieve(state["refined_query"] or state["query"]).
    # For synthesis, runs once per sub_query and merges (still RRF, channel
    # ranks computed per sub_query and summed).

# generate/nodes/rerank.py
def rerank(state: AgentState) -> AgentState:
    # retrieve.api.rerank(query, retrieved, top_k=8)

# generate/nodes/verify.py
def verify(state: AgentState) -> AgentState:
    # Uses state["judge_candidate"] (required; passed by answer()) to score
    # whether top-3 reranked chunks plausibly contain the answer. Returns
    # confidence 0..1 and a reason string.
    # If confidence < state["verify_confidence_threshold"]
    #    AND state["iteration"] < 2:
    #      sets refined_query (judge proposes a rewrite) and signals loop.
    # Else: signals continue → generate.
    #
    # Threshold default 0.6, overridable via env
    # LENSGRAPH_VERIFY_CONFIDENCE_THRESHOLD so phase-3 tuning sweeps don't
    # require code changes.

# generate/nodes/generate.py
def generate(state: AgentState) -> AgentState:
    # Calls providers.chat_completion(state["generator_candidate"], ...)
    # over the top-8 reranked chunks. Prompts for strict JSON conforming to
    # GenerationOutput (see §6). Uses the SHARED parse_generation_output()
    # helper (generate/parser.py) so this code path and minimal_generation
    # cannot drift. On parse failure: parse_ok=False, answer=None, latency
    # captured. No retry.

# generate/nodes/cite.py
def cite(state: AgentState) -> AgentState:
    # Map each model-emitted claim's citation indices (answer_claim_index)
    # to actual [video_id, start_sec, end_sec] tuples from state["reranked"].
    # Build AnswerResult. If parsed.abstain=True: final.abstain=True,
    # answer=None.
```

### Graph wiring

```python
# generate/graph.py
def build_graph() -> CompiledGraph:
    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank)
    g.add_node("verify", verify)
    g.add_node("generate", generate)
    g.add_node("cite", cite)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "verify")

    # Conditional: verify either loops back or continues
    def verify_router(state: AgentState) -> str:
        if state["verify_confidence"] < 0.6 and state["iteration"] < 2:
            return "retrieve"   # loop back with refined_query
        return "generate"
    g.add_conditional_edges("verify", verify_router,
                            {"retrieve": "retrieve", "generate": "generate"})

    g.add_edge("generate", "cite")
    g.add_edge("cite", END)
    return g.compile()
```

**Iteration cap = 2.** A single verify-then-retry covers the common "model needs to re-search" case; two covers the rare cascade. Beyond that, the user is better served by an abstention with a "refine your question" message than by a third loop.

### Tracing

`generate/trace.py` wraps every node with a context manager that:
1. Allocates a `span_id` (UUIDv7).
2. Records `started_at`, input snapshot, parent span.
3. On exit: records `ended_at`, output snapshot, metadata (`{model_id, latency_ms, token_count, error?}`), status.

Spans batch-flush at graph completion (one `INSERT ... VALUES (...), (...)` per trace) to avoid hot-path overhead. `traces` row updates inline at start + completion. Trace tail-sampling is a v2 concern; v0 stores all traces.

### Public entry point — no silent model defaults anywhere

```python
# generate/api.py
class BakeoffNotYetRunError(RuntimeError):
    """Raised when answer() is called without an explicit candidate AND no
    locked bakeoff winner exists for that component. Caller must pass an
    explicit candidate or wait for the bakeoff to lock a winner."""

class CrossFamilyViolationError(RuntimeError):
    """Raised when both generator and judge are explicitly passed AND share
    a model family. ADR 004 v3.1 anti-preference-leakage rule."""

async def answer(
    query: str,
    *,
    corpus_id: str = "ai_engineering_v0",
    generator_candidate: GeneratorCandidate | None = None,
    judge_candidate: JudgeCandidate | None = None,
    planner_candidate: CheapExtractionCandidate | None = None,
) -> AnswerResult:
    # Resolve generator: explicit > locked winner > raise.
    if generator_candidate is None:
        generator_candidate = load_bakeoff_winner_or_raise(
            component="generator", corpus_id=corpus_id,
        )

    # Resolve judge: explicit > locked winner > raise.
    if judge_candidate is None:
        judge_candidate = load_bakeoff_winner_or_raise(
            component="judge", corpus_id=corpus_id,
        )

    # Enforce cross-family discipline whenever both are present (always, now).
    if judge_candidate.family == generator_candidate.family:
        raise CrossFamilyViolationError(
            f"judge family '{judge_candidate.family}' must differ from "
            f"generator family '{generator_candidate.family}' (ADR 004 v3.1)"
        )

    # Planner is optional — None means deterministic Plan (no LLM, no default).
    # Explicit passing engages the LLM path.

    state: AgentState = {
        "query": query,
        "corpus_id": corpus_id,
        "generator_candidate": generator_candidate,
        "judge_candidate": judge_candidate,
        "planner_candidate": planner_candidate,
        "verify_confidence_threshold": float(
            os.getenv("LENSGRAPH_VERIFY_CONFIDENCE_THRESHOLD", "0.6")
        ),
        "iteration": 0,
        "trace_id": new_uuid7(),
    }
    graph = build_graph()
    final_state = await graph.ainvoke(state)
    return final_state["final"]
```

`load_bakeoff_winner_or_raise(component, corpus_id)` queries `eval_runs`:

```sql
SELECT generator_model_id      -- or judge_model_id, etc.
  FROM eval_runs
 WHERE summary->>'winner_locked' = 'true'
   AND summary->>'component'     = $1     -- 'generator' | 'judge' | 'cheap_extraction'
   AND chunking_strategy         = 'fixed_window'   -- v0 lock
 ORDER BY created_at DESC
 LIMIT 1
```

If the query returns no row, **it raises `BakeoffNotYetRunError`**. The error message is copy-pasteable:

```
BakeoffNotYetRunError: no locked winner for component='generator' in corpus
'ai_engineering_v0'. Pass generator_candidate explicitly, e.g.:

    make answer QUERY="..." GENERATOR=qwen3-235b-a22b-instruct \
                            JUDGE=deepseek-v3.2

Once the phase-2 generator+judge bakeoff lands a winner_locked=true row in
eval_runs, this fallback path auto-resolves.
```

**There is no silent default anywhere.** Pre-bakeoff, every `answer()` call requires the caller to name the generator and the judge. Post-bakeoff, the locked winners auto-resolve and the caller can omit them. Planner is the only model-using node where "no LLM at all" is the default — and that's a *deterministic* default, not a model choice.

**Cross-family enforcement.** Whenever both generator and judge candidates are present (always, now), `judge.family != generator.family` is enforced. ADR 004 v3.1's anti-preference-leakage rule becomes a runtime invariant, not a checklist item.

---

## 6. `minimal_generation` — internal bakeoff harness

NOT the product path. Used by phase-2 generator+judge bakeoff to isolate generator quality from Plan/Verify/Cite confounds.

```python
# eval/runners/minimal_generation.py
from pydantic import BaseModel, Field

class AnswerClaim(BaseModel):
    text: str = Field(min_length=1)

class Citation(BaseModel):
    video_id: str
    start_sec: float
    end_sec: float
    answer_claim_index: int = Field(ge=0)   # index into GenerationOutput.claims
                                            # (NOT into gold expected_claims — generator never sees those)

class GenerationOutput(BaseModel):
    answer: str
    claims: list[AnswerClaim]               # model's own decomposition
    citations: list[Citation]
    abstain: bool = False

@dataclass(frozen=True)
class GenerationResult:
    answer: str | None        # None when parse_ok=False
    claims: list[AnswerClaim]
    citations: list[Citation]
    parse_ok: bool
    latency_ms: int           # total wall-clock, non-streaming (see below)
    raw_response: str         # always populated for debugging + judge audit trail
    model_id: str
    abstain: bool

def generate_for_bakeoff(
    query: str,
    chunks: list[RetrievedChunk],
    candidate: GeneratorCandidate,
) -> GenerationResult: ...
```

**Citation fix (rev 2 finding #3).** Rev 1's `Citation.claim_index` referenced the gold `expected_claims` array. The generator must NEVER see gold labels. Rev 2: the generator emits its OWN `claims` list, and citations reference `answer_claim_index` into that list. The **judge** (phase 2) is what maps model-emitted claims to gold-expected claims using semantic similarity — that mapping is the judge's job, not the generator's. This isolation is required by the anti-contamination invariant.

**Prompt shape.** Single user message: query + top-`k` retrieved chunks with `[video_id @ start_sec-end_sec]` labels. System message: strict JSON conforming to `GenerationOutput`; instructions to decompose answer into atomic claims; abstain when chunks don't contain the answer; cite by `answer_claim_index` for every claim.

**Parse strategy.** `parse_generation_output(response: str) -> tuple[GenerationOutput | None, bool, str | None]` lives in `generate/parser.py` and is imported by BOTH `minimal_generation.generate_for_bakeoff` AND `generate/nodes/generate.py`. Single source of truth — the two code paths cannot drift on JSON parse semantics, which would make the phase-2 bakeoff and phase-3 LangGraph comparison measure different things. Internals: `GenerationOutput.model_validate_json(response)`; any validation error → `(None, False, error_message)`, latency captured by the caller, raw stored. **No retry.** Parse failures are real bakeoff signal.

**Latency definition.** Total wall-clock from request send to full response receive. ADR 004 v3.1's `p95 generation latency ≤ 2.0s` minimum is interpreted as total latency in batch (non-streaming) mode. Phase 1 + 2 do not stream. If TTFT becomes a phase-4 demo concern, the harness adds a streaming code path and re-runs the bakeoff.

**Provider abstraction** (`eval/runners/providers.py`):

```python
def chat_completion(
    model: str,                  # provider_model_id from yaml
    messages: list[dict],
    *,
    provider: Literal["deepinfra", "openrouter"] = "deepinfra",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_sec: float = 30.0,
    max_retries: int = 3,
    backoff_base_sec: float = 1.0,
) -> tuple[str, int]:            # (response_text, latency_ms)
    # Exponential-backoff retry on 429 and 5xx (NOT on 4xx other than 429).
    # Latency captured even on final failure (timeout counts).
    ...
```

OpenRouter engages only when `LENSGRAPH_PROVIDER_FAILOVER=1`; never the default request path.

---

## 7. Local dev environment

### Custom Postgres image (rev 2 finding #2)

The `pgvector/pgvector:pg16` image does NOT bundle pgmq. The PGMQ official image does not bundle pgvector ≥ 0.7. Solution: a small `Dockerfile.postgres` at repo root that layers pgmq onto pgvector:

```dockerfile
# Dockerfile.postgres
FROM pgvector/pgvector:pg16

# Install pgmq from PGDG apt repo (Debian-based pgvector image).
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-16-pgmq \
 && rm -rf /var/lib/apt/lists/*
```

`docker-compose.yml`:

```yaml
services:
  postgres:
    build:
      context: .
      dockerfile: Dockerfile.postgres
    image: lensgraph/postgres:pg16-pgvector-pgmq
    container_name: lensgraph-pg
    environment:
      POSTGRES_USER: lensgraph
      POSTGRES_PASSWORD: lensgraph
      POSTGRES_DB: lensgraph
    ports: ["5432:5432"]
    volumes: [ "pgdata:/var/lib/postgresql/data" ]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lensgraph -d lensgraph"]
      interval: 5s
      retries: 10

volumes:
  pgdata:
```

If `postgresql-16-pgmq` is not available in the base image's apt repos at build time (PGDG release cadence), the alternative is `FROM tembo.io/tembo/pg17-pgmq:latest` then `CREATE EXTENSION vector` — verify pgvector ≥ 0.7 is present before adopting.

### Setup time honesty

First-run downloads:

| Asset | Approx size |
|---|---|
| BGE-M3 | ~2.3 GB |
| BGE-reranker-v2-m3 | ~2.3 GB |
| ColQwen2.5 | ~14 GB |
| WhisperX large-v3 | ~3 GB |
| Custom Postgres image build | ~300 MB |

**Total: ~22 GB.** At typical home broadband ~100 Mbps that's 30 min unbottlenecked; realistically **45–60 min** including Docker pulls and HuggingFace handshakes.

### Host dependencies

- `ffmpeg ≥ 6.0` on `$PATH`. macOS: `brew install ffmpeg`. Used by `ingest/frames.py`. Not bundled in `docker-compose.yml` because frame sampling runs in the worker process, not in Postgres.
- `uv ≥ 0.4` for env management (already required).

### New `make` targets

Existing targets unchanged: `validate-evals`, `phase0-gate`, `validate-self-test`, `test`, `fmt`, `lint`, `clean`.

```
db-up            docker compose up -d postgres
db-down          docker compose stop postgres
db-build         docker compose build postgres        (custom image; gated in week-3 step 2)
db-verify-ext    psql -c 'CREATE EXTENSION pgmq; CREATE EXTENSION vector;' against the built image
db-migrate       uv run python -m db.migrate
db-reset         destructive: drop + recreate + apply all migrations from 0001.
                 No partial rollback — v0 disaster-recovery path. Confirmation prompt.
models-warm      download BGE-M3 + BGE-reranker + ColQwen2.5 + WhisperX without running
ingest           uv run python -m ingest.cli $(VIDEO_ID)
ingest-all       uv run python -m ingest.cli --corpus $(CORPUS)
workers          uv run python -m queues.workers --queues fetch,asr,frames,chunk,embed_text,embed_frames
retrieve-test    uv run python -m retrieve.api --query "$(QUERY)"

# Product path. Pre-bakeoff (today), every model node requires explicit choice
# OR accepts the deterministic Plan default. Post-bakeoff, GENERATOR/JUDGE
# auto-resolve to the locked winner and only QUERY is required.
answer           uv run python -m generate.api \
                   --query "$(QUERY)" \
                   --generator "$(GENERATOR)" \           # required pre-bakeoff
                   --judge "$(JUDGE)" \                   # required pre-bakeoff; cross-family enforced
                   $(if $(PLANNER),--planner "$(PLANNER)") # optional; omit for deterministic plan
                 # On BakeoffNotYetRunError, prints copy-pasteable fix message.

replay-dlq       uv run python -m queues.workers --replay-dlq $(QUEUE)
logs             tail -f $${LENSGRAPH_LOG_PATH:-./.lensgraph/logs/workers.log}
bakeoff-prep     verify chunks + embeddings exist for every video_id referenced by dev_gold;
                 print red/green per video
```

**Pre-bakeoff `make answer` example** (period before phase-2 winners are locked):

```bash
make answer QUERY="how does Tengyu Ma compare long context, fine-tuning, and RAG" \
            GENERATOR=qwen3-235b-a22b-instruct \
            JUDGE=deepseek-v3.2
# planner omitted → deterministic Plan (single_clip, no decomposition)

make answer QUERY="..." GENERATOR=gemma-4-31b JUDGE=deepseek-v3.2 PLANNER=gemma-4-e4b
# planner explicit → LLM Plan via local Gemma-4-e4b
```

**Cross-family violations fail fast.** `make answer QUERY="..." GENERATOR=gemma-4-31b JUDGE=gemma-4-31b` raises `CrossFamilyViolationError` before any LLM call lands.

**Post-bakeoff `make answer` (phase 2+):**

```bash
make answer QUERY="..."
# auto-resolves GENERATOR + JUDGE from eval_runs.summary.winner_locked = true
# explicit args still override
```

CI runs `make test` (fast tests only) between `validate-evals` and `lint`. `make test-slow` stays local-only (requires model downloads).

Env vars: `POSTGRES_DSN`, `LENSGRAPH_PROVIDER_FAILOVER`, `LENSGRAPH_LOG_PATH`, `LENSGRAPH_VERIFY_CONFIDENCE_THRESHOLD` (default 0.6), `DEEPINFRA_API_KEY`, `OPENROUTER_API_KEY` (optional), `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` (optional, only when local MPS OOMs on ColQwen2.5).

---

## 8. Build sequence (TDD: RED → GREEN → REFACTOR)

Each milestone produces an artifact verifiable against the 11 committed dev_gold examples. `make phase0-gate` is a hard precondition before any step (green at `b805f16`).

**Slow-test gating.** Tests that load real models (BGE-M3, BGE-reranker, ColQwen2.5, WhisperX) or hit Postgres are marked `@pytest.mark.slow`. `make test` runs fast tests only (CI). `make test-slow` runs everything (local pre-commit when touching `embed/`, `retrieve/`, `generate/`). Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["slow: requires model load or live Postgres"]
addopts = "-m 'not slow'"     # default to fast-only
```

### Week 3 — substrate + ingestion

1. RED: `eval/tests/test_db_migrations.py` — apply migrations 0001..0005 to a fresh test DB, assert schema shape (extensions, tables, key columns typed correctly, **sparsevec dim is 250002**).
2. GREEN: `db/migrate.py`, `db/migrations/0001..0005`, `db/conn.py`, `Dockerfile.postgres`. **GREEN-blocking gate before any further work:** `make db-build && make db-up && make db-verify-ext` MUST succeed — verifies both `CREATE EXTENSION pgmq` and `CREATE EXTENSION vector` succeed in the built image. If `postgresql-16-pgmq` is unavailable on PGDG, swap to the Tembo `quay.io/tembo/pg17-pgmq` base in `Dockerfile.postgres`, then re-verify pgvector ≥ 0.7 with `psql -c "\dx"`. This is the first failure surface where the design's image-strategy assumption gets tested; surfacing it now avoids cascading week-3/4 work on a broken substrate.
3. RED: `eval/tests/test_talks_repo.py` — round-trip a talks.yaml row.
4. GREEN: `db/repos/talks.py`, `db/repos/ingest_step_status.py` (upsert, claim-for-update, mark-completed/failed).
5. RED: `eval/tests/test_pgmq_smoke.py` — enqueue + transactional consume + ack a no-op job; verify status row transitions.
6. GREEN: `queues/pgmq_client.py`, `queues/workers.py` skeleton.
7. RED: `eval/tests/test_ingest_fetch.py` — `LocalFsFetcher` reads a pre-staged fixture VTT, writes talks row, enqueues `chunk` (skip asr/frames in this test via `format_tags` and quality probe).
8. GREEN: `ingest/fetch.py` with `Fetcher` Protocol + `YtDlpFetcher`; `ingest/quality.py`; `ingest/pipeline.py` reconciler.
9. **Artifact:** `make ingest VIDEO_ID=W_CYk2ogcDI` writes talks row + enqueues downstream jobs (no embeddings yet). `make bakeoff-prep` reports text/embeddings missing for W_CYk2ogcDI — expected.

### Week 4 — chunking + text embeddings + 4 text channels + visual prefilter

10. RED: `eval/tests/test_fixed_window.py` — 60s synthetic VTT → assert chunk counts, span widths, overlap, sentence-boundary snap, frame_secs population, `max_tokens` enforcement.
11. GREEN: `chunking/fixed_window.py`, `chunking/__init__.py` registry.
12. RED: `eval/tests/test_bge_m3.py` (`@pytest.mark.slow`) — dense shape (1024,), sparse non-empty, multi-vector token count matches tokenizer, `vocab_size == 250002` invariant.
13. GREEN: `embed/bge_m3.py` singleton (lazy MPS, batched encode).
14. Run `make ingest-all CORPUS=ai_engineering_v0`. **Artifact:** non-empty `chunks`, `dense_embeds`, `sparse_embeds`, `chunk_token_embeds` for all 3 ingested talks.
15. RED: `eval/tests/test_channels_smoke.py` (`@pytest.mark.slow`) — per channel, `retrieve_X(query)` returns a non-empty list and ranks are 1-indexed. **Functional only.** Quality metrics (TimestampRecall@30 per channel) are logged to `eval_runs` via a separate one-shot script, not asserted by pytest (finding #5).
16. GREEN: `retrieve/bm25.py`, `retrieve/dense.py`, `retrieve/sparse.py`, `retrieve/multivec.py`.
17. RED: `eval/tests/test_rrf.py` — fused score correctness on synthetic ranks (closed-form, no model).
18. GREEN: `retrieve/rrf.py`.
19. RED: `eval/tests/test_visual_prefilter.py` (`@pytest.mark.slow`) — pooled embedding for one slide-bearing video, HNSW returns frames in cosine-sim order.
20. GREEN: `embed/colqwen.py` exposes `encode_image_pooled()` + `encode_text_query()`; `retrieve/visual.py` two-stage flow.
21. **Artifact:** `make retrieve-test QUERY="..."` returns 5-channel union with channel_ranks for one of the 11 dev_gold examples; manual visual inspection shows the gold chunk in top-8.

### Week 5 — frames-full + reranker + LangGraph loop + bakeoff harness

22. RED: `eval/tests/test_frames.py` — synthetic short video → sampled frames at 10s intervals; sha256 stable across runs.
23. GREEN: `ingest/frames.py` via ffmpeg subprocess.
24. RED: `eval/tests/test_colqwen_patches.py` (`@pytest.mark.slow`) — patch embeddings + pooled embedding per frame.
25. GREEN: `embed/colqwen.py` full implementation (patches + pooled). Run frame ingest on all 3 talks.
26. RED: `eval/tests/test_rerank_functional.py` (`@pytest.mark.slow`) — reranker returns `top_k` chunks in monotonic score order, doesn't drop chunks. **Functional only.** Quality lift over RRF is captured by `scripts/log_retrieval_quality.py` writing to `eval_runs` (artifact, not test gate — finding #5). PLUS a catastrophic-regression smoke gate: `test_rerank_smoke.py::test_known_gold_chunk_in_top_5` asserts the `tengyu-rag-library-analogy` gold chunk lands top-5 both pre- and post-rerank. Tight, single-example assertion, designed to catch "reranker returns chunks in reverse order" failures — not a quality metric.
27. GREEN: `retrieve/rerank.py`.
28. RED: `eval/tests/test_minimal_generation.py` (`@pytest.mark.slow`, mocked provider) — happy path parse_ok=True via SHARED `parse_generation_output`, malformed JSON parse_ok=False, citation `answer_claim_index` references model's own claims, latency captured on both, raw_response always present.
29. GREEN: `eval/runners/minimal_generation.py`, `eval/runners/providers.py`, `generate/parser.py` (shared parser; consumed by both this step and step 31).
30. RED: `eval/tests/test_generate_nodes.py` (`@pytest.mark.slow`) — each LangGraph node tested in isolation with stub inputs; state transitions correct; trace spans emitted; spans store `chunk_ids: list[int]`, not chunk text. Plus `test_plan_deterministic.py` covering the no-LLM Plan path's classify-and-degenerate behavior.
31. GREEN: `generate/nodes/*.py`, `generate/state.py`, `generate/trace.py`, `db/repos/traces.py`.
32. RED: `eval/tests/test_generate_graph.py` (`@pytest.mark.slow`) — end-to-end on one dev_gold example with mocked generator + judge; assert (a) trace + spans persisted, (b) Verify→Retrieve loop fires when judge returns low confidence, (c) iteration cap stops at 2, (d) abstention path produces `final.abstain=True`, (e) `BakeoffNotYetRunError` raised when generator/judge omitted AND no winner_locked row exists, (f) `CrossFamilyViolationError` raised when generator+judge share a family.
33. GREEN: `generate/graph.py`, `generate/api.py` (with `load_bakeoff_winner_or_raise()`).
34. **Artifact:** `make answer QUERY="how does Tengyu Ma compare long context, fine-tuning, and RAG"` returns a cited answer + trace_id; `SELECT * FROM trace_spans WHERE trace_id = '<id>'` shows the full Plan→Retrieve→Rerank→Verify→Generate→Cite path. THIS IS THE WEEK-5 EXIT GATE.
35. **Bakeoff prep artifact:** one-off harness run via `minimal_generation.generate_for_bakeoff` on all 11 dev_gold examples with one generator candidate (arbitrarily selected, NOT "the winner") produces per-example latency, parse_ok, citations into `eval_runs` (code_path='minimal_generation'). UNLOCKS phase 2.

**Refactor passes** happen between green and the next red. Extract shared HTTP retry helper; pull provider auth into env-only construction; factor duplicated channel-query boilerplate into a `Channel` Protocol if real duplication emerges (not before).

---

## 9. Risks + top failure points

### High

1. **Per-token storage at v1.** ~75 tokens/chunk × 1024 dim × 4 bytes ≈ 300 KB/chunk. 250 talks × 50 chunks × 300 KB ≈ 3.75 GB just for multi-vector tokens. Schema fits laptop SSD; query cost contained by §4's "MaxSim only over dense+sparse top-100" rule. Plan B if v1 still struggles: drop multi-vector channel from RRF and document the loss.
2. **ColQwen2.5 MPS memory pressure.** 7B vision encoder on 16 GB unified memory is tight. Mitigation: 1 frame per 10s (not per cue), test on the shortest slide-bearing talk first (`aie_sg_2026_d2_arize_alyx` at 978s), Modal burst pre-wired via env var.
3. **pgvector sparsevec + pgmq image build.** Both pgvector 0.7+ sparsevec and the custom Postgres image are recent. Failure surfaces immediately in week-3 step 2, not later — fail-fast positioning.

### Medium

4. **WhisperX large-v3 MPS hangs.** Pre-wired timeout in `AsrJob.timeout_sec=1800`. On hang, DLQ + status='failed' + manual replay path.
5. **ffmpeg dependency.** Listed as host requirement (§7). Test in week-5 step 22 RED before any GREEN frame code.
6. **DeepInfra rate limits.** Bakeoff #2 issues ~240 calls per generator candidate. Provider abstraction has exponential-backoff retry (§6).
7. **LangGraph state-shape lock-in.** Per ADR 001, refactoring node signatures later is friction. Mitigation: keep `AgentState` keys minimal in v1; add via TypedDict `total=False` so optional fields are non-breaking.
8. **Per-query DeepInfra cost from live `make answer` traffic.** Per query: Plan (deterministic = $0, or LLM cheap_extraction ≈ $0.0003), Verify ≈ $0.0006 per iteration (up to 2 iterations worst case), Generate ≈ $0.0009. **Per-query total: $0.001–0.002.** ADR 004 v3.1's $18–33 12-week budget was computed for bakeoff traffic only (3 generators × 80 examples × ~10 cycles); it did NOT include continuous demo traffic. If `make answer` enters routine use (>50 queries/day sustained), revisit the cost ceiling: either rate-limit, cache by query hash, or amend ADR 004 with a separate demo-traffic budget line.
9. **Tuning fixed_window by vibes.** `window_sec=30.0` and `overlap_sec=5.0` are defaults grounded in the BGE-M3 token-budget arithmetic, NOT in eval scores. Resisting the urge to "just try 45s windows because they feel better" matters — those scores belong to the phase-2 chunking ablation, not to mid-week-4 intuition. See §10 chunker discipline.

### Low

8. **Chunk overlap multiplies per-token storage by ~17%.** Tune via chunker (`overlap_sec`), not schema.
9. **Trace storage growth.** Every query writes 1 traces row + 6+ trace_spans rows. At 100 queries/day = 0.6 K rows/day × 6 ≈ 4 K rows/day. Tiny for v0; tail-sampling deferred to v2.
10. **HNSW recall vs latency.** `ef_search` defaults to 40 in pgvector; at v1 scale we may need to raise. Phase 2 bakeoff is the right place to tune.

---

## 10. Intentionally deferred

Hold the line against folding these in early.

### Chunker discipline — fixed_window is the v1 baseline, not the final answer

`fixed_window` is the right phase-1 strategy: deterministic, heavily tested, fast to implement, and a clean baseline for the phase-2 chunking ablation. Three principles to hold:

1. **Keep the chunker registry from day one.** `chunking/__init__.py` exposes `get_chunker(name)` (§1). Phase 1 registers only `"fixed_window"`. Adding `"transcript_segment"`, `"slide_boundary"`, `"topic_llm"`, and `"hybrid"` in phase 2 is a one-line registry add per strategy — no surrounding refactor. The phase-2 eval runner picks via `--chunking <name>` CLI flag (per `.claude/CLAUDE.md` "Adding a new chunking strategy" workflow).
2. **Don't tune fixed_window by vibes mid-week-4.** `window_sec=30.0` and `overlap_sec=5.0` are grounded in the BGE-M3 token-budget arithmetic (§9 risk 9), not in eval scores. The instinct to "try 45s windows because the synthesis example seems to need more context" must wait for the phase-2 chunking ablation, which holds all-other-variables-constant and reports per-strategy numbers. Mid-flight intuition tuning corrupts the ablation by changing two things at once.
3. **The methodology writeup needs all 5 strategies.** The chunking ablation MDX (`eval/reports/<date>_chunking_ablation/methodology.mdx`) is the resume artifact. Publishing it on fixed_window alone is unpublishable — there's no comparison. Phase 2 must add the other 4 before any numbers ship.

### Phase 2 (weeks 6–8):
- **Corpus expansion to ~30–50 verified examples (currently 11).** Required as a prefix to bakeoff #1, because the ADR 004 selection rule's 3pp tie-break rounds to <1 example at n=11. Eats roughly week-5 spillover or carves into week-6 budget.
- Other chunking strategies (`transcript_segment`, `slide_boundary`, `topic_llm`, `hybrid`). Added to the chunker registry, not as new modules outside it.
- Topic-LLM chunker (requires `cheap_extraction` selection per ADR 004).
- Generator + judge bakeoff runner (uses `minimal_generation` directly; does NOT route through LangGraph to isolate generator confounds).
- Chunking ablation runner.
- Boundary audit infrastructure.
- Voyage / Gemini Embedding 2 fallbacks (only if BGE-M3 fails embeddings minimum).

**Phase 3 (weeks 9–10) — re-scoped in rev 2:**
- Phase 3 is now **TUNING + COMPARISON**, not building:
  - Re-run dev_gold against LangGraph using bakeoff winners.
  - Compare LangGraph-loop vs `minimal_generation` numbers to quantify the loop's contribution.
  - Tune verifier confidence threshold + iteration cap.
  - Final dev_gold numbers + writeup.
- Streaming responses (phase 4 demo concern).
- Conversational multi-turn refinement (v2+).

**Phase 4 (weeks 11–12):**
- FastAPI surface.
- Next.js + shadcn UI, streaming, citation rendering, agent-trace visualizer (reads `trace_spans` directly).
- Premium triangulation on locked `test_gold`.
- Public deploy.

**v2+:**
- External Langfuse sink (current Postgres-native tracing is sufficient).
- Auth, multi-tenancy, billing, rate limiting.
- Live ingestion of newly-uploaded talks.
- Cross-language retrieval.
- Mobile UI.
- Groq fallback for live-demo latency.
- Echo (voice agent).

---

## 11. Plan-review findings closed

### Rev 3 findings (second `/cdf:plan-review` pass)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 6 | High | `load_bakeoff_winner_or_default()` returned the first YAML candidate when no bakeoff existed — a silent model default, violating ADR 004 v3.1. | Renamed to `load_bakeoff_winner_or_raise()`. Pre-bakeoff, `answer()` and `make answer` require explicit `GENERATOR=...`. Post-bakeoff, the locked winner auto-resolves. `BakeoffNotYetRunError` documented with copy-pasteable fix message. §5. |
| 7 | High | Same silent-default risk applied to Verify (judge candidate) and Plan (cheap-extraction candidate). | Verify: `judge_candidate` added to `AgentState`, required from `answer()`, also auto-resolves post-bakeoff. Cross-family enforced (`CrossFamilyViolationError`) whenever both generator and judge are present. Plan: gains a **deterministic non-LLM path** (heuristic single_clip/synthesis classification + sub_queries=[query]) which is the default; explicit `PLANNER=...` engages the LLM path. No silent LLM default anywhere. §5. |
| 8 | High | Fan-out step completion semantics under-specified — how `embed_text`/`embed_frames` jobs get enqueued and how completion is detected. | Spelled out in §3 "Fan-out step semantics": chunk worker does transactional enqueue of N children + N status rows; completion when `count(completed children) == count(parent rows)`. Implementer cannot get this wrong by accident anymore. |
| 9 | Medium | `minimal_generation` parser and LangGraph generate-node parser could drift, making bakeoff vs LangGraph comparison measure different things. | Single `parse_generation_output()` in `generate/parser.py`, imported by both code paths. §6, §8 step 29. |
| 10 | Medium | Dockerfile.postgres approach not gated as a fail-fast check; if `postgresql-16-pgmq` apt package missing, would discover late. | Step 2 made a **GREEN-blocking gate**: `make db-build && db-verify-ext` must succeed before any further work; Tembo fallback is documented. §8 step 2. |
| 11 | Medium | Trace spans storing full chunk text in jsonb wastes ~10× storage. | Spans store `chunk_ids: list[int]`; trace viewer (phase 4) joins `chunks` for display. §5, §9 risk 9. |
| 12 | Low | Catastrophic rerank regression (e.g., reverse-order bug) had no CI gate after rerank quality moved to artifact-logging. | Added `test_rerank_smoke.py::test_known_gold_chunk_in_top_5` — tight single-example assertion. §8 step 26. |
| 13 | Low | Verify confidence threshold hardcoded to 0.6 — phase-3 tuning would need code edits. | Threshold from `LENSGRAPH_VERIFY_CONFIDENCE_THRESHOLD` env var, default 0.6. §5. |
| 14 | Low | `AsrJob.timeout_sec=900` borderline-tight for a 60-min talk at 5–10× real-time on MPS. | Default raised to 1800 (30 min). §3. |
| 15 | Low | Caption quality probe heuristic unspecified; ASR run/skip decision would be invented per implementer. | Explicit thresholds: `words_per_sec < 1.0` or `placeholder_ratio > 0.30` → rerun. §3. |
| 16 | Low | CI runs `validate-evals` + `lint` only; rev 2 added `make test` (fast tests) but didn't wire it. | CI runs `make test` between validate-evals and lint. §7. |

### Rev 2 findings (first `/cdf:plan-review` pass)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | High | Scalar `ingest_status.last_completed_step` cannot represent parallel branches or per-chunk/per-frame fan-out. | Replaced with `ingest_step_status (video_id, step, entity_id, status, ...)` per-row table. PK includes `entity_id` (0 for whole-video steps). `FOR UPDATE` on status check. §2, §3. |
| 2 | High | `pgvector/pgvector:pg16` doesn't bundle pgmq; `CREATE EXTENSION pgmq` would fail. | Custom `Dockerfile.postgres` layers `postgresql-16-pgmq` apt package onto pgvector base. Built via `docker compose build`. §7. |
| 3 | Medium | `Citation.claim_index` exposed gold `expected_claims` to the generator (test contamination). | Generator now emits its own `claims: list[AnswerClaim]`; `Citation.answer_claim_index` references the model's own list. Judge (phase 2) maps model-claims to gold-claims post-hoc. §6. |
| 4 | Medium | Visual MaxSim ran over dense+sparse top-100, making visual a reranker over transcript-selected candidates rather than an independent channel. | Added `frames.pooled_embedding vector(128)` with HNSW index. Visual channel now does HNSW prefilter on pooled embeddings (independent of text channels), then MaxSim refines. True 5-channel RRF. §2, §4. |
| 5 | Low | `test_rerank` asserted reranker quality lift over RRF on n=11 — a model-quality metric that flakes on noise. | Rerank/channel quality moved to artifact-logging via `scripts/log_retrieval_quality.py` writing to `eval_runs`. Pytest tests are functional only (top_k count, monotonic score order). §8 steps 15, 26. |

---

## 12. Plan-review hooks — knowingly traded dimensions

- **Product.** Week-5 artifact is a working `make answer QUERY="..." GENERATOR=... JUDGE=..."` end-to-end through LangGraph with trace persistence. There IS a demo at end of week 5 — text-only, no UI, but a working agent loop with citations. Trade vs rev 1: more code in phase 1, less invention in phase 3. Pre-bakeoff, every invocation must name the generator and judge explicitly; this is a deliberate friction to prevent silent ADR-004 violations.
- **Engineering.** Two ecosystem bets sit on the critical path: pgvector sparsevec + custom Postgres image (week 3, gated as a hard GREEN-block) and ColQwen2.5 MPS (week 5). Build sequence puts them where failure surfaces early enough to recover without cascading. LangGraph node-signature lock-in is real (ADR 001 con) — kept `AgentState` minimal and TypedDict-flexible to delay the pain. Cross-family enforcement is now a runtime invariant, not a checklist item.
- **UX/DX.** Fresh-laptop setup is **45–60 min** of model downloads + Docker build + Postgres spin-up. `make models-warm` lets the user kick this off during a coffee break. Logs land at `LENSGRAPH_LOG_PATH`. Pre-bakeoff `make answer` requires `GENERATOR=...` and `JUDGE=...` — the `BakeoffNotYetRunError` message tells the user exactly what to type. Post-bakeoff the CLI auto-resolves and only `QUERY` is required.
- **Risk.** Pre-wired: WhisperX timeout (1800s), ffmpeg dependency, provider retry, cross-family enforcement, deterministic Plan fallback, fan-out completion semantics. NOT pre-wired but flagged: HNSW `ef_search` tuning (deferred to phase 2), trace tail-sampling (deferred to v2), `make answer` rate-limit at >50 queries/day (revisit ADR 004 cost ceiling).
- **Execution readiness.** Step count grew from 31 (rev 1) to 35 (rev 2/3) with LangGraph in scope. **Honest budget: 30–35 h.** At 6–10 h/week this is ~3.5 weeks, not 3 — plan for week-5 work to spill into week 6 by 5–8 hours. Phase 2 still starts at "harness + answer() both ship," not at a calendar date.

---

## 13. What ships before any of this

`/cdf:plan-review` re-run against this rev 3 doc. On approval, week-3 RED tests land FIRST. No GREEN code touches `ingest/chunking/retrieve/generate/api/web` until each RED test exists for that scope AND `make phase0-gate` is still green at the commit boundary. Per CLAUDE.md hard rule 1.

**Honest budget.** 30–35 h of work into a 6–10 h/week pace. Plan for ~3.5 weeks (some week-6 spillover acceptable) rather than a tight 3. Don't compress by cutting test coverage — the RED tests are what make the GREEN code defensible during phase-2 bakeoffs.
