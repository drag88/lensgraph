# Architecture

## One-line summary

Postgres + pgvector (incl. sparsevec) as the single source of truth, LangGraph as the agent loop, ColQwen2.5 for visual document retrieval over slide frames, **5-channel retrieval fused via RRF** (BM25 ⊕ BGE-M3 dense ⊕ BGE-M3 sparse ⊕ BGE-M3 multi-vector ⊕ ColQwen2.5 visual), reranked by BGE-reranker-v2-m3, Langfuse for observability, Next.js + FastAPI as the surface.

## Layered view

```
┌─────────────────────────────────────────────────────────────────────┐
│ web/ — Next.js + shadcn/ui                                          │
│   streaming UI, citation rendering, agent-trace visualizer          │
├─────────────────────────────────────────────────────────────────────┤
│ api/ — FastAPI                                                       │
│   /query, /ingest, /eval, /traces                                   │
├─────────────────────────────────────────────────────────────────────┤
│ generate/ — LangGraph                                                │
│   nodes: Plan → Retrieve → Rerank → Verify → Generate → Cite        │
│   loops: Verify can re-enter Retrieve with refined query            │
├─────────────────────────────────────────────────────────────────────┤
│ retrieve/                                                            │
│   5-channel fusion via RRF:                                          │
│     BM25 (Postgres FTS)                                              │
│     BGE-M3 dense  (pgvector HNSW, 1024-dim)                          │
│     BGE-M3 sparse (pgvector sparsevec HNSW)                          │
│     BGE-M3 multi-vector (per-token arrays, MaxSim in app)            │
│     ColQwen2.5 visual (late-interaction over slide patches)          │
│   reranker: BGE-reranker-v2-m3 (local CPU)                           │
├─────────────────────────────────────────────────────────────────────┤
│ chunking/                                                            │
│   fixed_window | transcript_segment | slide_boundary | topic_llm    │
│   hybrid (slide_boundary primary, topic_llm fallback)               │
├─────────────────────────────────────────────────────────────────────┤
│ ingest/                                                              │
│   yt-dlp → WhisperX (if needed) → frame sampling → ColQwen2.5 embeds│
│   PGMQ jobs orchestrate the ingestion DAG per video                 │
├─────────────────────────────────────────────────────────────────────┤
│ Postgres                                                             │
│   tables: talks, transcripts, chunks (text), frames, frame_embeds,  │
│           dense_embeds, sparse_embeds, token_embeds,                 │
│           eval_runs, eval_results, traces                            │
│   extensions: pgvector, pgmq                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Why these choices

**LangGraph, not LangChain or LlamaIndex.** The retrieve → verify → re-retrieve loop is cyclical. LangGraph is the only mature option that treats agent state as a first-class graph. LangChain core is dying; LlamaIndex is broader but its video / multimodal indexing is weaker than rolling our own ColQwen2.5 integration.

**Postgres + pgvector, no Redis, no separate queue.** Solo-project complexity budget is finite. PGMQ (Postgres-backed message queue) handles ingestion jobs; pgvector handles embeddings; relational tables handle everything else. One backup, one connection string, one mental model.

**ColQwen2.5 for slide retrieval.** Slides are visual documents. OCR-then-embed is the 2023 approach and loses layout. ColQwen2.5 (late-interaction over patch embeddings via `colpali-engine`) is the current ViDoRe V2 leader and the 2026 default for document-heavy retrieval.

**Cross-family LLM judge.** If Claude drafts curation candidates, GPT-4-class is the judge, and vice versa. Mitigates same-model bias in faithfulness scoring.

**Specific models and local-vs-API per component:** see `docs/decisions/004-model-selection.md` (v3). The project does NOT lock a single-model default; it ships a candidate set per component and a formal selection rule that picks the cheapest candidate within 3pp of the leader on the locked eval thresholds. Generator candidates: Gemma 4 31B, Qwen3-235B-A22B-Instruct, DeepSeek V3.2 (all DeepInfra). Judge candidates: cross-family vs whichever generator wins. Embeddings: local BGE-M3 first. Visual retrieval: ColQwen2.5 via colpali-engine. ASR + reranker + cheap extraction run locally on Mac MPS/CPU. Claude Sonnet 4.6 and GPT-5.5 reserved for a one-off premium triangulation run on the locked test set. Estimated 12-week API spend: **~$18–$33** depending on bakeoff outcome.

**Conference talks as launch corpus.** See `docs/decisions/003-conference-talks-corpus.md`.

## Data flow — a query

1. User sends `{question, corpus_id}` to `POST /query`.
2. LangGraph `Plan` node decomposes the question (extract entities, decide if single-clip or synthesis).
3. `Retrieve` node fans out across 5 channels: BM25, BGE-M3 dense (pgvector), BGE-M3 sparse (pgvector sparsevec), BGE-M3 multi-vector (MaxSim over per-token arrays), and ColQwen2.5 over frame patches. Top-k = 30 per channel; RRF fuses the unions.
4. `Rerank` collapses the union to top-8 via cross-encoder.
5. `Verify` node checks whether the top-8 actually contain the answer. If confidence < threshold and not yet retried, edit query and loop back to Retrieve.
6. `Generate` produces an answer constrained to the retrieved spans.
7. `Cite` attaches `[video_id, start_sec, end_sec]` to each claim.
8. Response streams back to UI; full trace lands in Langfuse.

## Data flow — ingestion (one video)

```
yt-dlp ────────► transcript.vtt ──► sha256 ──► talks.yaml
   │                  │
   ▼                  ▼
frame sample       chunk (5 strategies, configurable)
   │                  │
   ▼                  ▼
ColQwen2.5        BGE-M3 all three channels
patch embeds      (dense + sparse + multi-vector)
   │                  │
   └──► frames        └──► chunks
        + frame_embeds      + dense_embeds   (pgvector HNSW)
                            + sparse_embeds  (pgvector sparsevec HNSW)
                            + token_embeds   (per-token arrays; MaxSim in app)
```

Each step is a PGMQ job. A video that fails mid-ingestion can resume from the last successful step.

## Eval as a first-class subsystem

`eval/` is not a tests folder. It is the product. Schemas, corpora, validator, fixtures, judges, and reports all live there. CI gates: any PR that drops `Timestamp Recall@3` by >2pp or `Faithfulness` by >3pp on the dev set fails. The test set is locked; touching it during development invalidates published metrics.

## What is deferred

- Auth, multi-tenancy, billing.
- Hybrid local + cloud inference routing (v2).
- Video chunking by speaker diarization (v2 — useful for panels).
- Cross-language support (v2).
