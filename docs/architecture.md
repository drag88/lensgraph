# Architecture

## One-line summary

Postgres + pgvector as the single source of truth, LangGraph as the agent loop, ColPali for visual document retrieval over slide frames, BM25 + vector + reranker as the three-stage retrieval stack, Langfuse for observability, Next.js + FastAPI as the surface.

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
│   BM25 (tantivy via Postgres FTS) ⊕ pgvector (HNSW)                 │
│   reranker: BGE-reranker-v2-m3 or Voyage rerank-2                   │
├─────────────────────────────────────────────────────────────────────┤
│ chunking/                                                            │
│   fixed_window | transcript_segment | slide_boundary | topic_llm    │
│   hybrid (slide_boundary primary, topic_llm fallback)               │
├─────────────────────────────────────────────────────────────────────┤
│ ingest/                                                              │
│   yt-dlp → WhisperX (if needed) → frame sampling → ColPali embeds   │
│   PGMQ jobs orchestrate the ingestion DAG per video                 │
├─────────────────────────────────────────────────────────────────────┤
│ Postgres                                                             │
│   tables: talks, transcripts, chunks (text), frames, frame_embeds,  │
│           text_embeds, eval_runs, eval_results, traces              │
│   extensions: pgvector, pgmq                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Why these choices

**LangGraph, not LangChain or LlamaIndex.** The retrieve → verify → re-retrieve loop is cyclical. LangGraph is the only mature option that treats agent state as a first-class graph. LangChain core is dying; LlamaIndex is broader but its video / multimodal indexing is weaker than rolling our own ColPali integration.

**Postgres + pgvector, no Redis, no separate queue.** Solo-project complexity budget is finite. PGMQ (Postgres-backed message queue) handles ingestion jobs; pgvector handles embeddings; relational tables handle everything else. One backup, one connection string, one mental model.

**ColPali for slide retrieval.** Slides are visual documents. OCR-then-embed is the 2023 approach and loses layout. ColPali / ColQwen2-style late-interaction over patch embeddings is the 2026 default for any document-heavy retrieval.

**Cross-family LLM judge.** If Claude drafts curation candidates, GPT-4-class is the judge, and vice versa. Mitigates same-model bias in faithfulness scoring.

**Conference talks as launch corpus.** See `docs/decisions/003-conference-talks-corpus.md`.

## Data flow — a query

1. User sends `{question, corpus_id}` to `POST /query`.
2. LangGraph `Plan` node decomposes the question (extract entities, decide if single-clip or synthesis).
3. `Retrieve` node fans out: BM25 over transcripts + pgvector over text embeds + ColPali over frame embeds. Top-k = 30 per channel.
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
ColPali embed     text embed (Voyage-3-large)
   │                  │
   └──► frames        └──► chunks
        + frame_embeds      + text_embeds
```

Each step is a PGMQ job. A video that fails mid-ingestion can resume from the last successful step.

## Eval as a first-class subsystem

`eval/` is not a tests folder. It is the product. Schemas, corpora, validator, fixtures, judges, and reports all live there. CI gates: any PR that drops `Timestamp Recall@3` by >2pp or `Faithfulness` by >3pp on the dev set fails. The test set is locked; touching it during development invalidates published metrics.

## What is deferred

- Auth, multi-tenancy, billing.
- Hybrid local + cloud inference routing (v2).
- Video chunking by speaker diarization (v2 — useful for panels).
- Cross-language support (v2).
