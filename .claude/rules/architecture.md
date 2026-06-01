---
description: Bird's-eye system view, codemap, cross-cutting concerns, invariants
---

# Architecture

## Bird's Eye View

LensGraph is a multimodal video RAG system over engineering conference talks. A user asks a natural-language question; the system returns the exact timestamped clip plus transcript quote, slide screenshot, source citation, and an inspectable trace of retrieve → rerank → verify → generate.

At v0 only the eval harness exists. The eval is the product. Implementation code (`ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, `web/`) lands only after the phase-0 gate passes: ≥10 verified non-negative gold examples across ≥2 distinct talks, plus strict transcript validation.

## Codemap

| Directory | Purpose | Status |
|---|---|---|
| `eval/schemas/` | JSON Schemas (Draft 2020-12) — `talk`, `gold_example`, `boundary_audit`, `model_candidates`. Source of truth for every data shape in the system. | populated |
| `eval/corpora/<corpus>/` | `talks.yaml` + four JSONL files (`dev_gold`, `test_gold`, `negative`, `synthesis`) + optional `boundary_audit.jsonl`. Provenance owned by talks, not examples. | scaffolded, mostly empty |
| `eval/curation/` | Curation playbook + helper scripts (`clip_vtt.py`, `talks_shortlist.md`). | populated |
| `eval/tests/fixtures/` | `valid_*.json` and `invalid_*.json` — the validator self-test asserts the schema conditionals actually reject what they should. | populated |
| `eval/runners/judges/` | Versioned judge prompts (hashed and logged with every eval run). | placeholder |
| `eval/reports/` | Versioned eval runs: `<date>_<run_id>/{summary.json, per_example.jsonl, methodology.mdx, raw/}`. | placeholder |
| `eval/config/` | `model_candidates.yaml` — structured candidate config consumed by the phase-2 bakeoff runner. Schema-validated. | populated |
| `eval/validate.py` | Single entry-point validator: schema + cross-field semantics + transcript SHA drift + fixture self-test + phase-0 gate. | populated |
| `docs/` | `PRD.md`, `architecture.md`, `eval-methodology.md`, `roadmap.md`, `decisions/`. | populated |
| `transcripts/` | Local-only WhisperX / VTT files. Gitignored; referenced by `talks.yaml`. | runtime |
| `ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, `web/` | Empty placeholders. Locked until phase-0 gate passes. | empty by design |

## Cross-Cutting Concerns

**Schema-first.** Every persistent shape passes through a JSON Schema before any code touches it. New fields require a schema update + fixture pair before the code change.

**Provenance ownership.** Transcripts have `captions_source`, `transcript_path`, `transcript_sha256`, `accessed_at` on the **talk** record. Examples reference talks by `video_id` only. This keeps synthesis examples (which span multiple videos) unambiguous and means a transcript update is a one-place edit.

**Verification discipline.** Every committed gold example carries `verified: true`. The validator rejects anything else. Curators flip the flag only after watching the actual clip.

**Cross-family judge.** Curation and judging happen in different model families. Same-family judging inflates faithfulness 5–15pp (see ADR 004). The eval methodology treats this as a contamination invariant.

**Conventional commits.** `feat:`, `fix:`, `refactor:`, `docs:`, `eval:`, `chore:`. The `eval:` type covers anything inside `eval/`.

## Architectural Invariants

- **`test_gold.jsonl` is locked.** Its SHA256 is a contract recorded in the project README. Changes invalidate every published metric and require a methodology revision note.
- **Postgres is the only datastore.** PGMQ handles queues; pgvector handles embeddings (dense + sparse + multi-vector). No Redis, Celery, RabbitMQ, or second backing store. See `docs/decisions/002-postgres-only.md`.
- **LangGraph is the only agent framework.** No LangChain core, no LlamaIndex. See `docs/decisions/001-langgraph-not-llamaindex.md`.
- **No model is the default until the bakeoff runs.** Per-component candidate sets in `eval/config/model_candidates.yaml`; ADR 004 selection rule (cheapest within 3pp of leader, all minimums met) picks the winner.
- **`test_gold` informs no decision.** Only `dev_gold` informs selection. `test_gold` is unlocked once, for the final writeup numbers.
- **Curators never see retrieval output.** Curation tooling is isolated from any system that produces ranked chunks.
- **DeepInfra is the primary inference provider.** OpenRouter is optional failover. Adding a provider requires an ADR amendment.

## Project-Type Sections

### Data Flow — a query (post phase-3)

```
POST /query → Plan → Retrieve (5 channels via RRF) → Rerank (bge-reranker-v2-m3)
            → Verify → [loop back to Retrieve if confidence < threshold]
            → Generate (constrained to retrieved spans) → Cite → stream UI + Langfuse trace
```

Five retrieval channels: BM25 (Postgres FTS), BGE-M3 dense (pgvector HNSW), BGE-M3 sparse (pgvector sparsevec HNSW), BGE-M3 multi-vector (per-token arrays + MaxSim in app), ColQwen2.5 visual (late interaction over slide patches). Top-k = 30 per channel; RRF fuses unions; reranker collapses to top-8.

### Data Flow — ingestion (one video, post phase-1)

```
yt-dlp → transcript.vtt → SHA256 → talks.yaml entry
                  ↓                          ↓
            frame sample              chunk (5 strategies, CLI-selectable)
                  ↓                          ↓
            ColQwen2.5                 BGE-M3 all three channels
            patch embeds               (dense + sparse + multi-vector)
                  ↓                          ↓
           frames + frame_embeds      chunks + dense_embeds + sparse_embeds + token_embeds
```

Each step is a PGMQ job. A video that fails mid-ingestion resumes from the last successful step.

### Environment Matrix (planned)

| Layer | Local (dev) | Hosted | Reserved |
|---|---|---|---|
| ASR | WhisperX large-v3 on MPS | — | — |
| Text embeddings | BGE-M3 (CPU/MPS) | — | Voyage / Gemini Embedding 2 (fallback only if BGE-M3 fails minimums) |
| Visual retrieval | ColQwen2.5 via `colpali-engine` | Modal (if MPS too slow) | Gemini Embedding 2 (single-vector baseline for writeup only) |
| Reranker | bge-reranker-v2-m3 on CPU | — | — |
| Cheap extraction | gemma-4-e4b / qwen3-8b | DeepInfra (fallback) | — |
| Generator | — | DeepInfra (gemma-4-31b / qwen3-235b / deepseek-v3.2) | — |
| Judge | — | DeepInfra (cross-family vs chosen generator) | — |
| Premium triangulation | — | Anthropic + OpenAI (one-off on `test_gold`) | Claude Sonnet 4.6, GPT-5.5 |
| Datastore | Postgres + pgvector + PGMQ (docker-compose) | same | — |
| Observability | Langfuse (self-host or cloud) | same | — |

12-week API budget estimate: ~$18–$33 depending on bakeoff outcome.

## What is deferred

Auth, multi-tenancy, billing, live ingestion, cross-language retrieval, mobile UI, speaker-diarization chunking, hybrid local+cloud routing, Groq fallback (until Groq adds the full candidate set). See `docs/PRD.md` for the full non-goals list.
