# ADR 004 — Model selection: local vs API per component

**Status:** Accepted
**Date:** 2026-05-24

## Context

LensGraph runs eight model-shaped components: ASR, text embeddings, visual document retrieval (ColPali-style), reranking, planner LLM, generator LLM, topic-chunker LLM, and a cross-family LLM judge. Each can plausibly run locally or via API. Picking ad hoc leads to inconsistent latency, cost surprises, and same-family judge contamination.

Hardware assumption: an Apple Silicon Mac (M-series, 16+ GB unified memory). Anything requiring a dedicated NVIDIA GPU runs on Modal serverless instead of locally.

## Decision matrix

| Component | Where it runs | Specific model | Why |
|---|---|---|---|
| **ASR** (ingestion) | **Local** (MPS) | `WhisperX large-v3` | Batch job, runs ~3x realtime on M3. No per-minute cost. Quality matches API. |
| **Text embeddings** | **API** | `voyage-3-large` (1024-dim) | Best-in-class retrieval embeddings; not open-weight. Cheap at $0.18/MTok. |
| **Visual document retrieval** | **Local** (MPS) for dev, **Modal** for batch | `ColQwen2-v0.1` via `colpali-engine` | Fits in 16GB MPS for query-time, slow but acceptable; Modal H100 for full-corpus ingest. No competitive API option. |
| **Reranker** | **Local** (CPU) | `BAAI/bge-reranker-v2-m3` | ~560MB, runs on CPU at ~50ms per 10 docs. Voyage rerank-2 is an API alternative; we choose local for zero per-call cost. |
| **Planner / Generator LLM** | **API** | `claude-sonnet-4-6` | Best cost/quality balance for agentic flow with tools. No realistic local substitute at this quality. |
| **Cheap extraction LLM** (candidate question drafting, topic-chunker, claim decomposition) | **API** | `claude-haiku-4-5` | ~10x cheaper than Sonnet; quality sufficient for structured extraction. |
| **LLM judge** (faithfulness, boundary) | **API**, **cross-family** | `gpt-4.1` (or current OpenAI flagship) | Cross-family discipline (ADR 001 reasoning extended): Anthropic curates and generates, OpenAI judges. Avoids same-model bias. |
| **Eval-only sanity LLM** | **API** | Whatever was NOT used for curation/generation | Sanity-check spot-runs during methodology development. |

## Provider keys

All API keys live in `.env` (template at `.env.example`):

- `ANTHROPIC_API_KEY` — Claude Sonnet 4.6 + Haiku 4.5
- `OPENAI_API_KEY` — judge only
- `VOYAGE_API_KEY` — embeddings
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` — observability for every model call

No keys are committed. Pre-commit hook will run `make validate-evals` and a basic `.env` leak scan before allowing a push.

## Cost ceiling (back-of-envelope, 12-week budget)

| Cost driver | Estimate | Notes |
|---|---|---|
| Full eval run (Sonnet generation + GPT-4 judge across ~80 examples × 5 chunking strategies) | ~$13 | Run weekly during phases 2–3 = ~$80 total |
| Topic-LLM chunking on full corpus (Haiku) | ~$3 | One-time per corpus build, repeated ~3x = ~$10 |
| Voyage embeddings on full corpus | ~$5 | ~25M tokens of transcripts × $0.18/MTok |
| Voyage embeddings on query side over 12 weeks | ~$2 | Negligible |
| Langfuse Cloud | $0 (free tier) | <50K events/month covered |
| Modal GPU bursts (ColPali ingest of 12 talks) | ~$8 | ~2 GPU-hours total |
| **Total** | **~$110** | Comfortable under any reasonable solo budget |

If costs exceed $50/month sustained, the first action is to cap eval-run frequency to fortnightly during phase 2, not to swap models.

## Why not local everything

Tempting because cost. Rejected because:

- **Voyage / Anthropic / OpenAI quality gap is real** at the 7–13B local sizes that fit on a Mac. The eval would measure my limited inference setup, not the system design.
- **Cross-family judge discipline** requires two distinct model families. Running both locally means hosting two large models simultaneously — impractical on 16GB.
- **Demo legibility.** A recruiter clicking the live demo wants sub-2s responses. Local generation on a Mac cannot do that.

## Why not API everything

Tempting because simplicity. Rejected because:

- **ColPali has no competitive hosted API** as of 2026-05. Local or Modal are the only options.
- **WhisperX local is free and good.** Paying for hosted ASR on 500h of corpus is wasteful.
- **Reranker calls per query** would dominate latency budget if pushed to an API on every retrieval; local BGE is ~5ms.

## Revisit when

- An open-weight model matches Claude Sonnet 4.6 on agentic tool-use benchmarks → drop the generator API dep.
- A first-party ColPali API ships with sub-200ms latency → drop the Modal dep.
- Total monthly API spend exceeds $50 for two consecutive months → tighten eval cadence before swapping providers.
