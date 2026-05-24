# ADR 004 — Model selection: local vs API per component

**Status:** Accepted
**Date:** 2026-05-24 (revised same day after May 2026 model-landscape research)
**See:** Changelog at bottom for the original closed-API-default version superseded by this one.

## Context

LensGraph runs eight model-shaped components: ASR, text embeddings, visual document retrieval (ColPali-style), reranking, planner LLM, generator LLM, cheap-extraction LLM, and a cross-family LLM judge. Each can plausibly run locally, via an open-weight hosted API, or via a closed-frontier API. Picking ad hoc leads to inconsistent latency, runaway cost, and same-family judge contamination.

Two May 2026 facts force a re-think of the original closed-API-default proposal:

1. **Gemma 4 (Google, released April 2, 2026) scores 86.4% on τ2-bench Retail agentic tool use** and ranks #3 open model on Arena — closing most of the agentic gap to GPT-5.5 Pro (90.1) and Claude Sonnet 4.6. Apache 2.0, native function calling, native multimodal.
2. **DeepInfra prices Llama-class open-weight inference at ~$0.12/$0.30 per MTok** — roughly 30× cheaper than Claude Sonnet 4.6 and 8× cheaper than GPT-4.1. Together AI is ~4× more expensive than DeepInfra for the same models; Groq leads on latency but not cost.
3. **ICLR 2026 "Preference Leakage" paper** formalized cross-family judge contamination: a judge inflates outputs from its own family by 5–7%, including via "inheritance relationships." This is now empirically documented, not anecdotal.

Hardware assumption: Apple Silicon Mac (M-series, 16+ GB unified memory). Anything requiring a dedicated NVIDIA GPU runs on Modal serverless instead of locally.

## Decision matrix

| Component | Where it runs | Specific model | Why |
|---|---|---|---|
| **ASR** (ingestion) | **Local** (MPS) | `WhisperX large-v3` | Batch job, ~3x realtime on M3. No per-minute cost. Quality matches API. |
| **Text embeddings** | **API** | `voyage-3-large` (1024-dim) | Best-in-class retrieval embeddings; not open-weight; $0.18/MTok. No competitive open replacement. |
| **Visual document retrieval** | **Local** (MPS) for dev, **Modal** for batch | `ColQwen2.5` via `colpali-engine` | Tops ViDoRe V2; +5 nDCG@5 over ColQwen2-v0.1; uses Qwen2.5-VL backbone. Fits MPS for query-time. No competitive hosted API exists. |
| **Reranker** | **Local** (CPU) | `BAAI/bge-reranker-v2-m3` | ~560MB, ~50ms per 10 docs on CPU. Zero per-call cost. |
| **Planner / Generator LLM** | **API** (open-weight hosted) | **`gemma-4-31b` on DeepInfra** | 86.4% on τ2-bench agentic; #3 open on Arena; 30× cheaper than Sonnet 4.6. Native function calling and JSON. |
| **Cheap extraction LLM** (candidate question drafting, topic-chunker, claim decomposition) | **Local** (MPS) | **`gemma-4-e4b`** | Native multimodal 4B, Apache 2.0, fits comfortably in 16GB MPS. Free. Falls back to DeepInfra Gemma if quality insufficient. |
| **LLM judge** (faithfulness, boundary) | **API** (open-weight hosted), **cross-family** | **`deepseek-v3` on DeepInfra** | Different model family from Gemma (generator), Claude, and GPT. Strongest anti-preference-leakage posture given the ICLR 2026 paper. ~8× cheaper than GPT-4.1. |
| **Premium triangulation LLM** (run once on test_gold for the chunking ablation writeup) | **API** | `claude-sonnet-4-6` | One-time comparison so the published methodology reports both open-weight and frontier-API numbers. ~$3 one-off cost. |

## Provider keys

All API keys live in `.env` (template at `.env.example`):

- `DEEPINFRA_API_KEY` — generator, judge, and Gemma fallback
- `VOYAGE_API_KEY` — embeddings
- `ANTHROPIC_API_KEY` — premium triangulation run only
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` — observability for every model call

`OPENAI_API_KEY` is no longer required for the default path. Retain in `.env.example` as optional for ad-hoc spot-checks.

## Cost ceiling (back-of-envelope, 12-week budget)

| Cost driver | Estimate | Notes |
|---|---|---|
| Full eval run (Gemma 4 31B generation + DeepSeek-V3 judge across ~80 examples × 5 chunking strategies) | ~$0.50 | Run weekly during phases 2–3 = ~$4 total |
| Topic-LLM chunking on full corpus (Gemma 4 E4B local) | $0 | Local on MPS, batch overnight if needed |
| Voyage embeddings on full corpus | ~$5 | ~25M tokens × $0.18/MTok |
| Voyage embeddings on query side over 12 weeks | ~$2 | Negligible |
| Premium triangulation run (Claude Sonnet 4.6 once on test_gold) | ~$3 | One-time, for the writeup comparison |
| Langfuse Cloud | $0 (free tier) | <50K events/month covered |
| Modal GPU bursts (ColQwen2.5 ingest of 12 talks) | ~$8 | ~2 GPU-hours total |
| **Total** | **~$25** | vs ~$110 under the previous closed-API-default plan |

If costs exceed $30/month sustained, the first action is to cap eval-run frequency to fortnightly, not to swap models.

## Why not closed-frontier-API as default

The original ADR 004 v1 chose Claude Sonnet 4.6 as the default generator. Three months ago this was correct. As of May 2026 it is no longer correct because:

- **Gemma 4 31B's agentic gap to Sonnet 4.6 is ~3–4 percentage points** on τ2-bench, not 15–20pp. That gap is now within eval noise on an 80-example corpus.
- **The cost gap is 30×.** A portfolio project that pays for frontier inference when a near-equivalent open-weight option costs pennies looks like it lacks cost discipline — the opposite of the interview story we want.
- **Cross-family judge discipline is empirically required** (ICLR 2026 preference-leakage paper). Using Anthropic for both generation and judging — even with different model SKUs — exhibits the inheritance-relationship contamination the paper identifies. An open-weight judge in a third family is now the rigorous default.
- **Triangulation is still possible.** One spot-run with Claude Sonnet 4.6 on the locked test set, costing ~$3, gives the methodology writeup a "we tested both and shipped the cheaper option with the gap quantified" story — which is a much stronger interview answer than "we picked Claude."

## Why not local everything

Tempting because zero cost. Rejected because:

- **Voyage / Anthropic / DeepInfra-hosted-Gemma-31B quality** at the agentic generator role meaningfully exceeds what fits on 16GB MPS. The eval would measure my local inference setup, not the system design.
- **Cross-family judge discipline** requires the judge to be in a different family from the generator. DeepSeek-V3 via DeepInfra is cheap and separates families cleanly.
- **Demo legibility.** A recruiter clicking the live demo wants sub-2s responses. DeepInfra at 79–258 TPS on Gemma 4 31B meets that bar; local Mac MPS at 31B parameters does not.

## Why not API everything

Tempting because simplicity. Rejected because:

- **ColPali / ColQwen has no competitive hosted API** as of 2026-05. Local + Modal are the only options.
- **WhisperX local is free and good.** Paying for hosted ASR on 500h of corpus is wasteful.
- **Reranker calls per query** would dominate latency budget if pushed to an API on every retrieval; local BGE is ~5ms.
- **Gemma 4 E4B local** is genuinely good at structured-extraction tasks and free. Pushing chunker calls to API for the convenience of not running a local model is leaving 10% of total budget on the table for no benefit.

## Revisit when

- An open-weight model passes Claude Sonnet 4.6 on τ2-bench (currently behind by ~3pp) → drop the premium triangulation step.
- DeepInfra raises Gemma 4 31B pricing above $1/MTok output → re-shop providers (Together, Fireworks, Cerebras).
- A first-party ColPali API ships with sub-200ms latency → drop the Modal dependency.
- Total monthly API spend exceeds $30 for two consecutive months → tighten eval cadence before swapping providers.

## Changelog

**2026-05-24 (revised, same day):** Pivoted from closed-API default (Claude Sonnet 4.6 generator, GPT-4.1 judge) to open-weight default (Gemma 4 31B on DeepInfra generator, DeepSeek-V3 on DeepInfra judge, Gemma 4 E4B local extraction). Driven by Gemma 4's April 2, 2026 release closing the agentic gap to <4pp and DeepInfra pricing being 30× cheaper than frontier APIs. ICLR 2026 "Preference Leakage" paper makes the open-weight cross-family judge the rigorous default. Estimated 12-week budget drops from ~$110 to ~$25. Premium triangulation step (one Sonnet 4.6 run on test_gold) retained for the methodology writeup. ColPali/ColQwen2-v0.1 upgraded to ColQwen2.5 (drop-in, +5 nDCG@5).

**2026-05-24 (original):** Initial ADR. Closed-API default. Now superseded above.
