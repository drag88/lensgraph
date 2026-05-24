# Roadmap — 12 weeks

Calendar weeks; "week" = ~6–10 focused hours.

## Phase 0 — Eval foundation (weeks 1–2)

Goal: a green `make validate-evals` against a real corpus of ~80 examples, and a passing `make phase0-gate` (≥10 verified).

| Week | Deliverable |
|---|---|
| 1 | Schemas, validator, fixtures, CI green. Curation playbook drafted. 10–12 talks selected with `talks.yaml` complete. |
| 2 | 80 gold/negative/synthesis examples curated and verified. Locked `test_gold.jsonl` SHA in README. |

**Exit gate:** `make validate-evals` AND `make phase0-gate` pass locally. CI runs `make validate-evals` + `make lint` on every push/PR. `phase0-gate` is intentionally a *local* gate (CI cannot scope it per-directory) and is the developer's responsibility before any commit that touches `ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, or `web/`.

## Phase 1 — Ingestion + retrieval scaffolding (weeks 3–5)

Goal: end-to-end ingestion and retrieval working, with the thin generation harness ready in time for week-5 bakeoffs.

| Week | Deliverable |
|---|---|
| 3 | yt-dlp + WhisperX ingestion, fixed-30s chunking, **BGE-M3 all three channels** (dense in pgvector, sparse in pgvector sparsevec, multi-vector as per-token arrays). BM25 via Postgres FTS. |
| 4 | RRF fusion across BGE-M3 channels + BM25. MaxSim aggregation for the multi-vector channel in app code. Frame sampling + ColQwen2.5 ingestion for slide-bearing talks. Visual retrieval as the 5th channel. |
| 5 | **Thin generation harness** in `eval/runners/minimal_generation.py` — takes `(query, retrieved_chunks, candidate_config)`, returns `(answer, citations, parse_ok, latency_ms)`. No LangGraph, no verify loop, no rerank logic — just enough to score generator candidates. **Embedding bakeoff** runs immediately after (TimestampRecall@5 measured per-channel AND fused across BGE-M3 / Voyage / Gemini Embedding 2; selection rule picks winner). |

**Exit gate:** end-to-end retrieval produces ranked chunks on `dev_gold`; minimal generation harness produces parseable answers; embedding winner locked in `eval/config/model_candidates.yaml`.

## Phase 2 — Bakeoffs (weeks 6–8)

Goal: two centerpiece methodology artifacts (generator + chunking ablations). Both apply the ADR 004 selection rule, both publish reproducible MDX reports.

| Week | Deliverable |
|---|---|
| 6 | **Generator + judge bakeoff** on `dev_gold` using the minimal generation harness from week 5. Judge candidates evaluated first via 20-example boundary audit (kappa ≥ 0.60 minimum, cross-family from each generator candidate). Generator candidates then scored with the locked judge. Publish `eval/reports/<date>_generator_bakeoff/methodology.mdx`. Lock winner in `model_candidates.yaml`. |
| 6 | Implement `transcript_segment` + `slide_boundary` chunkers. Wire chunking strategy as CLI flag. |
| 7 | Implement `topic_llm` + `hybrid` chunkers. Run all five on `dev_gold` with the locked generator + judge. |
| 8 | Boundary audit (20–30 manual scores) for the chunking ablation. Publish `eval/reports/<date>_chunking_ablation/methodology.mdx`. |

**Exit gates (all must hold):**
- Generator bakeoff: winner meets every minimum in ADR 004; runner-up + cost differential documented in MDX.
- Chunking ablation: five-strategy comparison published with judge-human kappa ≥ 0.60.

## Phase 3 — Full agent loop + answer tier (weeks 9–10)

Goal: replace the minimal generation harness with the production LangGraph loop.

| Week | Deliverable |
|---|---|
| 9 | LangGraph: Plan → Retrieve → Rerank → Verify → Generate → Cite. Langfuse wired up. Verifier-gated re-retrieval working. |
| 10 | Re-run the eval against the LangGraph loop using the locked generator + judge. Compare to the minimal-harness numbers from phase 2 to quantify what the agent loop adds. Final `dev_gold` numbers. |

**Exit gate:** `dev_gold` ClaimsSupported ≥ 0.85; corpus-negative RefusalRate ≥ 0.90; LangGraph-vs-minimal delta documented.

## Phase 4 — Surface + ship (weeks 11–12)

Goal: a demo a stranger understands in 60 seconds.

| Week | Deliverable |
|---|---|
| 11 | FastAPI + Next.js streaming UI with citation rendering and the agent-trace visualizer. |
| 12 | **Run locked `test_gold` once** for the writeup. Optionally run premium triangulation (Claude Sonnet 4.6 + GPT-5.5 on the same `test_gold` once each, ~$5 total). Update README resume line with measured numbers. Record demo. Deploy. Ship the blog post. |

**Exit gate:** the project URL is in the resume; the README resume line contains measured numbers, not targets.

## Deferred to v2

- Diff mode across versioned talks.
- Live ingestion of newly-uploaded talks.
- Cross-language retrieval.
- Mobile UI.
- Groq fallback for live-demo latency (only after Groq adds the candidate generator set).
- Echo (voice agent) — separate project.

## Weekly cadence rules

- Public GitHub from week 1. No private repo.
- One blog post or thread per week minimum, even if short.
- If a week ends with no green eval, that week's work does not exist for the purposes of the roadmap.
- Before any commit that touches `ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, or `web/`: run `make phase0-gate` locally and ensure it passes.
