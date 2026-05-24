# Roadmap — 12 weeks

Calendar weeks; "week" = ~6–10 focused hours.

## Phase 0 — Eval foundation (weeks 1–2)

Goal: a green `make validate-evals` against a real corpus of ~80 examples, and a passing `make phase0-gate` (≥10 verified).

| Week | Deliverable |
|---|---|
| 1 | Schemas, validator, fixtures, CI green. Curation playbook drafted. 10–12 talks selected with `talks.yaml` complete. |
| 2 | 80 gold/negative/synthesis examples curated and verified. Locked `test_gold.jsonl` SHA in README. |

**Exit gate:** `make validate-evals` AND `make phase0-gate` pass locally. CI runs `make validate-evals` + `make lint` on every push/PR. `phase0-gate` is intentionally a *local* gate (CI cannot scope it per-directory) and is the developer's responsibility before any commit that touches `ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, or `web/`.

## Phase 1 — Ingestion + retrieval + LangGraph loop (weeks 3–5)

Goal: end-to-end ingestion, retrieval, and the **LangGraph agent loop** (the product path) working, plus the internal `minimal_generation` bakeoff harness ready for phase 2. Detailed design lives in `docs/phase-1-design.md`.

> **Scope correction (2026-05-24):** LangGraph moved from old phase 3 into phase 1. The agent loop is part of the v1 PRD and `docs/architecture.md`; deferring it created a two-month contradiction between architecture and implementation. Old phase 3 is re-scoped to tuning + comparison (see below).

| Week | Deliverable |
|---|---|
| 3 | Custom Postgres image (pgvector + pgmq layered), per-step `ingest_step_status` schema, yt-dlp ingestion + WhisperX fallback, talks repo, PGMQ workers skeleton. End-of-week artifact: `make ingest VIDEO_ID=...` writes talks row + enqueues downstream jobs. |
| 4 | Fixed-30s chunking, **BGE-M3 all three channels** (dense in pgvector, sparse in pgvector sparsevec, multi-vector as per-token table), BM25 via Postgres FTS, RRF fusion across 4 text channels, frame sampling + ColQwen2.5 pooled-embedding visual prefilter for slide-bearing talks. End-of-week artifact: `make retrieve-test QUERY="..."` returns 5-channel union with channel_ranks. |
| 5 | BGE-reranker-v2-m3, full ColQwen2.5 patch embeddings + visual MaxSim refine, **LangGraph loop** in `generate/` (Plan → Retrieve → Rerank → Verify → Generate → Cite with Verify→Retrieve back-edge, iteration cap = 2), Postgres-native tracing (`traces` + `trace_spans`), internal `eval/runners/minimal_generation.py` bakeoff harness. End-of-week artifact: `make answer QUERY="..."` returns cited answer + trace_id; bakeoff prep run logged to `eval_runs`. |

**Exit gate:** `make answer QUERY="..."` returns a cited answer end-to-end with persisted trace; `minimal_generation` harness produces parseable answers for all 11 dev_gold examples; `make phase0-gate` still green; `make test` + `make test-slow` pass.

Week-5 budget is acknowledged tight (~32 h of work into 18–30 h budget); expect ~2–4 h spillover into week 6. Phase 2 starts at "both code paths ship," not at a calendar date.

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

## Phase 3 — Loop tuning + final dev_gold (weeks 9–10)

Goal: tune the LangGraph loop (built in phase 1) using phase-2 bakeoff winners, and produce final `dev_gold` numbers. This phase is mostly tuning and comparison, not building.

| Week | Deliverable |
|---|---|
| 9 | Plug bakeoff winners (generator, judge, embeddings) into the LangGraph loop. Tune verifier confidence threshold and iteration cap against `dev_gold`. Wire optional external Langfuse sink if useful (Postgres-native tracing already exists from phase 1). |
| 10 | Re-run eval against the tuned LangGraph loop. Compare to the `minimal_generation` numbers from the phase-2 generator bakeoff to quantify what the loop (Plan + Verify + Cite) adds over a thin harness. Final `dev_gold` numbers. |

**Exit gate:** `dev_gold` ClaimsSupported ≥ 0.85; corpus-negative RefusalRate ≥ 0.90; LangGraph-vs-`minimal_generation` delta documented in the phase-3 report.

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
