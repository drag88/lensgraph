# Roadmap — 12 weeks

Calendar weeks; "week" = ~6–10 focused hours.

## Phase 0 — Eval foundation (weeks 1–2)

Goal: a green `make validate-evals` against a real corpus of 80 examples.

| Week | Deliverable |
|---|---|
| 1 | Schemas, validator, fixtures, CI green. Curation playbook drafted. 10–12 talks selected with `talks.yaml` complete. |
| 2 | 80 gold/negative/synthesis examples curated and verified. Locked `test_gold.jsonl` SHA in README. |

**Exit gate:** `make validate-evals` passes against the full v0 corpus.

## Phase 1 — Ingestion + baseline retrieval (weeks 3–5)

Goal: end-to-end pipeline working with the simplest possible chunking and retrieval.

| Week | Deliverable |
|---|---|
| 3 | yt-dlp + WhisperX ingestion, fixed-30s chunking, text embeddings, pgvector + BM25 retrieval. |
| 4 | First eval run end-to-end. Baseline numbers in `eval/reports/`. |
| 5 | Frame sampling + ColPali ingestion for slide-bearing talks. Visual retrieval as a parallel channel. |

**Exit gate:** baseline TimestampRecall@3 on `dev_gold` is reported. No tuning yet.

## Phase 2 — Bakeoffs (weeks 6–8)

Goal: two centerpiece artifacts — the model bakeoff and the chunking ablation. Both apply the same eval-driven selection discipline.

| Week | Deliverable |
|---|---|
| 6 | **Generator + judge bakeoff** (ADR 004 v3 selection rule). Pin retrieval + chunking variables; vary only the generator candidate (Gemma 4 31B / Qwen3-235B / DeepSeek V3.2). Judge calibration on 20 manual boundary scores per judge candidate. Publish `eval/reports/<date>_generator_bakeoff/methodology.mdx`. Lock winner. Estimated cost: <$2. |
| 6 | Implement transcript_segment + slide_boundary chunkers. Wire chunking strategy as CLI flag. |
| 7 | Implement topic_llm + hybrid chunkers. Run all five on `dev_gold` with the locked generator + judge. |
| 8 | Boundary audit (20–30 manual scores) for the chunking ablation. Publish `eval/reports/<date>_chunking_ablation/methodology.mdx`. |

**Exit gates (both must hold):**
- Generator bakeoff: winner meets every minimum in ADR 004; runner-up and cost differential documented.
- Chunking ablation: five-strategy comparison published with judge-human kappa ≥ 0.6.

## Phase 3 — Agent loop + answer tier (weeks 9–10)

Goal: actual question answering with citations, not just retrieval.

| Week | Deliverable |
|---|---|
| 9 | LangGraph: Plan → Retrieve → Rerank → Verify → Generate → Cite. Langfuse wired up. |
| 10 | Claim-level faithfulness eval. Verifier-gated re-retrieval. Abstention working. Final dev numbers. |

**Exit gate:** `dev_gold` ClaimsSupported ≥ 0.85; corpus-negative RefusalRate ≥ 0.90.

## Phase 4 — Surface + ship (weeks 11–12)

Goal: a demo a recruiter understands in 60 seconds.

| Week | Deliverable |
|---|---|
| 11 | FastAPI + Next.js streaming UI with citation rendering and the agent-trace visualizer. |
| 12 | Lock `test_gold` numbers. Record demo. Write resume bullets. Deploy. Ship the blog post. |

**Exit gate:** the project URL is in the resume.

## Deferred to v2

- Diff mode across versioned talks.
- Live ingestion of newly-uploaded talks.
- Cross-language retrieval.
- Mobile UI.
- Echo (voice agent) — separate project.

## Weekly cadence rules

- Public GitHub from week 1. No private repo.
- One blog post or thread per week minimum, even if short.
- If a week ends with no green eval, that week's work does not exist for the purposes of the roadmap.
