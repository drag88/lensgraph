# LensGraph

Multimodal video RAG over engineering conference talks. Ask a natural-language question, get the exact timestamped clip plus transcript quote, slide screenshot, source citation, and a visible trace of retrieval → rerank → verify → generate.

> **Status:** v0 — eval harness only. No retrieval code yet by design.

## The thesis

Most portfolio AI projects are RAG chatbots over PDFs. LensGraph is multimodal, agentic, and rigorously evaluated. The centerpiece artifacts are not the demo — they are the **two methodology reports**:

1. The **generator + judge bakeoff** that applies the ADR 004 selection rule across Gemma 4 31B / Qwen3-235B / DeepSeek V3.2 on a hand-curated gold set, with a cross-family LLM-judge calibration audit (Cohen's kappa vs human).
2. The **chunking ablation** comparing five chunking strategies under the locked generator + judge, with claim-level faithfulness and citation accuracy.

The resume line is currently a **target**, replaced with measured numbers only after the phase-4 locked-test-set run:

> **[TARGET — replaced with measured numbers in phase 4, week 12]**
> "Built a multimodal video RAG system over 500+ hours of public conference talks. Selected the generator via a 3-candidate bakeoff (Gemma 4 31B / Qwen3-235B / DeepSeek V3.2) using a published selection rule; chosen model achieved ClaimsSupported ≥ 0.85 at ~30× lower cost than frontier APIs. Evaluated five chunking strategies; hybrid (slide-boundary + topic-LLM fallback) targets top-3 timestamp recall lift of 71% → 89% and hallucination reduction of 13% → 4%."

## What is here today

- `docs/PRD.md` — product requirements
- `docs/architecture.md` — system design
- `docs/eval-methodology.md` — why the eval is the project
- `docs/roadmap.md` — 12-week build plan
- `docs/decisions/` — architecture decision records (LangGraph, Postgres-only, corpus, model selection)
- `eval/schemas/` — three executable JSON Schemas (talk, gold example, boundary audit)
- `eval/config/model_candidates.yaml` — structured candidate config consumed by the bakeoff runner
- `eval/corpora/ai_engineering_v0/` — corpus directory, currently empty but schema-valid
- `eval/curation/playbook.md` — how to build the gold set in 5 working days
- `eval/validate.py` — schema validator with self-test fixtures and phase-0 gate
- `eval/tests/fixtures/` — valid + invalid examples that prove the schema conditionals work
- `.github/workflows/ci.yml` — CI runs `make validate-evals` + `make lint` on every push/PR

## What is intentionally not here yet

`ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, `web/` are empty. Eval comes first. `make phase0-gate` must pass (≥10 verified non-negative gold examples across ≥2 talks, plus strict transcript validation) before any code lands in those directories. This is non-negotiable; it is the methodological commitment of the project.

## Quickstart

```bash
uv sync
make validate-evals            # schemas + corpora + fixture self-test (CI gate)
make validate-evals-strict     # also fails on missing transcript files (commit gate)
make validate-self-test        # only run fixture self-test (fast schema iteration)
make phase0-gate               # fails unless >= 10 verified examples + strict checks pass
                               # run before touching ingest/, chunking/, retrieve/, generate/, api/, web/
```

## License

Apache-2.0 for code. Gold set examples are factual annotations over publicly available conference talks under fair use; talk transcripts are stored under `transcripts/` and not redistributed.
