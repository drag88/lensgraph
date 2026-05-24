# LensGraph

Multimodal video RAG over engineering conference talks. Ask a natural-language question, get the exact timestamped clip plus transcript quote, slide screenshot, source citation, and a visible trace of retrieval → rerank → verify → generate.

> **Status:** v0 — eval harness only. No retrieval code yet by design.

## The thesis

Most portfolio AI projects are RAG chatbots over PDFs. LensGraph is multimodal, agentic, and rigorously evaluated. The centerpiece artifact is not the demo — it is the **chunking ablation report** comparing five chunking strategies on a hand-curated gold set, with claim-level faithfulness scoring and judge-human agreement.

The story on the resume is:

> "Built a multimodal video RAG system over 500+ hours of public conference talks. Evaluated five video chunking strategies on a hand-curated gold set; the hybrid (slide-boundary + topic-LLM fallback) strategy lifted top-3 timestamp recall from 71% → 89% and reduced hallucinated answers from 13% → 4% via verifier-gated retrieval."

## What is here today

- `docs/PRD.md` — product requirements
- `docs/architecture.md` — system design
- `docs/eval-methodology.md` — why the eval is the project
- `docs/roadmap.md` — 12-week build plan
- `docs/decisions/` — architecture decision records
- `eval/schemas/` — three executable JSON Schemas (talk, gold example, boundary audit)
- `eval/corpora/ai_engineering_v0/` — corpus directory, currently empty but schema-valid
- `eval/curation/playbook.md` — how to build the gold set in 5 working days
- `eval/validate.py` — schema validator with self-test fixtures
- `eval/tests/fixtures/` — valid + invalid examples that prove the schema conditionals work

## What is intentionally not here yet

`ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, `web/` are empty. Eval comes first. The first 10 gold examples must exist and `make validate-evals` must pass before any retrieval code is written. This is non-negotiable; it is the methodological commitment of the project.

## Quickstart

```bash
uv sync
make validate-evals            # schemas + corpora + fixture self-test (CI gate)
make validate-evals-strict     # also fails on missing transcript files (commit gate)
make validate-self-test        # only run fixture self-test (fast schema iteration)
```

## License

Apache-2.0 for code. Gold set examples are factual annotations over publicly available conference talks under fair use; talk transcripts are stored under `transcripts/` and not redistributed.
