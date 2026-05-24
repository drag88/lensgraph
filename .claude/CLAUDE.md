# LensGraph — Claude Code Instructions

## What this project is

Multimodal video RAG over engineering conference talks. Eval-first; the eval harness is the product. See `README.md`, `docs/PRD.md`, `docs/architecture.md`, `docs/eval-methodology.md`, `docs/roadmap.md`.

## Hard rules

1. **Eval before code.** Do not write or modify `ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, or `web/` until `make phase0-gate` passes (≥10 verified examples + strict transcript validation). The gate is enforceable: it is `uv run python eval/validate.py --strict --min-verified 10`. Run it locally before any commit touching those directories.

2. **The test set is locked.** Never modify `eval/corpora/*/test_gold.jsonl` during development. Its SHA256 in the README is a contract. If a change is genuinely required, document it as a methodology revision and reset all downstream metrics.

3. **Provenance lives on talks, not examples.** `transcript_sha256`, `captions_source`, `transcript_path`, `accessed_at` belong in `talks.yaml`. Do not add per-example provenance fields.

4. **Cross-family judge discipline.** If curation candidates were drafted with Claude, the LLM judge must be OpenAI-class; and vice versa. Same-model curation + judging is contamination.

5. **One datastore.** Postgres only. Do not introduce Redis, Celery, RabbitMQ, or any second backing store. PGMQ handles queues; pgvector handles embeddings. See `docs/decisions/002-postgres-only.md`.

6. **One agent framework.** LangGraph only. No LangChain core, no LlamaIndex. See `docs/decisions/001-langgraph-not-llamaindex.md`.

6a. **Model selection follows ADR 004 v3.1: candidate set + selection rule, not single-model defaults.** Structured candidate config lives at `eval/config/model_candidates.yaml` — that file is the source of truth for the bakeoff runner. ADR prose is for humans.
   - Generator candidates: gemma-4-31b · qwen3-235b-a22b-instruct · deepseek-v3.2 (all DeepInfra).
   - Judge candidates: cross-family from chosen generator (gemma / qwen / deepseek).
   - Cheap extraction: gemma-4-e4b · qwen3-8b local first; fall back to chosen hosted generator.
   - Text embeddings: **BAAI/bge-m3 all three channels** (dense in pgvector, sparse in pgvector sparsevec, multi-vector via per-token arrays + MaxSim). Voyage / Gemini Embedding 2 only as fallbacks if BGE-M3 fails the embeddings minimum.
   - Visual retrieval: ColQwen2.5 via colpali-engine (local/Modal); Gemini Embedding 2 as single-vector baseline for the writeup only.
   - Reranker: BAAI/bge-reranker-v2-m3 local CPU.
   - ASR: WhisperX large-v3 local MPS.
   - Premium triangulation: claude-sonnet-4-6 + gpt-5.5, one-off on test_gold.

   **Inference provider:** DeepInfra is primary (only host with the full candidate set). OpenRouter is optional failover. Groq is excluded until they add Gemma 4 31B / DeepSeek V3.2 / Qwen3-235B.

   **Selection rule:** within each component, pick the cheapest candidate whose `dev_gold` eval score is within 3pp of the leader AND meets all minimums (defined in ADR 004). `test_gold` is reserved for the final writeup — never for selection. Do not lock a default before running the phase-2 bakeoff. Do not introduce a new model or provider without an ADR amendment.

7. **`verified: true` is sacred.** Never flip `verified: true` on an example without watching the actual clip. The validator gates committed corpora on this field.

8. **Run `make validate-evals-strict` before any commit that touches `eval/`.** The default `validate-evals` is the CI gate and is lenient on missing transcript files (transcripts are gitignored). The `-strict` target fails on missing transcripts and is the local commit gate. CI cannot enforce this — discipline is on you.

## Conventions

- Python ≥ 3.11, `uv` for env, `ruff` for format + lint, line length 100.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `eval:`, `chore:`.
- New eval examples land in `dev_gold.draft.jsonl` first (gitignored), get verified, then promoted to the appropriate committed file.
- Architecture decisions go in `docs/decisions/NNN-slug.md`. Format: Context, Options, Decision, Consequences, Revisit when.

## When asked to add a new chunking strategy

1. Implement it in `chunking/<strategy_name>.py` with a single `chunk(transcript, frames) -> list[Chunk]` entry point.
2. Register it in `chunking/__init__.py` so `eval/runners/run_eval.py` can pick it via `--chunking <name>`.
3. Run `make validate-evals` and a baseline eval run before claiming any improvement.
4. Never compare strategies under different retrieval/generation settings — only chunking varies.

## When asked to add a new gold example

1. Open `eval/curation/playbook.md`. Follow it.
2. Watch the clip. Always.
3. Write `verified: false` first, verify, then flip to `true`.
4. Run `make validate-evals` before committing.

## What not to do

- Do not add features beyond what an explicit task requires.
- Do not add error handling, fallbacks, or backwards-compat shims for code that does not exist yet.
- Do not write comments that restate what the code does. Comments are for non-obvious why.
- Do not create new markdown documents unless explicitly asked.
- Do not skip pre-commit hooks or pre-push checks.
- Do not run destructive git commands without confirmation.

## Background reading priority

If context is limited, load in this order:
1. `README.md`
2. `docs/PRD.md`
3. `docs/eval-methodology.md`
4. `docs/architecture.md`
5. `docs/decisions/*.md`
6. `eval/README.md`
