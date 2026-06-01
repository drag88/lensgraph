# LensGraph - Agent Instructions

## Role and Project

You are helping build LensGraph, a Python project for multimodal video RAG over
engineering conference talks.

The important output is not a flashy demo. The important output is a system that
can prove, with repeatable evals, which retrieval, chunking, model, and visual
settings work.

Read these first when context is limited:

1. `README.md`
2. `dev/active/phase-2-week-6/handoff.md`
3. `docs/architecture.md`
4. `docs/eval-methodology.md`
5. `docs/decisions/*.md`
6. `eval/README.md`

## Quick Start

```bash
uv sync
make validate-evals
make validate-evals-strict
make phase0-gate
make test
make fmt
make lint
```

Use `make help` for the full target list.

## Current Shape

- `eval/` is the center of the project: schemas, corpora, runners, tests, reports.
- `db/` owns Postgres connection setup, raw SQL migrations, and repository helpers.
- `ingest/` fetches transcripts, checks quality, writes talks rows, and enqueues
  downstream work through PGMQ.
- `chunking/` contains transcript chunking strategies.
- `retrieve/` contains BM25, BGE-M3 dense/sparse/multi-vector retrieval, visual
  retrieval, RRF fusion, and reranking.
- `generate/` contains the LangGraph answer loop: Plan -> Retrieve -> Rerank ->
  Verify -> Generate -> Cite.
- `queues/` owns Postgres-backed queue helpers and workers.
- `scripts/` contains operational runners such as video fetch and eval prep.

## Hard Rules

1. **Keep evals ahead of claims.** Before committing changes under `ingest/`,
   `chunking/`, `retrieve/`, `generate/`, `api/`, or `web/`, run
   `make phase0-gate`. If it fails, fix that before continuing.

2. **Schema before consuming code.** If a persisted shape changes, update
   `eval/schemas/`, add a matching `valid_*.json` and `invalid_*.json` fixture
   under `eval/tests/fixtures/`, then run `make validate-self-test`.

3. **Do not use `test_gold` for development decisions.** Treat
   `eval/corpora/*/test_gold.jsonl` as locked. Use `dev_gold` for iteration.
   If `test_gold` must change, document it as a methodology reset and rerun any
   metrics that depended on it.

4. **Only mark examples verified after watching the clip.** Do not set
   `verified: true` unless the actual source clip supports the span and claims.

5. **Put transcript provenance on talks, not examples.** `transcript_sha256`,
   `captions_source`, `transcript_path`, and `accessed_at` belong in
   `eval/corpora/<corpus>/talks.yaml`. Examples reference talks by `video_id`.

6. **Keep curation and judging separate.** The model family that drafts or edits
   examples must not be the same family that judges those examples.

7. **Postgres is the only datastore.** Use Postgres, pgvector, and PGMQ. Do not
   add Redis, Celery, RabbitMQ, a second vector DB, or another backing store
   without updating the decision docs.

8. **LangGraph is the answer-loop framework.** Do not introduce LlamaIndex,
   LangChain agent chains, or a second graph framework without updating the
   decision docs.

9. **No silent model defaults.** Candidate config lives in
   `eval/config/model_candidates.yaml`. Selection uses `dev_gold`, picks the
   cheapest candidate within 3 percentage points of the leader that also meets
   the required minimums, and keeps `test_gold` for final reporting only.

10. **Validate eval changes locally.** Run `make validate-evals-strict` before
    committing changes under `eval/`.

## Common Workflows

### Change a Schema

1. Edit the schema in `eval/schemas/`.
2. Add one valid fixture and one invalid fixture in `eval/tests/fixtures/`.
3. Run `make validate-self-test`.
4. Update consuming code.
5. Run the narrow tests that cover the consumer, then `make validate-evals-strict`.

### Add or Edit a Gold Example

1. Read `eval/curation/playbook.md`.
2. Draft in a gitignored draft file with `verified: false`.
3. Watch the source clip at the proposed timestamp.
4. Move the row into the committed corpus file and set `verified: true`.
5. Run `make validate-evals-strict`.

### Add a Chunking Strategy

1. Add `chunking/<strategy_name>.py`.
2. Keep the public entry point simple: `chunk(transcript, frames) -> list[Chunk]`.
3. Register it in `chunking/__init__.py`.
4. Compare it under the same retrieval and generation settings as the baseline.
5. Run `make validate-evals` and the relevant eval runner before claiming it is
   better.

### Run the Answer Loop

Pre-selection runs must name generator and judge candidates explicitly:

```bash
make answer QUERY="..." GENERATOR=<generator_id> JUDGE=<judge_id>
```

The CLI reads `QUERY`, `GENERATOR`, `JUDGE`, `PLANNER`, and `CORPUS_ID` from the
environment. The judge and generator must be from different model families.

### Touch the Database

1. Add raw SQL in `db/migrations/NNNN_slug.sql`.
2. Keep repository logic under `db/repos/`.
3. Run the relevant DB tests. If a live DB is needed, use the Makefile targets:
   `make db-up`, `make db-verify-ext`, `make db-migrate`.

`POSTGRES_DSN` overrides the local default:
`postgresql://lensgraph:lensgraph@localhost:5432/lensgraph`.

## Plain Developer Docs

Write docs for the next developer, not for a process audit.

- Lead with what changed, what is true now, what is broken, and what to do next.
- Keep opening summaries short: at most 5 bullets or 8 lines.
- Use plain repo language. Prefer "comparison" over "bakeoff" unless naming an
  existing command, file, or commit. Prefer "decision doc" over "ADR" unless
  pointing at `docs/decisions/*`.
- Avoid roleplay and process labels. Do not write "mission", "agent-team",
  "workstream", "slice", "actionability boundary", or similar framing when
  "task", "owner", "next step", or "stop condition" is clearer.
- Avoid dramatic wording. Do not write lines like "paid for in blood",
  "do not re-litigate", or "non-negotiable" unless quoting an external source.
- Use concrete nouns. Replace vague words like "substrate", "surface",
  "artifact", and "invariant" with the actual thing: frames, DB rows, report,
  rule, metric, or command.
- Do not repeat the same rule in multiple sections. State it once and reference
  the file or command when needed.
- Prefer short sentences. Split any sentence with more than two clauses or
  several parentheticals.
- Use tables for data, not for hiding prose decisions.
- No AI attribution in committed docs or commits.

Every handoff should answer these questions:

1. What is the current state?
2. What commands were run?
3. What failed or was not run?
4. What is the next command?
5. What files are likely to change next?

## Coding Conventions

- Python 3.11+.
- Use `uv` for environment and command execution.
- Use Ruff for format and lint. Line length is 100; formatter owns wrapping.
- Pytest defaults to `eval/tests` and excludes `slow` tests unless explicitly
  selected.
- Prefer functions and small modules until a class clearly reduces complexity.
- Keep comments for non-obvious why. Do not comment what the next line does.
- Do not add speculative fallbacks or compatibility shims for code paths that do
  not exist yet.
- Do not skip pre-commit or pre-push checks.
- Do not run destructive git commands without explicit confirmation.

## Commit Style

Use conventional commits:

- `feat:` for new behavior
- `fix:` for bug fixes
- `refactor:` for behavior-preserving code changes
- `docs:` for documentation
- `eval:` for anything under `eval/`
- `chore:` for tooling, build, or repository maintenance

Keep commits atomic. Do not add "Generated with Claude Code" or similar footers.

## Tools and Agents

- Read code before editing.
- Use `rg` before slower search tools.
- Run independent reads or checks in parallel when it saves time.
- Use subagents for independent multi-file investigation, not for simple edits.
- Keep final edits in the main context unless the user explicitly asks for
  delegated implementation.
- Verify before saying work is done. Prefer concrete commands and report what
  passed, failed, or was not run.

## Key Paths

- `eval/schemas/` - JSON Schemas.
- `eval/corpora/<corpus>/` - `talks.yaml` plus gold, negative, synthesis, and
  boundary-audit JSONL files.
- `eval/tests/fixtures/` - validator self-test fixtures.
- `eval/config/model_candidates.yaml` - model/provider candidate config.
- `eval/validate.py` - schema, corpus, transcript, fixture, and phase-0 checks.
- `eval/runners/` - eval runners and provider helpers.
- `eval/reports/` - run outputs and methodology writeups.
- `db/migrations/` - raw SQL migrations.
- `docs/decisions/` - decision docs for architecture and methodology choices.
- `dev/active/` - current handoffs.
- `transcripts/` and `videos/` - local runtime data; do not assume they exist in
  CI.

## CI

GitHub Actions runs `uv sync --frozen`, `make validate-evals`, and `make lint`.
It does not run `make test` or `make phase0-gate`; run those locally when the
change needs them.
