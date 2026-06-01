---
description: Code patterns and conventions that apply across the project
---

# Code Patterns

## Schema-First Authoring

Order of operations whenever a persisted field changes:

1. Update the relevant schema in `eval/schemas/`.
2. Add a `valid_*.json` fixture exercising the new field.
3. Add an `invalid_*.json` fixture exercising the failure case (missing field, wrong type, conditional violated).
4. Run `make validate-self-test`. Confirm the new invalid fixture fails for the right reason.
5. Update consuming code.
6. Update any corpora that need backfilled values.

Skipping the fixture step means the schema rule is unproven. The validator self-test gates this on every CI run.

## Validator Structure

`eval/validate.py` is a single-file entry point with three modes invoked through `make`:

- `validate-evals` — schemas + corpora cross-refs + fixture self-test. Warns on missing transcripts. CI gate.
- `validate-evals-strict` — same, but missing transcripts become errors. Commit gate.
- `validate-self-test` — only the fixture self-test. Fast iteration during schema work.
- `phase0-gate` — strict + `--min-gold 10 --min-talks 2`. Required before touching `ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, `web/`.

Cross-field semantic checks that JSON Schema cannot express (e.g. `end_sec > start_sec`, span video_id existence in `talks.yaml`, transcript SHA drift, `verified: true` on committed examples, `model_candidates.yaml` `inference_providers.primary` is actually used by a candidate) live in Python after schema validation.

## Talks Own Provenance

The single non-negotiable shape rule. Talks own `captions_source`, `transcript_path`, `transcript_sha256`, `accessed_at`. Examples carry only `video_id`. Synthesis examples carry no top-level `video_id`; their spans each carry their own.

The shape table:

| `question_type` | top-level `video_id` | `gold_spans` | `expected_claims` | `abstention_expected` |
|---|---|---|---|---|
| `single_clip` | required | exactly 1 | required | must be `false` |
| `synthesis` | forbidden | ≥2, each with `video_id` | required | must be `false` |
| `negative` + `scope=video` | required | empty | not required | must be `true` |
| `negative` + `scope=corpus` | forbidden | empty | not required | must be `true` |

Enforced by JSON Schema conditionals in `eval/schemas/gold_example.schema.json`. Negatives carry no signal-bearing answer, so they do not contribute to the phase-0 gate count.

## Verified Flag Discipline

Every committed gold example has `verified: true`. The validator rejects anything else in committed corpora files.

Workflow:

1. Draft into `dev_gold.draft.jsonl` (gitignored) with `verified: false`.
2. Open the source video at the proposed span. Watch the clip. Confirm the claims hold.
3. Flip `verified: true`.
4. Move the line into the appropriate committed file (`dev_gold.jsonl`, `negative.jsonl`, `synthesis.jsonl`).
5. Run `make validate-evals-strict` before commit.

Test set is locked. Never modify `test_gold.jsonl` during development. Its SHA256 is recorded in the project README and is a contract.

## Fixture Naming

`eval/tests/fixtures/`:

- `valid_*.json` — the validator must accept it. One file per conditional branch that should succeed.
- `invalid_*.json` — the validator must reject it. One file per rule the schema is supposed to enforce.
- Any other name fails the self-test by design.

Add a fixture pair (valid + invalid) for every schema conditional, not one for the whole schema.

## Conventional Commits

Format: `<type>: <subject>`. Types:

- `feat:` — new behaviour
- `fix:` — bug fix
- `refactor:` — non-behavioural code change
- `docs:` — documentation, ADRs, README
- `eval:` — anything inside `eval/` (schemas, corpora, validator, fixtures, runners, reports)
- `chore:` — tooling, CI, build config

No AI attribution lines. No "Generated with Claude Code" footers. Atomic commits — one logical change each.

## Architecture Decisions

Format every ADR as `docs/decisions/NNN-slug.md` with sections: Context, Options, Decision, Consequences, Revisit when. Status on top: `Status: Accepted | Superseded by NNN | Rescinded`. Date in ISO format.

Existing ADRs:
- 001 — LangGraph, not LangChain or LlamaIndex
- 002 — Postgres as the only datastore
- 003 — Conference talks as the launch corpus
- 004 — Model selection (candidate set + selection rule; not a single-model default)

Any change that contradicts an ADR requires either an amendment to the existing ADR (`v2`, `v3`, etc., dated) or a superseding ADR.

## Ruff Configuration

Line length 100; formatter handles wrapping (`ignore = ["E501"]`). Selected lint rules: E (pycodestyle errors), F (pyflakes), I (import order), B (bugbear), UP (pyupgrade), N (pep8-naming). Double quotes. Run `make fmt` then `make lint` before commit.

## Plain-Python Bias

The validator is one file, no plugin architecture, no class hierarchy. Bias toward the same shape for the rest of the eval harness: one runner per concern, functions over classes, JSON Schema + plain dicts over dataclasses.

When a class genuinely earns its keep (stateful judges, long-lived embedding pipelines), introduce it. Until then, prefer functions.

## What Not to Do

- Do not add per-example provenance fields. They go on the talk.
- Do not flip `verified: true` on an example without watching the clip.
- Do not modify `test_gold.jsonl` during development. Treat it as immutable.
- Do not introduce a second datastore, second agent framework, or second inference provider without an ADR amendment.
- Do not skip the fixture self-test when changing a schema. A schema rule without a fixture is unproven.
- Do not write code in `ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, `web/` until `make phase0-gate` passes.
