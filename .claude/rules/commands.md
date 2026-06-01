---
description: Make targets, validator invocations, and pre-commit / pre-push expectations
---

# Commands

## Setup

```bash
uv sync                                 # install (Makefile target: make install)
```

## Validation

```bash
make validate-evals                     # CI gate: schemas + corpora + fixture self-test (warns on missing transcripts)
make validate-evals-strict              # commit gate: same, but fails on missing transcript files
make validate-self-test                 # fast iteration: only fixture self-test
make phase0-gate                        # gate before touching ingest/chunking/retrieve/generate/api/web
                                        # = validate-evals-strict + --min-gold 10 --min-talks 2
```

Under the hood every target invokes `uv run python eval/validate.py` with different flags. See `eval/validate.py --help` for the full flag surface.

## Lint and Format

```bash
make fmt                                # ruff format .
make lint                               # ruff check .
```

CI runs `make lint` after validation. Run `make fmt` before commit if you have not configured a format-on-save hook.

## Pre-commit Expectations

Before any commit that touches `eval/`:

```bash
make validate-evals-strict
make lint
```

Before any commit that touches `ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, `web/`:

```bash
make phase0-gate
make lint
```

The phase-0 gate fails until ≥10 verified non-negative gold examples exist across ≥2 distinct talks AND strict transcript validation passes. CI cannot enforce this — discipline lives in CLAUDE.md hard rule 1.

## Adding a Gold Example

```bash
# 1. Read the playbook (~5 min)
$EDITOR eval/curation/playbook.md

# 2. Draft into the gitignored draft file
$EDITOR eval/corpora/ai_engineering_v0/dev_gold.draft.jsonl

# 3. Watch the actual clip at the proposed span. Confirm the claims.

# 4. Flip verified:true and promote the line to the committed file.

# 5. Validate locally before commit.
make validate-evals-strict
```

## Adding a Schema Conditional

```bash
# 1. Edit eval/schemas/<file>.schema.json — add the conditional.
# 2. Add a valid fixture (this should pass).
$EDITOR eval/tests/fixtures/valid_<new_branch>.json
# 3. Add an invalid fixture (this should fail for the right reason).
$EDITOR eval/tests/fixtures/invalid_<rule_violated>.json
# 4. Run the self-test only — fast iteration.
make validate-self-test
```

## Running the Validator Directly

```bash
uv run python eval/validate.py                                   # default = validate-evals
uv run python eval/validate.py --self-test                       # fixtures only
uv run python eval/validate.py --strict                          # commit gate
uv run python eval/validate.py --strict --min-gold 10 --min-talks 2   # phase0-gate
```

## Clean

```bash
make clean                              # remove __pycache__ and .ruff_cache
```

## CI Workflow

`.github/workflows/ci.yml` on every push/PR to `main`:

1. `astral-sh/setup-uv@v6` with cache
2. `uv sync --frozen`
3. `make validate-evals`
4. `make lint`

If validation or lint fails, the PR cannot merge. The phase-0 gate is intentionally absent from CI.
