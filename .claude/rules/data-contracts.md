---
description: JSON Schema source-of-truth — talk, gold_example, boundary_audit, model_candidates
paths: eval/**/*
---

# Data Contracts

The four JSON Schemas in `eval/schemas/` are the source of truth for every persisted shape in the system. Code conforms to schemas; schemas do not conform to code.

All schemas use Draft 2020-12, are checked by `Draft202012Validator.check_schema(...)` at load time, and run with a `FormatChecker` attached so `date-time` and `uri` formats actually validate (requires `jsonschema[format-nongpl]`).

## `talk.schema.json` — `eval/corpora/<corpus>/talks.yaml` entries

Owns transcript provenance. Examples reference talks by `video_id` only.

**Required**: `video_id`, `title`, `speaker`, `url`, `duration_sec`, `format_tags`, `license`, `captions_source`, `transcript_path`, `transcript_sha256`, `accessed_at`.

Key fields:

- `video_id` — YouTube ID for direct videos, or a stable pseudo-ID for chapter-sliced talks.
- `captions_source` — one of `youtube_auto`, `youtube_manual`, `whisperx_large_v3`, `manual_transcript`.
- `transcript_path` — relative path from project root (e.g. `transcripts/zjkBMFhNj_g.vtt`). Validated for existence in strict mode.
- `transcript_sha256` — 64-char lowercase hex. Validator re-hashes the file and fails on drift.
- `format_tags` — enum array; at least one of `slides_heavy`, `code_heavy`, `whiteboard`, `conversational`, `narrative`, `panel`, `lightning`, `live_demo`.
- `accessed_at` — RFC3339 timestamp.

Chapter-sliced talks set `source_video_id`, `source_start_sec`, `source_end_sec` together (dependentRequired). The Python validator also checks `duration_sec ≈ source_end_sec - source_start_sec` (±1s).

## `gold_example.schema.json` — `dev_gold.jsonl`, `test_gold.jsonl`, `negative.jsonl`, `synthesis.jsonl`

Three question shapes, enforced via `allOf` / `if-then` conditionals:

| `question_type` | top-level `video_id` | `gold_spans` | `expected_claims` | `abstention_expected` |
|---|---|---|---|---|
| `single_clip` | required | exactly 1 (video_id optional in span) | required | must be `false` |
| `synthesis` | forbidden (`not.required`) | `minItems: 2`, each span requires `video_id` | required | must be `false` |
| `negative` + `negative_scope: video` | required | `maxItems: 0` | not required | must be `true` |
| `negative` + `negative_scope: corpus` | forbidden | `maxItems: 0` | not required | must be `true` |

Other required top-level fields: `id` (kebab/snake), `question` (≥15 chars), `split` (`dev`|`test`), `modality` (array, ≥1 of `transcript`/`slide`/`screen_code`/`diagram`/`whiteboard`/`audio_only`), `difficulty` (`easy`|`medium`|`hard`), `verified` (must be `true` for committed examples), `curator`, `curated_at` (RFC3339).

**Cross-field semantic checks (Python, after schema):**

- `end_sec > start_sec` for every `gold_spans` entry.
- Every referenced `video_id` must exist in the corpus's `talks.yaml` — no bypass when `talks.yaml` is empty.
- `verified: true` is mandatory for any example in a committed corpus file (`*_gold.jsonl`, `negative.jsonl`, `synthesis.jsonl`).

**Phase-0 gate semantics:**

`count_verified_gold_and_talks()` counts only `single_clip + synthesis` (not negatives). Rationale: negatives carry no retrieval signal, so 10 negatives could trivially unlock implementation work that has no signal-bearing examples to validate against.

## `boundary_audit.schema.json` — `boundary_audit.jsonl`

Pairs LLM-judge scores with manual scores for the same system-produced clip. Used to compute Cohen's kappa per chunking strategy.

**Required**: `example_id` (FK into gold), `chunking_strategy` (`fixed_window`|`transcript_segment`|`slide_boundary`|`topic_llm`|`hybrid`), `system_clip` (`start_sec`, `end_sec`, optional `video_id`), `judge_score` and `human_score` (both contain `edge_sensibility` 1-5 + `standalone` 1-5; judge may add `rationale`, human may add `notes`), `auditor`, `audited_at`.

Same `end_sec > start_sec` check applies to `system_clip` in Python after schema.

## `model_candidates.schema.json` — `eval/config/model_candidates.yaml`

Source of truth for the phase-2 bakeoff runner. Permissive on per-option shape (text-embedding vs generator pricing differs) but strict on what catches real bugs.

**Required top-level**: `last_updated` (date), `prices_checked_at` (date), `sources` (URI map, ≥1), `inference_providers` (with `primary`, optional `fallback_router`, optional `excluded` list with `provider` + reason ≥10 chars), `selection_rule` (`tie_break_pp` 0-100, `dev_set`, `test_set_for_writeup_only`), `candidates` (component groups, ≥1).

Each candidate option requires `id` (lowercase + dot/dash/underscore), `provider` (enum: `deepinfra`/`openrouter`/`anthropic`/`openai`/`voyage`/`google`/`local`), `provider_model_id`, `family`. Optional pricing fields: `price_per_mtok_usd`, `price_input_per_mtok_usd`, `price_output_per_mtok_usd` (all ≥0 or null).

**Cross-check (Python, after schema):**

`inference_providers.primary` must appear as a `provider` on at least one candidate option. Catches the bug where the YAML claims DeepInfra is primary but every option lists OpenRouter.

## Fixture Self-Test

`eval/tests/fixtures/` mirrors the schemas. Naming is enforced:

- `valid_*.json` — validator must accept. One per conditional branch that should succeed.
- `invalid_*.json` — validator must reject. One per rule the schema is supposed to enforce.
- Any other filename fails the self-test.

When you add a schema conditional, you add a fixture pair. The self-test asserts the rule actually fires (`invalid_*.json` must be rejected for the right reason — silent acceptance is a self-test failure).

## What JSON Schema Cannot Express (Lives in Python)

- Cross-field arithmetic (`end_sec > start_sec`, `duration_sec ≈ slice length`).
- `format` keyword activation (requires `FormatChecker` instance — without it, `format` is annotation-only).
- File-system existence (`transcript_path` on disk, strict mode only).
- Hash drift (`transcript_sha256` matches actual file contents).
- Cross-file references (`video_id` exists in `talks.yaml`).
- Cross-section consistency (`inference_providers.primary` is used by ≥1 candidate option).
- Committed-corpus semantics (`verified: true` requirement).

Add new cross-field checks to `eval/validate.py` near the existing ones, not in a separate file.
