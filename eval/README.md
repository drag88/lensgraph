# eval/

The eval is the project. Everything in `eval/` is treated as production code, not test scaffolding.

## Layout

```
eval/
├── README.md                              # you are here
├── schemas/
│   ├── talk.schema.json                   # talks.yaml entries
│   ├── gold_example.schema.json           # gold/negative/synthesis examples
│   └── boundary_audit.schema.json         # judge calibration records
├── corpora/
│   └── ai_engineering_v0/
│       ├── talks.yaml                     # source/video provenance
│       ├── dev_gold.jsonl                 # tuneable
│       ├── test_gold.jsonl                # locked (SHA256 in README)
│       ├── negative.jsonl                 # abstention examples
│       ├── synthesis.jsonl                # multi-video examples
│       └── boundary_audit.jsonl           # judge calibration
├── curation/
│   └── playbook.md                        # how to build the gold set
├── runners/
│   ├── run_eval.py                        # (v1)
│   └── judges/                            # (v1)
├── reports/                               # (versioned eval runs)
├── tests/
│   └── fixtures/                          # valid_*.json must pass; invalid_*.json must fail
└── validate.py                            # schema validator + self-test
```

## Schema model

**Provenance is owned by talks, not by examples.** A talk record stores `transcript_path`, `transcript_sha256`, `captions_source`, and `accessed_at`. Examples reference talks by `video_id` only. This makes synthesis examples (which span multiple videos) unambiguous.

**Gold examples take three shapes**, enforced by JSON Schema conditionals:

| `question_type` | `video_id` (top-level) | `gold_spans` | `expected_claims` | `abstention_expected` |
|---|---|---|---|---|
| `single_clip` | **required** | exactly 1, video_id optional in span | **required** | must be false |
| `synthesis` | **forbidden** | ≥2, video_id **required per span** | **required** | must be false |
| `negative` + `scope=video` | **required** | 0 (none allowed) | not required | must be true |
| `negative` + `scope=corpus` | **forbidden** | 0 (none allowed) | not required | must be true |

`expected_claims` is an array of atomic statements. The faithfulness judge scores each independently; this is dramatically more reliable than judging a free-form answer as one blob.

## Running validation

```bash
make validate-evals           # corpora + cross-refs + fixture self-test (CI gate)
make validate-evals-strict    # also fails on missing transcript files (commit gate)
make validate-self-test       # only fixture self-test (fast schema iteration)
```

`validate-evals` runs both corpora validation AND the fixture self-test by default — a single command covers both data drift and schema regressions.

The self-test is the "is the schema doing what I claim it is" check. Every conditional rule (e.g. "synthesis must have per-span video_id", "end_sec must exceed start_sec", "curated_at must be RFC3339") has a corresponding `invalid_*.json` fixture that the validator must reject.

### What is enforced beyond JSON Schema

JSON Schema 2020-12 cannot express cross-field constraints (e.g. `end_sec > start_sec`) or activate `format` keywords automatically. The validator wires both:

- `FormatChecker` is attached so `date-time` and `uri` formats are actually validated (requires `jsonschema[format-nongpl]` extra, declared in `pyproject.toml`).
- Span ranges (`end_sec > start_sec`) are checked in Python after schema validation, for both `gold_spans` and `boundary_audit.system_clip`.
- `verified: true` is required for any example committed to a corpus file.
- `video_id` references in examples must exist in the corpus's `talks.yaml` — no exception when `talks.yaml` is empty.
- In `--strict` mode, missing transcript files on disk fail (not warn).

## Adding examples

1. Read `curation/playbook.md`.
2. Draft examples with `verified: false`.
3. Verify by hand against the actual clip.
4. Flip `verified: true` and commit. The validator rejects unverified examples in committed corpus files.

## The locked test set

`test_gold.jsonl` is locked. Its SHA256 is recorded in the project README. Touching it during development invalidates any published metric that referenced the prior hash. Treat it as a contract with future-you.
