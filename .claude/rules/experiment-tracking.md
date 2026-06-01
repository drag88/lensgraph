---
description: Eval run structure, dev/test discipline, judge versioning, modality slicing, selection rule
paths: eval/**/*
---

# Experiment Tracking

The eval is the project. Every methodology decision below is load-bearing for the resume line in the README.

## Eval Run Output Structure

Every run lands in:

```
eval/reports/<YYYY-MM-DD>_<run_id>/
├── summary.json          # all metrics, all slices, one file
├── per_example.jsonl     # one row per gold example with system output
├── methodology.mdx       # human-readable writeup, linked from /docs
└── raw/                  # full traces, gitignored
```

`methodology.mdx` is the artifact that goes on the resume. It must reproduce: command run, fixture diff, `test_gold` SHA at time of run, exact model IDs from `eval/config/model_candidates.yaml`, runner-up + cost differential, modality breakdown.

## Dataset Discipline

| Set | v0 size | Purpose | Touch policy |
|---|---|---|---|
| `dev_gold.jsonl` | ~35 | Tune chunking, retrieval, generation | Touched freely |
| `test_gold.jsonl` | ~15 | Final claims only | Locked. SHA256 in README. |
| `negative.jsonl` | ~20 | Abstention signal (mostly `corpus`-scope) | Touched freely |
| `synthesis.jsonl` | ~10 | Cross-talk reasoning | Reported separately, not blended |
| `boundary_audit.jsonl` | 20–30 | Judge calibration | Touched freely |

v0 total: ~80 examples across 10–12 talks. v1 target: ~250.

**`test_gold` informs no decision.** Statements like "swapped chunker because test_gold improved" are methodologically invalid. The test set is unlocked once, for the final writeup in phase 4 week 12.

## Metrics

### Retrieval tier (chunking strategies)

- `TimestampRecall@k` for k ∈ {1, 3, 5} — does the gold span fall inside any retrieved chunk?
- `GoldSpanContained@k` — is the gold span fully contained in a single retrieved chunk? Penalizes over-segmentation.
- `IoU@1` — intersection-over-union of top chunk vs gold. Exposes chunks that are too large or off-center.
- `Latency` p50, p95 per stage (retrieve, rerank, generate).

### Boundary tier (chunk shape)

- `EdgeSensibility` (1–5) — does the clip start at a sentence boundary and end at a logical conclusion? LLM-judged.
- `Standalone` (1–5) — comprehensible without prior context? LLM-judged.
- `JudgeHumanKappa` — Cohen's kappa between LLM judge and manual audit on the boundary audit set, per chunking strategy. **Resume metric.**

### Answer tier (generation faithfulness)

- `ClaimsSupported` — fraction of `expected_claims` that the answer asserts AND that are grounded in retrieved spans. Claim-level, not blob-level.
- `CitationAccuracy` — fraction of cited timestamps that actually contain the cited claim. Catches "right answer, wrong source."
- `AnswerRelevancy` (RAGAS) — sanity check.

### Abstention tier (refusal behavior)

- `RefusalRate` on corpus-negatives — should be ≥ 0.90.
- `FalseRefusalRate` on `dev_gold` — should be ≤ 0.05.

## Modality-Sliced Reporting

Every metric is reported overall AND sliced by `modality`:

- `transcript-only` — text-only, no slides.
- `slide-required` — answer lives on a slide; ColPali should win here.
- `screen_code` — code on screen; tests OCR + visual.
- `mixed` — multimodal questions.

If `slide-required` recall does not exceed `transcript-only` baseline once ColPali is enabled, the visual retrieval is ornamental and gets cut.

## Judge Versioning

- Judge prompts live in `eval/runners/judges/` and are content-hashed.
- The hash is logged in `summary.json` and `per_example.jsonl` for every run.
- A boundary-audit run with kappa < 0.6 voids the judge for chunking comparison until the rubric is revised. **Do not publish a chunking ablation when kappa < 0.6.**
- Cross-family rule (ADR 004): the judge family differs from the generator family AND from the curation family. Anthropic curates → OpenAI judges; OpenAI curates → Anthropic judges. Same for open-weight (Gemma curates → Qwen or DeepSeek judges, etc.).

## Selection Rule (ADR 004 v3.1)

For every component (generator, judge, embeddings, reranker, ASR, cheap extraction, visual retrieval):

> Pick the cheapest candidate whose `dev_gold` eval score is within `tie_break_pp` (= 3pp) of the leader AND meets all per-component minimums.

Two consequences worth restating:

1. **No model is the project's default until the bakeoff runs.** Statements like "we use Gemma" are forbidden in any artifact dated before the bakeoff results commit.
2. **The runner-up and the cost differential are part of the writeup, not footnotes.** The interview story is "I built the selection process," not "I picked the right model."

The selection rule is encoded in `eval/config/model_candidates.yaml::selection_rule` and validated by the schema.

## Anti-Contamination Rules

1. `test_gold.jsonl` is locked. SHA256 in the README. Any change invalidates published metrics.
2. Curation tooling never sees retrieval output.
3. The chunking ablation runs all five strategies against the same set with all other variables held constant. No "we tweaked retrieval for the hybrid run."
4. Negative scope = `corpus` is the primary abstention signal. `video`-scope negatives test single-video reasoning; do not blend.
5. Same-family curation + judging is contamination. The validator does not enforce this — the discipline lives in `docs/eval-methodology.md` and ADR 004.

## What is Not Evaluated Yet (v1+)

Conversational multi-turn refinement; personalization signals; latency under load.

## Centerpiece Writeups

- `eval/reports/<date>_generator_bakeoff/methodology.mdx` — generator + judge selection (phase 2 week 6).
- `eval/reports/<date>_chunking_ablation/methodology.mdx` — five-strategy comparison (phase 2 week 8). **This is the resume link.**
