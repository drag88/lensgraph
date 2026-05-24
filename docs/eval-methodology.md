# Eval Methodology

## Stance

The eval is the project. Every other artifact exists to be evaluated by it. Build the eval before any retrieval code. This is non-negotiable.

## Datasets

| Set | Size (v0) | Purpose |
|---|---|---|
| `dev_gold.jsonl` | ~35 | Used to tune chunking, retrieval, generation. Touched freely. |
| `test_gold.jsonl` | ~15 | Locked. SHA256 in README. Used only for final claims. |
| `negative.jsonl` | ~20 | Abstention signal. Mix of `corpus`-scope (primary) and `video`-scope. |
| `synthesis.jsonl` | ~10 | Cross-talk reasoning. Reported separately, not blended. |
| `boundary_audit.jsonl` | 20–30 | Judge calibration. Manual scores + LLM judge scores, both stored. |

v0 total: ~80 examples across 10–12 talks. v1 target: ~250.

## Metrics

### Retrieval tier — judges chunking strategies

- `TimestampRecall@k` for k ∈ {1, 3, 5} — does the gold span fall inside any retrieved chunk?
- `GoldSpanContained@k` — is the gold span fully contained in a single retrieved chunk? Differs from recall in penalizing over-segmentation.
- `IoU@1` — intersection-over-union of top chunk vs gold. Exposes chunks that are too large or off-center.
- `Latency` p50, p95 per stage (retrieve, rerank, generate).

### Boundary tier — judges chunk shape

- `EdgeSensibility` (1–5) — does the clip start at a sentence boundary and end at a logical conclusion? LLM-judged.
- `Standalone` (1–5) — comprehensible without prior context? LLM-judged.
- `JudgeHumanKappa` — Cohen's kappa between LLM judge and manual audit on the boundary audit set, reported per chunking strategy. **This metric goes on the resume.**

### Answer tier — judges generation faithfulness

- `ClaimsSupported` — fraction of `expected_claims` that the answer asserts AND that are grounded in retrieved spans. Claim-level, not blob-level.
- `CitationAccuracy` — fraction of cited timestamps that actually contain the cited claim. Catches "right answer, wrong source" failures.
- `AnswerRelevancy` (RAGAS) — sanity-check tier.

### Abstention tier — judges refusal behavior

- `RefusalRate on corpus-negatives` — should be ≥0.90.
- `FalseRefusalRate on dev_gold` — should be ≤0.05.

## Modality-sliced reporting

Every metric is reported overall AND sliced by `modality`:

- `transcript-only` — text-only, no slides referenced.
- `slide-required` — answer lives on a slide; ColPali should win here.
- `screen_code` — code on screen; tests OCR + visual.
- `mixed` — multimodal questions.

If `slide-required` recall does not exceed `transcript-only` baseline once ColPali is enabled, the visual retrieval is ornamental and gets cut.

## Judge discipline

- **Cross-family.** The judge must be in a different model family from the generator. See ADR 004 v3 for the candidate set and the empirical basis (ICLR 2026 "Preference Leakage" paper).
- **Versioned prompts.** Every judge prompt lives in `eval/runners/judges/` and is hashed; the hash is logged with every run.
- **Human override.** Boundary judge requires a 20–30 example manual audit. If kappa < 0.6, the LLM judge is not trustworthy and needs a rubric revision before any chunking comparison is published.

## Selection rule

Per ADR 004 v3, model selection for every component obeys one rule:

> Pick the cheapest candidate whose eval score is within 3pp of the leader AND meets all per-component minimums.

This is what the bakeoff in roadmap phase 2 produces. Two consequences worth stating:

1. **No model is the project's "default" until the bakeoff runs.** Statements like "we use Gemma" are forbidden in any artifact dated before the bakeoff results commit.
2. **The runner-up and the cost differential are part of the writeup, not footnotes.** The interview story is "I built the selection process," not "I picked the right model."

## Anti-contamination rules

1. `test_gold.jsonl` is locked. A SHA256 hash is in the README. Any change invalidates published metrics.
2. Curation tooling never sees retrieval output.
3. The chunking ablation runs all five strategies against the same set with all other variables held constant. No "we tweaked retrieval for the hybrid run" cheating.
4. Negative scope = `corpus` is the primary abstention signal. `video`-scope negatives test single-video reasoning; do not blend.

## Public artifacts

Every eval run produces:

```
eval/reports/<YYYY-MM-DD>_<run_id>/
├── summary.json          # all metrics, all slices
├── per_example.jsonl     # one row per gold example with system output
├── methodology.mdx       # human-readable writeup, linked from /docs
└── raw/                  # full traces, gitignored
```

The chunking ablation centerpiece is `eval/reports/<date>_chunking_ablation/methodology.mdx`. This is the link on the resume.

## What is not evaluated yet (v1+)

- Conversational multi-turn refinement.
- Personalization signals.
- Latency under load.
