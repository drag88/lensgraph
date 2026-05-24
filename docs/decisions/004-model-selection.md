# ADR 004 — Model selection: candidate set + eval-driven selection rule

**Status:** Accepted (v3, 2026-05-24)
**Supersedes:** v1 (closed-API default) and v2 (Gemma-as-default). See Changelog.

## Context

A project whose thesis is "the eval is the product" must not pick its production models by vibe. Declaring a winner before the eval exists is the same failure mode the rest of the codebase is built to prevent.

Two facts force this rewrite of v2:

1. **The model field is moving weekly.** Gemma 4 dropped 7 weeks ago. Qwen3-235B-A22B-Instruct is listed on DeepInfra at ~$0.07/$0.10 per MTok — roughly half the price of Gemma 4 31B. DeepSeek V3.2 is in the same competitive band. Any single-model default is stale by week 6.
2. **The right answer is the cheapest model that clears the eval.** Not the most capable. Not the most popular. The cheapest one that meets the published thresholds on the locked test set.

So v3 changes the shape of the decision: **a candidate set per component, plus a formal selection rule that operates on eval output.**

## The selection rule

```
For component K with candidate set C_K and price() function:

  scores = { c: eval_score(c) for c in C_K }
  best   = max(scores.values())
  qualifying = { c for c in C_K
                 if scores[c] >= best - 0.03
                 and meets_minimums(c, K) }
  chosen = argmin(price(c) for c in qualifying)
```

**Tie-break tolerance: 3 percentage points** below the leader counts as a tie. Within the tie, cheapest wins.

**Per-component minimums:**

| Role | Minimum gates (all must hold) |
|---|---|
| Generator | ClaimsSupported ≥ 0.85, CitationAccuracy ≥ 0.85, JSON-parse success ≥ 0.98, corpus-negative RefusalRate ≥ 0.90, p95 generation latency ≤ 2.0s |
| Judge | Cohen's kappa vs human boundary audit ≥ 0.60; must be in a different model family from the generator |
| Cheap extraction | JSON-parse success ≥ 0.98 on chunker output; topic boundaries IoU ≥ 0.70 vs hand-segmented set |
| Embeddings | TimestampRecall@5 on dev_gold ≥ 0.75 (vector channel only) |
| Reranker | Top-3 reorder improves Recall@3 by ≥ 5pp over no-rerank baseline |

A candidate that fails any minimum is disqualified regardless of score.

## Candidate set per component

| Role | Candidate set | Why these |
|---|---|---|
| **Planner / generator** | `gemma-4-31b` (DeepInfra) · `qwen3-235b-a22b-instruct` (DeepInfra) · `deepseek-v3.2` (DeepInfra) | Top three open-weight contenders for agentic tool use as of May 2026. All have native function calling and structured JSON. Wide price spread (5×) makes the selection rule meaningful. |
| **Cheap extraction** | `gemma-4-e4b` (local MPS) · `qwen3-8b` (local MPS) · DeepInfra fallback to whatever-won-generator | Local first to keep cost at zero. Falls back to chosen hosted generator if local can't hit JSON-parse minimum. |
| **LLM judge** | `deepseek-v3.2` (DeepInfra) · `qwen3-235b-a22b-instruct` (DeepInfra) · spot-check passes via `claude-sonnet-4-6` and `gpt-5.5` | Must be cross-family from chosen generator (anti-preference-leakage, ICLR 2026). If generator is Gemma → judge is Qwen or DeepSeek. If generator is Qwen → judge is DeepSeek or Gemma. Frontier APIs reserved for occasional triangulation. |
| **Text embeddings** | `BAAI/bge-m3` (local) · `voyage-3-large` (API) · `gemini-embedding-2` (API) | Local BGE-M3 first — dense+sparse+multi-vector in one model, no per-call cost. Upgrade to Voyage or Gemini only if eval shows BGE-M3 underperforming. Embeddings are one-time, so keeping infra simple wins. |
| **Visual document retrieval** | `ColQwen2.5` via `colpali-engine` (local MPS / Modal) · `gemini-embedding-2` as single-vector baseline | ColQwen2.5 is current ViDoRe V2 leader and the portfolio-credible choice (late-interaction is a real engineering story). Gemini Embedding 2 retained as a single-vector baseline for the writeup. |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (local CPU) · hosted alternative if local latency > 80ms p95 | Local default; only swap if the latency budget breaks. |
| **ASR** | `WhisperX large-v3` (local MPS) | No real contender for offline batch ASR on Mac. |
| **Premium triangulation** (one-off on test_gold for the writeup) | `claude-sonnet-4-6` · `gpt-5.5` | Run once at the end. Costs ~$3 each. Lets the methodology report frontier-API numbers alongside the chosen open-weight stack. |

## DeepInfra pricing (verified May 2026)

| Model | Input $/MTok | Output $/MTok |
|---|---|---|
| Qwen3-235B-A22B-Instruct | $0.071 | $0.10 |
| Gemma 4 31B | $0.13 | $0.38 |
| DeepSeek V3.2 | $0.32 | $0.89 |
| Claude Sonnet 4.6 (Anthropic API) | $3.00 | $15.00 |

The price spread between Qwen3 and Gemma is ~3×. Between Qwen3 and Sonnet is ~150×. This is why the selection rule matters: if Qwen3 lands within 3pp of Gemma on the eval, the project saves ~3× on every generation call.

## The bakeoff (roadmap phase 2, week 6)

1. Pin all retrieval/rerank/judge variables. Vary only the generator candidate.
2. Run the full dev_gold eval against each candidate. Log per-example traces in Langfuse with `candidate=<name>`, `git_sha`, and `judge=<name>`.
3. Run the judge calibration audit (20 manual boundary scores) for each judge candidate. Pick the cheapest judge that hits kappa ≥ 0.60.
4. Apply the selection rule. Publish the table in `eval/reports/<date>_generator_bakeoff/methodology.mdx`.
5. Lock the winner into `pyproject.toml` config. Document the runner-up and the cost differential so the writeup can show what the project saved by running the bakeoff.

Estimated bakeoff cost: **under $2 total** (3 generators × 80 dev examples + judge passes). Cheaper than one round of manual prompt-tuning at frontier-API prices.

## Cost ceiling (12-week budget, revised)

Cannot be predicted precisely until the bakeoff runs. Bounded estimates:

| Cost driver | Low (Qwen3 wins) | High (Gemma wins, Sonnet triangulation) |
|---|---|---|
| Eval runs through phase 2-3 (~10 cycles, 80 examples × ~5K tokens out) | ~$0.50 | ~$5 |
| Judge passes across eval cycles | ~$2 | ~$8 |
| Bakeoff (one-time, generator + judge selection) | ~$2 | ~$2 |
| Topic-LLM chunking on full corpus (local) | $0 | $0 |
| Embeddings (BGE-M3 local if it qualifies) | $0 | ~$5 (if Voyage needed) |
| Premium triangulation (one run, two frontier models) | ~$5 | ~$5 |
| Modal GPU bursts (ColQwen2.5 ingest) | ~$8 | ~$8 |
| **Total** | **~$18** | **~$33** |

Both bounds are well under the ~$110 v1 estimate and the ~$25 v2 estimate. The selection rule is what makes the lower bound reachable.

If sustained monthly spend exceeds $20, action is: cap eval cadence, not swap providers.

## Why not single-model default

v2 said "Gemma 4 31B is the default." That was premature because:

- **No eval data exists yet** to justify the choice over Qwen3 or DeepSeek.
- **The price spread is large enough that the difference matters.** A 3× cost gap is not noise; it is the difference between $33 and $18 over 12 weeks, and the difference between "I picked it" and "I selected it with data" in an interview.
- **A bakeoff costs $2 and produces the strongest possible writeup section.** Skipping it for the convenience of locking in Gemma sooner is anti-thesis.

## Why these specific candidates, not more

Adding more candidates costs almost nothing in bakeoff dollars but a lot in cognitive load and reporting clarity. Three is enough to demonstrate the selection process. Adding a fourth (e.g. Mistral Large 2, GLM 5.1) is an explicit decision that needs justification beyond "more is more."

## Revisit when

- A candidate not in the set ships and clearly beats the chosen winner on the published thresholds → add to the candidate set and re-run the bakeoff. Cost: $2.
- BGE-M3 fails the embeddings minimum → promote Voyage / Gemini Embedding 2 from fallback to primary.
- Total monthly API spend exceeds $20 for two consecutive months → tighten eval cadence first, then re-shop providers.
- DeepInfra raises Qwen3 or Gemma pricing materially → re-shop (Together, Fireworks, Cerebras, Groq).

## Changelog

**2026-05-24 (v3):** Replaced single-model default with candidate-set + selection-rule. Added Qwen3-235B-A22B-Instruct as primary candidate (~3× cheaper than Gemma on DeepInfra). Switched embeddings primary to local BGE-M3 (was: Voyage). Added explicit per-component minimums and 3pp tie-break tolerance. Added bakeoff as a phase-2 roadmap deliverable. Estimated 12-week spend now bounded $18–$33.

**2026-05-24 (v2, superseded same day):** Pivoted from closed-API default to single-model open-weight default (Gemma 4 31B generator, DeepSeek-V3 judge, Gemma 4 E4B local extraction). Driven by Gemma 4's April release closing the agentic gap and DeepInfra's 30× pricing advantage. The single-model framing was wrong for a project whose thesis is eval-driven selection.

**2026-05-24 (v1, superseded same day):** Initial ADR. Closed-API default (Claude Sonnet 4.6 generator, GPT-4.1 judge). Correct for a January-2026 worldview; obsolete after May 2026 research.
