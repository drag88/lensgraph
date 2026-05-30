# Phase A — recall fix (offline pooling probe)

Side-channel probe. n = 10 visual-required rows. 514 frames re-scored per query under each prefilter signal. No model load except one-time query encoding; no production/DB writes. prefilter_k = 500.

**These numbers are DIAGNOSTIC ONLY and must not guide any model or pooling decision.** The `prod_mean_cosine` arm (the production stage-(a) signal) did NOT reproduce the committed ablation ranks — 6/10 vs 4/10 in prefilter@200, and the per-example gold ranks below diverge wildly from `2026-05-28_visual_ablations/ablation_data.jsonl`. That mismatch is what the harness exposed: a broken, non-deterministic ColQwen load. The LoRA adapter is randomly re-initialized on every process start (cosine 0.844 between two encodes of the same query across processes), so each recall count below is computed from a random-draw query encoder and is not reproducible. The arm labels (`maxsim_hier_*`, `maxsim_full`, `maxsim_gauss`) describe the intended pooling experiment, but no pooling/ranking conclusion can be drawn until ColQwen loading is fixed. See `methodology.mdx` for the root cause and the fix path.

## VisualFrameRecall@k by arm

| arm | @5 | @10 | @20 | @200 |
|---|---|---|---|---|
| `prod_mean_cosine` | 1/10 | 1/10 | 1/10 | 6/10 |
| `maxsim_full` | 0/10 | 0/10 | 1/10 | 7/10 |
| `maxsim_hier_pf3` | 1/10 | 1/10 | 1/10 | 8/10 |
| `maxsim_hier_pf2` | 0/10 | 0/10 | 1/10 | 7/10 |
| `maxsim_gauss` | 0/10 | 0/10 | 0/10 | 5/10 |

## Gold-frame rank by arm (of 514)

| example_id | `prod_mean_cosine` | `maxsim_full` | `maxsim_hier_pf3` | `maxsim_hier_pf2` | `maxsim_gauss` |
|---|---|---|---|---|---|
| `visual-required-tengyu-accuracy-plot-axes` | 69 | 146 | 158 | 157 | 116 |
| `visual-required-tengyu-accuracy-plot-models` | 127 | 16 | 3 | 14 | 37 |
| `visual-required-tengyu-matryoshka-chart-axes` | 141 | 94 | 48 | 69 | 136 |
| `visual-required-tengyu-hybrid-search-diagram` | 242 | 188 | 179 | 167 | 235 |
| `visual-required-sally-current-plan-block` | 173 | 96 | 95 | 120 | 188 |
| `visual-required-sally-jq-grepjson-queries` | 4 | 324 | 169 | 230 | 306 |
| `visual-required-sally-lessons-recoverable-exceptions` | 35 | 167 | 67 | 61 | 167 |
| `visual-required-google-agents-cli-seven-skills-list` | 382 | 108 | 131 | 119 | 338 |
| `visual-required-google-adk-resumability-config-code` | 441 | 434 | 214 | 324 | 430 |
| `visual-required-google-adk-resume-version-and-caveat` | 438 | 392 | 229 | 342 | 450 |
