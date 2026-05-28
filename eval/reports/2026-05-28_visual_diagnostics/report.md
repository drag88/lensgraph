# Visual Retrieval Diagnostics — 2026-05-28

Side-channel report. n = 10 visual-required examples (tagged `visual-required` in `eval/corpora/ai_engineering_v0/visual_gold.jsonl`).

Scope: 3-stage `retrieve.visual.retrieve` pipeline + the `retrieve_frames` path.
No production code modified. No `eval_runs` / `eval_results` written.

## Aggregate rank curves

| Metric | @5 | @10 | @20 | @50 |
|---|---|---|---|---|
| `VisualFrameRecall@k` | 0/10 (0%) | 0/10 (0%) | 1/10 (10%) | 1/10 (10%) |
| `VisualChunkTR@k` | 1/10 (10%) | 1/10 (10%) | 2/10 (20%) | 2/10 (20%) |

## Near-miss vs total-miss split

- Near miss = gold rank 6-20 (within reach of a small reranker or wider k).
- Total miss = gold rank > 50 or absent from top-200.

| Path | near-miss (6-20) | total-miss (>50 or absent) |
|---|---|---|
| frame_recall | 1/10 | 9/10 |
| chunk_tr | 1/10 | 8/10 |

## Stage-loss attribution

Where the gold-overlapping chunk falls out of the top-5 band, earliest stage first.

| Stage | n | description |
|---|---|---|
| stage_1_miss | 5 | gold-window frame absent from top-200 pooled HNSW prefilter |
| stage_2_miss | 0 | prefilter found the frame but frame→chunk SQL join produced no gold-overlapping chunk |
| stage_3_miss | 4 | gold-overlapping chunk surfaced post-join but MaxSim ranked it outside top-5 |
| passed_at_5  | 1 | gold chunk ranks at or above 5 |

## Per-example detail

| example_id | frame_rank | chunk_rank | stage 1 | stage 2 | stage 3 | RRF (with → without) |
|---|---|---|---|---|---|---|
| `visual-required-google-adk-resumability-config-code` | absent (None) | absent (None) | MISS (0/200) | MISS (0/117) | n/a | False → True |
| `visual-required-google-adk-resume-version-and-caveat` | absent (None) | absent (None) | MISS (0/200) | MISS (0/113) | n/a | True → True |
| `visual-required-google-agents-cli-seven-skills-list` | absent (None) | absent (None) | MISS (0/200) | MISS (0/108) | n/a | True → True |
| `visual-required-sally-current-plan-block` | >50 (127) | >50 (111) | rank 127/200 | 3/118 chunks | rank 111/118 | True → True |
| `visual-required-sally-jq-grepjson-queries` | >50 (68) | >50 (93) | rank 68/200 | 4/113 chunks | rank 93/113 | True → True |
| `visual-required-sally-lessons-recoverable-exceptions` | >50 (85) | >50 (94) | rank 85/200 | 2/109 chunks | rank 94/109 | True → True |
| `visual-required-tengyu-accuracy-plot-axes` | >50 (105) | 6-20 (13) | rank 105/200 | 6/108 chunks | rank 13/108 | True → True |
| `visual-required-tengyu-accuracy-plot-models` | 6-20 (16) | 1-5 (5) | rank 16/200 | 6/124 chunks | rank 5/124 | True → True |
| `visual-required-tengyu-hybrid-search-diagram` | absent (None) | absent (None) | MISS (0/200) | MISS (0/114) | n/a | True → True |
| `visual-required-tengyu-matryoshka-chart-axes` | absent (None) | absent (None) | MISS (0/200) | MISS (0/107) | n/a | True → True |

## Per-example narrative

### `visual-required-google-adk-resumability-config-code`

Gold span: `nXafozNIk3c` [2010, 2030] sec. frame_recall_rank = `None`; chunk_tr_rank = `None`.

- **Stage 1 (pooled HNSW prefilter):** MISS — no gold-window frame in the top `200` pooled candidates. The right frame is buried below the prefilter horizon; widening `prefilter_k` may or may not help, but the cosine signal on the pooled vector is not surfacing it within 200.
- **Stage 2 (frame→chunk join):** n/a — stage 1 already failed.
- **Stage 3 (MaxSim refine):** n/a — no gold-overlapping chunk reached this stage.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=False`, `without_visual=True` (disagree).

### `visual-required-google-adk-resume-version-and-caveat`

Gold span: `nXafozNIk3c` [2000, 2030] sec. frame_recall_rank = `None`; chunk_tr_rank = `None`.

- **Stage 1 (pooled HNSW prefilter):** MISS — no gold-window frame in the top `200` pooled candidates. The right frame is buried below the prefilter horizon; widening `prefilter_k` may or may not help, but the cosine signal on the pooled vector is not surfacing it within 200.
- **Stage 2 (frame→chunk join):** n/a — stage 1 already failed.
- **Stage 3 (MaxSim refine):** n/a — no gold-overlapping chunk reached this stage.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=True`, `without_visual=True` (agree).

### `visual-required-google-agents-cli-seven-skills-list`

Gold span: `nXafozNIk3c` [520, 595] sec. frame_recall_rank = `None`; chunk_tr_rank = `None`.

- **Stage 1 (pooled HNSW prefilter):** MISS — no gold-window frame in the top `200` pooled candidates. The right frame is buried below the prefilter horizon; widening `prefilter_k` may or may not help, but the cosine signal on the pooled vector is not surfacing it within 200.
- **Stage 2 (frame→chunk join):** n/a — stage 1 already failed.
- **Stage 3 (MaxSim refine):** n/a — no gold-overlapping chunk reached this stage.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=True`, `without_visual=True` (agree).

### `visual-required-sally-current-plan-block`

Gold span: `aie_sg_2026_d2_arize_alyx` [314, 360] sec. frame_recall_rank = `127`; chunk_tr_rank = `111`.

- **Stage 1 (pooled HNSW prefilter):** PASS — the first gold-window frame appears at rank `127` out of `200` pooled candidates.
- **Stage 2 (frame→chunk join):** PASS — `3` of `118` candidate chunks overlap the gold span.
- **Stage 3 (MaxSim refine):** the gold-overlapping chunk ranks `111` of `118` after MaxSim. Does not pass at @5.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=True`, `without_visual=True` (agree).

### `visual-required-sally-jq-grepjson-queries`

Gold span: `aie_sg_2026_d2_arize_alyx` [546, 600] sec. frame_recall_rank = `68`; chunk_tr_rank = `93`.

- **Stage 1 (pooled HNSW prefilter):** PASS — the first gold-window frame appears at rank `68` out of `200` pooled candidates.
- **Stage 2 (frame→chunk join):** PASS — `4` of `113` candidate chunks overlap the gold span.
- **Stage 3 (MaxSim refine):** the gold-overlapping chunk ranks `93` of `113` after MaxSim. Does not pass at @5.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=True`, `without_visual=True` (agree).

### `visual-required-sally-lessons-recoverable-exceptions`

Gold span: `aie_sg_2026_d2_arize_alyx` [590, 625] sec. frame_recall_rank = `85`; chunk_tr_rank = `94`.

- **Stage 1 (pooled HNSW prefilter):** PASS — the first gold-window frame appears at rank `85` out of `200` pooled candidates.
- **Stage 2 (frame→chunk join):** PASS — `2` of `109` candidate chunks overlap the gold span.
- **Stage 3 (MaxSim refine):** the gold-overlapping chunk ranks `94` of `109` after MaxSim. Does not pass at @5.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=True`, `without_visual=True` (agree).

### `visual-required-tengyu-accuracy-plot-axes`

Gold span: `W_CYk2ogcDI` [305, 410] sec. frame_recall_rank = `105`; chunk_tr_rank = `13`.

- **Stage 1 (pooled HNSW prefilter):** PASS — the first gold-window frame appears at rank `105` out of `200` pooled candidates.
- **Stage 2 (frame→chunk join):** PASS — `6` of `108` candidate chunks overlap the gold span.
- **Stage 3 (MaxSim refine):** the gold-overlapping chunk ranks `13` of `108` after MaxSim. Does not pass at @5.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=True`, `without_visual=True` (agree).

### `visual-required-tengyu-accuracy-plot-models`

Gold span: `W_CYk2ogcDI` [305, 410] sec. frame_recall_rank = `16`; chunk_tr_rank = `5`.

- **Stage 1 (pooled HNSW prefilter):** PASS — the first gold-window frame appears at rank `16` out of `200` pooled candidates.
- **Stage 2 (frame→chunk join):** PASS — `6` of `124` candidate chunks overlap the gold span.
- **Stage 3 (MaxSim refine):** the gold-overlapping chunk ranks `5` of `124` after MaxSim. Passes at @5.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=True`, `without_visual=True` (agree).

### `visual-required-tengyu-hybrid-search-diagram`

Gold span: `W_CYk2ogcDI` [460, 520] sec. frame_recall_rank = `None`; chunk_tr_rank = `None`.

- **Stage 1 (pooled HNSW prefilter):** MISS — no gold-window frame in the top `200` pooled candidates. The right frame is buried below the prefilter horizon; widening `prefilter_k` may or may not help, but the cosine signal on the pooled vector is not surfacing it within 200.
- **Stage 2 (frame→chunk join):** n/a — stage 1 already failed.
- **Stage 3 (MaxSim refine):** n/a — no gold-overlapping chunk reached this stage.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=True`, `without_visual=True` (agree).

### `visual-required-tengyu-matryoshka-chart-axes`

Gold span: `W_CYk2ogcDI` [405, 510] sec. frame_recall_rank = `None`; chunk_tr_rank = `None`.

- **Stage 1 (pooled HNSW prefilter):** MISS — no gold-window frame in the top `200` pooled candidates. The right frame is buried below the prefilter horizon; widening `prefilter_k` may or may not help, but the cosine signal on the pooled vector is not surfacing it within 200.
- **Stage 2 (frame→chunk join):** n/a — stage 1 already failed.
- **Stage 3 (MaxSim refine):** n/a — no gold-overlapping chunk reached this stage.
- **Stage 4 (RRF, informational):** prior eval shows `with_visual=True`, `without_visual=True` (agree).
