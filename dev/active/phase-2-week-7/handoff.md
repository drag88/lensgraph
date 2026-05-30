# Handoff — Phase 2 Week 7 (visual eval v4 — 720p + ablation)

Opened: 2026-05-28 (visual eval v3 closed earlier today at HEAD 1767704)
Status: visual eval v4 shipped. 720p re-ingest done. Prefilter ablation done. The blocker is now isolated: **ColQwen2.5 MaxSim**, not prefilter, not OCR, not source resolution.
Prior handoff state: visual eval v3 stack accepted by Codex review at HEAD 1767704.

## What is true now

1. yt-dlp bumped to `2026.03.17` (operator-side). `MIN_HEIGHT_PX = 480` guard in `scripts/fetch_video.py` passed.
2. Three talks re-fetched at 1280×720; 360p originals preserved at `videos/ai_engineering_v0/.360p_backup/`.
3. Frames re-extracted at 1280×720; PNG dimensions verified on disk.
4. 514 of 529 frames carry `pooled_embedding`. 15 frames hit a new ColQwen NaN/inf path on AV1-source 720p inputs (12 in `nXafozNIk3c`, 3 in `W_CYk2ogcDI`). Defensive skip added to `ingest/handlers.embed_frames_handler` (uncommitted) — bad frames carry NULL pooled_embedding so HNSW retrieval omits them.
5. Patches: 375,734 total (was 163,990 at 360p); ~730/frame at 720p vs ~310/frame at 360p — 2.3× the patch count.
6. visual_eval re-run: `visual-eval-0acd933a1c18`. `VisualFrameRecall@5 = 0.056`, `VisualChunkTR@5 = 0.056`, `AnswerTermHit@5 = 0.000`, `VisualAnswerGrounding@5 = 0.000` (n_eval=10). `lift_chunk_tr_pp = 0.0` (was -5.56 at v3). `lift_answer_term_pp = 0.0`.
7. Ablation `prefilter_k ∈ {200, 500, 1000}` executed against the 720p substrate. At K=200, 7 of 10 visual-required rows fail at stage 1 (gold frame absent from prefilter). At K=500, **10 of 10 frames reach the prefilter**. But MaxSim then ranks the gold-overlapping chunk at positions 60-172 of ~210 chunks across all K values. Pass@5 stays 0/10 in every condition.

## Headline numbers — before/after table

| Metric | 360p (v3) | 720p (v4) | Δ |
|---|---:|---:|---:|
| `VisualFrameRecall@5` (n=18) | 0.056 | 0.056 | 0 |
| `VisualChunkTR@5` (n=18) | 0.000 | 0.056 | +1 |
| `AnswerTermHit@5` (n_eval=10) | 0.000 | 0.000 | 0 |
| `VisualAnswerGrounding@5` (n_eval=10) | 0.000 | 0.000 | 0 |
| `lift_chunk_tr_pp` | -5.56 | 0.0 | +5.56 |
| `lift_answer_term_pp` | 0.0 | 0.0 | 0 |

Plus the ablation outcome: prefilter widening fully rescues stage 1 but pass@5 stays 0/10. **The remaining blocker is MaxSim.**

## Per-example failure attribution — visual-required slice

All 10 rows now categorised `ii_maxsim_demote` per `eval/reports/2026-05-28_visual_ablations/report.md`:

- 7 rows fail prefilter at K=200; all 10 rescued at K=500.
- All 10 rows: MaxSim demotes gold chunk to rank 60-172 (of ~210 chunks) regardless of K.
- Gaps from gold MaxSim score to top-5 cutoff are small (~+1.0 to +2.5 units), but the top-5 chunks are consistently from unrelated time windows in the same talk (e.g. "buy more H100s" instead of resumability config code). MaxSim prefers whole-frame visual similarity over slide-text relevance.

## Recommendation

**Try a different aggregator on ColQwen patches before swapping models.** Specifically:

1. **Option A — replace MaxSim** in `retrieve/visual.py` with a cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, already in the bakeoff #1 stack) over the top-100 K=500 prefilter candidates. Code-only change; no new model selection or budget.
2. **Option B — add a second visual candidate** in `eval/config/model_candidates.yaml` (e.g. ColPali v1.3 with PaliGemma backbone, or a CLIP-slide variant) and amend ADR 004. Larger lift.
3. **Option C — retire visual retrieval as ornamental.** The 4-channel text RRF passes 18/18 on span overlap. If A and B both fail, this is the honest answer; the v4 report names it.

Recommended next session: implement A as a `chunking_strategy=' ...'`-equivalent toggle on the visual channel so the change is reversible. Re-run visual_eval. If A doesn't move the metric, evaluate B vs C.

**Do NOT start Bakeoff #2** until visual is resolved.

## Commands run this session

```
yt-dlp version bump:        2026.03.03 → 2026.03.17
uv run python -m scripts.fetch_video --corpus ai_engineering_v0 --corpus-all
                            OK; 3 videos at 1280×720; height guard passed
uv run python -m ingest.cli (×3 talks)              OK; 529 frame rows enqueued
uv run python -m scripts.drain_ingest_queue ingest_frames    OK; 529 PNGs at 720p
uv run python -m scripts.drain_ingest_queue ingest_embed_frames
                            FIRST RUN: crashed at frame 21/318 (psycopg DataException — NaN in vector)
                            PATCH: added np.isfinite() guard in embed_frames_handler
                            SECOND RUN: drained 509 messages in ~28 min, 15 NaN-skips
uv run python -m eval.runners.run_visual_eval
                            OK → visual-eval-0acd933a1c18
                            frame_recall@5=0.056, chunk_tr@5=0.056,
                            answer_grounding@5=0.000 (n_eval=10), n=18
PYTHONPATH=. uv run python eval/reports/2026-05-28_visual_ablations/run_ablations.py
                            OK → ablation_data.jsonl (30 rows) + report.md
make validate-evals-strict  OK (5 valid + 15 invalid fixtures)
make phase0-gate            OK (29 verified / 3 talks)
make test                   142 passed, 201 deselected
make lint                   All checks passed
```

The slow integration test (`pytest -m slow eval/tests/test_run_visual_eval.py`) was not re-run in v4 because the only changes to `eval/runners/` since v3 are TODO comments. The headline `run_visual_eval` execution above doubles as the slow test path.

## What failed or was not run

- `ingest/handlers.py` defensive NaN/inf skip is in the working tree but NOT committed. Lives in v4 ingest scope; commit alongside the v4 report.
- Real-slide OCR sanity check (spot-test `pytesseract.image_to_string` on the 18 curated `.jpg` evidence frames) — still open. Same status as v3.
- OCR audit payload upgrade (structured `evaluated_frame_ids` / `ocr_excerpts` / `matched_term` / `failure_reason`). Still required before any future positive `VisualAnswerGrounding` result. TODO(v4) markers in `measure_visual.py:562` and `run_visual_eval.py:127` still apply.
- Investigation of the ColQwen NaN/inf path on specific 720p inputs. 15 frames skipped. Repro frame: `frames/nXafozNIk3c/frame_000021.png`.

## Files likely to change next

If recommendation **A (different aggregator)**:
- `retrieve/visual.py` — replace MaxSim refine with bge-reranker-v2-m3 cross-encoder over top-100 K=500.
- `eval/runners/measure_visual.py` + `eval/runners/run_visual_eval.py` — new metric? No, the existing metrics work; the change is in the retrieval implementation only.
- `eval/tests/test_run_visual_eval.py` — confirm the slow path still passes.

If recommendation **B (second visual candidate)**:
- `eval/config/model_candidates.yaml` — add the new option.
- `docs/decisions/004-model-selection.md` — amendment.

If recommendation **C (retire visual)**:
- `eval/runners/measure_visual.py::_rrf_with_visual` — drop the visual channel; 4-ch becomes the production path.
- ADR 004 amendment + ADR 005 retire note.
- Methodology MDX claiming "visual was tried and retired" with the v3+v4 reports as evidence.

## Out of this commit stack (intentionally)

Same caveat as v3 — these were modified pre-session and stay unstaged:
- `.claude/CLAUDE.md` deletion + root `CLAUDE.md` / `AGENTS.md`
- `.claude/rules/*`, `.claude/skills/*`, `.cursor/*`, `.cursorignore`, `.cursorindexingignore`, `.vscode/settings.json`
- `prompts/*`, `docs/codebase-explorer.html`
- formatter churn under `generate/`, `eval/runners/providers.py`, embeddings scripts, unrelated tests

## Stop-condition met

The v4 stop condition was: "**Either** `VisualAnswerGrounding@5` improves after 720p + ablation, **or** produce a defensible report showing whether the remaining blocker is prefilter, MaxSim, OCR, or ColQwen itself."

`VisualAnswerGrounding@5` did not improve (still 0/10). The defensible failure report (this file + `eval/reports/2026-05-28_visual_eval_v4/methodology.mdx` + `eval/reports/2026-05-28_visual_ablations/report.md`) shows **the remaining blocker is MaxSim**, with per-example data:

- prefilter loses on 7 of 10 at K=200; rescues all 10 at K=500.
- MaxSim ranks gold at 60-172 of ~210 across all K values.
- Top-5 chunks are from unrelated time windows in the same talk.

That meets the second branch of the stop condition.
