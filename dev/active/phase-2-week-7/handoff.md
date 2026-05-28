# Handoff — Phase 2 Week 7 (visual eval v3)

Opened: 2026-05-28
Status: visual eval v3 shipped. Frame-side answer-grounding metric (`VisualAnswerGrounding@k`) now live and validated. The metric returns 0/10 on the visual-required slice: ColQwen2.5 does not surface answer-bearing frames on the current 360p ingest. **This is a defensible failure with a named next fix** — re-ingest at 720p (yt-dlp fix landed this session; operator action needed to bump the binary).
Prior handoff: `dev/active/phase-2-week-6/handoff.md` (CLOSED at `7c84cff`).
Memory-safety note: this session used a single-ColQwen-load team design to avoid the Cursor crashes that happened in two prior sessions.

## What is true now

1. `eval/runners/measure_visual.py` carries the new frame-side metric `VisualAnswerGrounding@k`. The primitive `frame_ocr_matches_at_k` and the aggregator `visual_answer_grounding_at_k` are exported. `frame_ocr_text` uses `pytesseract` (Tesseract 5.x, PSM 3, eng) and is LRU-cached at 4096 entries.
2. `eval/runners/run_visual_eval.py` loads `visual_evidence` from `visual_gold.jsonl` and writes the new metric into `eval_runs.summary` + `eval_results.metrics`.
3. `eval/tests/test_measure_visual.py` has 47 fast tests passing (9 new OCR-related, including the prose-vs-code regression).
4. `scripts/fetch_video.py` now enforces `MIN_HEIGHT_PX = 480` via post-fetch `ffprobe` so a silent 360p SABR fallback fails loudly. `eval/tests/test_fetch_video.py` has 2 new tests covering the guard (skip cleanly if `ffmpeg` is absent).
5. `docs/decisions/005-chapter-slice-ingest.md` v3 changelog entry explains the 360p root cause (`yt-dlp 2026.03.03` SABR throttling) and pins `yt-dlp >= 2026.3.17` as the operator floor.
6. `eval/reports/2026-05-28_visual_ablations/run_ablations.py` is written (722 LOC) but NOT executed. `DEFERRED.md` documents the resumption path.
7. Run `visual-eval-239f681cbf69` landed: `VisualFrameRecall@5=0.056` (1/18), `VisualChunkTR@5=0.000` (0/18), `AnswerTermHit@5=0.000` (0/10), `VisualAnswerGrounding@5=0.000` (0/10), `lift_chunk_tr_pp=-5.56`, `lift_answer_term_pp=0.0`.

## Headline finding

ColQwen2.5 does not surface a frame within the gold window for ANY of the 10 visual-required rows. The window gate fails 10/10. OCR is never reached. Adding the visual channel to the 4-channel text RRF makes span-overlap slightly worse (-5.56 pp, one example regresses; the rest match). This matches the prior diagnostic's 5/10 stage-1 misses + 4/10 stage-3 demotions.

## Recommendation

Keep fixing visual. Operator action needed first:

1. Bump yt-dlp to >= 2026.3.17 (`uv tool install yt-dlp@2026.3.17` or equivalent).
2. Re-fetch the 3 talks at 720p.
3. Re-ingest frames + ColQwen patches (~30 min Mac MPS).
4. Re-run `uv run python -m eval.runners.run_visual_eval`.
5. Run `uv run python eval/reports/2026-05-28_visual_ablations/run_ablations.py` for the `prefilter_k ∈ {200, 500, 1000}` sweep.
6. If `VisualAnswerGrounding@5` is still 0/10 after both fixes, the failure is at the model level; consider amending ADR 004 to add a second visual candidate.

Do NOT start Bakeoff #2 (generator + judge) until the visual question is resolved.

## Commands run this session

```
make db-up && make db-migrate          # OK, volume pgdata
make validate-evals-strict             # OK (5 valid + 15 invalid fixtures)
make phase0-gate                       # OK (29 verified across 3 talks)
make test                              # 142 passed (was 130; +9 OCR tests, +2 fetch_video tests, +1 measure_visual rebalance)
make lint                              # clean (after fixer's diff)
uv run python -m eval.runners.run_visual_eval
                                       # OK → visual-eval-239f681cbf69
                                       # frame_recall@5=0.056, chunk_tr@5=0.000, answer_grounding@5=0.000 (n_eval=10), n=18
uv run pytest -m slow eval/tests/test_run_visual_eval.py
                                       # 2 passed in 859.23s (0:14:19) — the second (and final) ColQwen load this session
```

All five gates green.

## What failed or was not run

- 720p re-ingest. Deferred to the operator + the next session. Agent B's fix is in but the binary bump + ~30 min re-ingest belongs to a deliberate next session.
- `run_ablations.py`. Deferred for memory reasons — see DEFERRED.md.

## Memory regression — root cause + mitigation

Two prior sessions crashed Cursor by running multiple `uv run python` processes that each loaded ColQwen 2.5 (~7-8 GB on MPS unified memory) on a 24 GB Mac. The processes were independent — no shared model server — so memory consumption was additive. Cursor (Electron renderer) got squeezed when swap usage exceeded ~7 GB.

Mitigation in this session:
- Only one ColQwen load total — in main context, after agents finished.
- Three agents did code-only work (yt-dlp guard fix, ablation script authoring, fast-test verification) — none touched ColQwen.
- Documented in `eval/reports/2026-05-28_visual_eval_v3/methodology.mdx` "Memory budget" section.

For future sessions touching visual retrieval: never run more than one ColQwen-loading process at a time on this hardware. Run them serially or split across sessions.

## Next command

```bash
git status --short                     # confirm clean stack
# Then commit in chunks (excluding the .claude/* and CLAUDE.md/AGENTS.md relocation that was pre-existing)
```

## Files likely to change next

Commits this session (in order):

1. **fetch_video guard + ADR v3** — `scripts/fetch_video.py`, `docs/decisions/005-chapter-slice-ingest.md`, `eval/tests/test_fetch_video.py`.
2. **VisualAnswerGrounding@k metric** — `eval/runners/measure_visual.py`, `eval/runners/run_visual_eval.py`, `eval/tests/test_measure_visual.py`, `pyproject.toml` (pytesseract pin), `uv.lock`.
3. **Visual eval v3 report** — `eval/reports/2026-05-28_visual_eval_v3/{methodology.mdx,summary.json,per_example.jsonl,integrity_check.md}`.
4. **Deferred ablation** — `eval/reports/2026-05-28_visual_ablations/{run_ablations.py,DEFERRED.md}`.
5. **Handoff** — this file.

Keep OUT of the commit stack (pre-existing relocation): `.claude/CLAUDE.md` deletion, root `CLAUDE.md`, root `AGENTS.md`, `.claude/rules/*`, `.claude/skills/*`, `.cursor/*`, `docs/codebase-explorer.html`, `prompts/*`.

## Open items

- 720p re-ingest path (above).
- **Real-slide OCR sanity check.** Tesseract 5.x is installed and the new metric primitives have mocked-OCR tests covering the substring + window-gate logic. We have NOT yet run `pytesseract.image_to_string(...)` on a real slide PNG in this corpus. Cheap to run on the 18 curated `.jpg` evidence frames under `eval/curation/visual_gold_review/2026-05-27/<video_id>/` — spot-checks the curator's `visual_evidence.required_any` terms are at least findable. Required before claiming any positive `VisualAnswerGrounding` result.
- **OCR audit payload (v4).** The runner writes only a bool. Any future positive result needs structured fields (`evaluated_frame_ids`, `ocr_excerpts`, `matched_term`, `failure_reason`). `TODO(v4)` markers are in `measure_visual.py` + `run_visual_eval.py`. Documented in `integrity_check.md` §B. Inert today (0/10 positives) but load-bearing when re-ingest changes the upstream retrieval.
- Stage-1 prefilter ablation (script written; deferred).
- Stage-3 MaxSim diagnosis (diagnostic report from prior session names 3 rows where MaxSim ranks gold at 93/94/111 of ~110).
