# Handoff — Phase 2, chunking ablation (2026-06-01, gold-span sizing RESOLVED)

Consolidated current-state handoff. Full session-by-session detail lives in
`dev/active/phase-2-week-9/handoff.md`; this is the forward-looking entry point.

## Five-line read

1. **Gold-span sizing RESOLVED: clips are answer REGIONS (~108s, intended).** The
   questions are deliberately multi-claim (4–5 claims each), so tightening spans
   would mean rewriting the eval. Spans + questions kept as-is; the **metric** was
   fixed instead. Documented in `docs/eval-methodology.md` ("Region vs clip").
2. **New selector added: `RegionIoU@5`** — IoU of the union of top-5 chunks vs the
   gold region. It discriminates where TR@5 (saturated), Contained@5 (needs a
   108s chunk), and IoU@1 (rank-1 size artifact) could not. TR@5/Contained/IoU@1
   are now **descriptive only** for chunking selection.
3. **Ablation re-run under the selector — it discriminates and FLIPS the read.**
   RegionIoU@5 (rrf_4ch): **fixed_window 0.35 vs transcript_segment 0.25**, and
   fixed_window leads on *every* channel. (IoU@1 had pointed the other way — that
   was the size artifact.) fixed_window's 30s windows tile the region; ts's big
   chunks overhang. **fixed_window = provisional retrieval-localization leader.**
4. **Still NOT locked.** A full ADR-004 chunking decision is retrieval + boundary
   quality. The boundary judge (`boundary_v1`) is not cold-validated (edge-kappa
   0.81 inflated, standalone ≈ 0.07), and n=10 dev_gold is small. No winner
   locked, no ADR amended. Generator bakeoff #2 still pre-dedup/mis-scoped latency.
5. **What's new in code:** `region_iou_at_k` + 6 tests in
   `eval/runners/measure_embeddings.py` / `eval/tests/test_retrieval_metrics.py`;
   selector wired into `scripts/run_chunking_ablation.py`; report + methodology
   doc updated. Run IDs `chunking-ablation-fixe-b793d883` / `-tran-831b4d84`.

## Current state (what is true now)

- HEAD: run `git log --oneline -8` for the current branch tip. Gates green:
  `validate-evals-strict`, `phase0-gate` (29/3), `make test` (203), `lint`.
  Postgres stopped (volume `pgdata` persists data).
- **DB:** 3 talks. chunks: **213 fixed_window + 82 transcript_segment** (both
  embedded; dense=sparse=token=tsv aligned per strategy). 529 frames (visual,
  closed/untouched). eval_runs has embeddings + generator + chunking-ablation rows.
- **Chunking:** `chunking/fixed_window.py` (baseline), `chunking/transcript_segment.py`
  (sentence-aligned), shared `chunking/vtt.py` (roll-up dedup). Both registered.
- **Strategy-aware path:** `retrieve/{dense,sparse,multivec,bm25}.retrieve` take
  `chunking_strategy` (default `fixed_window` — production unchanged);
  `chunk_handler` honors a `strategy` payload key (default fixed_window).
- **Metrics:** `eval/runners/measure_embeddings.py` now has `gold_span_contained_at_k`,
  `iou_at_1`, `channel_recall`. Ablation runner: `scripts/run_chunking_ablation.py`.
- **Reports:** `eval/reports/2026-06-01_chunking_ablation/` (the gold-span finding),
  `eval/reports/2026-05-31_generator_bakeoff/` (provisional, annotated pre-dedup),
  `eval/reports/2026-05-27_embeddings_bakeoff/` (annotated re-run).
- **Judges:** `eval/runners/judges/{faithfulness_v1,boundary_v1}.md` (content-hashed).
- **Corpus:** `negative.jsonl` 20 corpus-scope rows. `boundary_audit.jsonl` empty
  (NOT committed — provisional data). `test_gold.jsonl` untouched throughout.

## What is broken / not trustworthy

- **Chunking winner not locked** — span sizing is resolved and `RegionIoU@5`
  discriminates (fixed_window 0.35 > ts 0.25, leads every channel), but locking
  needs the boundary tier (a chunking decision is retrieval + boundary quality)
  and the boundary judge is not cold-validated. Retrieval-localization lead only.
- **Boundary judge not cold-validated** — edge-kappa 0.81 is inflated (human took
  LLM suggestions); standalone kappa ≈ 0.07. No boundary/chunking-quality claim.
- **Generator bakeoff provisional** — ran on the 212-chunk pre-dedup substrate;
  `p95_latency_sec_max: 2.0` is mis-scoped for the non-streaming JSON harness
  (real p95 40–66s). Re-run on the 213-chunk substrate before any selection.
- **Local label/worksheet files are untracked** (`dev/active/phase-2-week-9/
  boundary_audit*.{md,csv,xlsx}`, `edge_recheck.csv`) — transient, not committed.

## Next priorities (in order)

1. ~~Gold-span sizing decision~~ **DONE** — regions; `RegionIoU@5` is the selector.
2. ~~Re-run the chunking ablation~~ **DONE** — discriminates, fixed_window ahead.
3. **Cold boundary-kappa pass** — user labels a fresh ~20 clips (edge + standalone,
   no suggestions) → validate `boundary_v1` → unlock the boundary tier. **This is
   now the gate** to combining RegionIoU@5 (retrieval) + boundary quality into an
   ADR-004 chunking decision and possibly locking fixed_window.
4. **Add corpus-fit strategies** now that the selector discriminates: `slide_boundary`
   (uses the visual frames; best corpus fit) and a cheap embedding-similarity
   semantic chunker (tests the tengyu sparse-punctuation weakness). Defer
   topic_llm/hybrid. Each runs through `run_chunking_ablation` and is read on
   RegionIoU@5.
5. **Generator bakeoff #2 re-run** on the clean substrate + recalibrate the latency
   minimum. Then citation accuracy (answer tier).

## Commands

```bash
git log --oneline -8
make db-up && make db-migrate
uv run python -m scripts.run_chunking_ablation --measure-only   # re-measure (no re-chunk)
make validate-evals-strict / phase0-gate / test / lint
make db-down
```

## Stop condition

Postgres stopped. For the current branch tip run `git log --oneline -8`.
`test_gold.jsonl` untouched.
Visual retrieval / ColQwen loader closed — do not reopen.
