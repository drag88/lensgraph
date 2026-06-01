# Handoff — Phase 2, chunking ablation open (2026-06-01)

Consolidated current-state handoff. Full session-by-session detail lives in
`dev/active/phase-2-week-9/handoff.md`; this is the forward-looking entry point.

## Five-line read

1. **Chunking ablation is built and run, PROVISIONAL, no winner.** fixed_window
   (baseline, 213 chunks) vs transcript_segment (sentence-aligned, 82 chunks).
   Both score TR@5 = 1.00 but **GoldSpanContained@5 = 0.00 and IoU@1 ≈ 0.2** —
   neither localizes tightly.
2. **Root cause is the gold spans, not the chunkers.** dev_gold single_clip spans
   average **108s** (62–155s). No chunk granularity contains or tightly matches a
   108s span, so IoU/Contained can't crown a winner yet. **This is the #1 thing to
   resolve next** (a curation/methodology decision the user must own).
3. **The substrate is clean.** A YouTube rolling-caption VTT dedup bug (chunk text
   was triplicated + time-shifted) was fixed and all 3 talks re-chunked +
   re-embedded: 212 → 213 fixed_window chunks, embeddings winner unchanged
   (`bge-m3-all-channels`), rrf_4ch TR@5 improved 0.90 → 1.00.
4. **Two judges exist, both provisional.** Generator bakeoff #2 (faithfulness) ran
   on the pre-dedup substrate, no winner (latency minimum mis-scoped). Boundary
   judge (`boundary_v1`) has a provisional edge-kappa 0.81 but the human labels
   weren't independent (accepted LLM suggestions) and standalone is unvalidated —
   needs a **cold labeling pass**.
5. **Nothing is locked. No ADR amended.** All selections await validated judges +
   resolved gold-span sizing.

## Current state (what is true now)

- HEAD: `08d4dc8`. Gates green: `validate-evals-strict`, `phase0-gate` (29/3),
  `make test` (197), `lint`. Postgres stopped (volume `pgdata` persists data).
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

- **No chunking winner** — IoU/Contained are dominated by the 108s gold-span size,
  not strategy quality. Resolve span sizing before reading them as a selection.
- **Boundary judge not cold-validated** — edge-kappa 0.81 is inflated (human took
  LLM suggestions); standalone kappa ≈ 0.07. No boundary/chunking-quality claim.
- **Generator bakeoff provisional** — ran on the 212-chunk pre-dedup substrate;
  `p95_latency_sec_max: 2.0` is mis-scoped for the non-streaming JSON harness
  (real p95 40–66s). Re-run on the 213-chunk substrate before any selection.
- **Local label/worksheet files are untracked** (`dev/active/phase-2-week-9/
  boundary_audit*.{md,csv,xlsx}`, `edge_recheck.csv`) — transient, not committed.

## Next priorities (in order)

1. **Gold-span sizing decision (gates everything).** Review the 10 dev_gold
   single_clip spans (avg 108s). Decide: is a "clip" the full ~108s answer region
   (then chunk-IoU is the wrong yardstick — report region-recall) or a tight
   ~20–40s moment (then re-curate spans by watching, which makes IoU/Contained
   discriminating and lets the chunking ablation actually select)? **User owns this.**
2. **Re-run the chunking ablation** under the resolved span definition.
3. **Cold boundary-kappa pass** — user labels a fresh ~20 clips (edge + standalone,
   no suggestions) → validate `boundary_v1` → unlock the boundary tier.
4. **Add corpus-fit strategies** once the metric discriminates: `slide_boundary`
   (uses the visual frames; best corpus fit) and a cheap embedding-similarity
   semantic chunker (tests the tengyu sparse-punctuation weakness). Defer
   topic_llm/hybrid.
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

Postgres stopped. Verified baseline: HEAD `08d4dc8`. `test_gold.jsonl` untouched.
Visual retrieval / ColQwen loader closed — do not reopen.
