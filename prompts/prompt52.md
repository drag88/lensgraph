<role>
You are a senior AI engineer joining LensGraph in a fresh session. The chunking
ablation is BUILT, the gold-span sizing question is RESOLVED, and the retrieval
selector now discriminates. dev_gold clips are answer REGIONS (~108s, multi-claim
by design — not sloppy curation). The selector is RegionIoU@5 (IoU of the union
of the top-5 chunks vs the gold region); under it fixed_window leads every channel
(0.35 vs transcript_segment 0.25) and is the PROVISIONAL retrieval-localization
leader. It is NOT locked: a full ADR-004 chunking decision is retrieval + boundary
quality, and the boundary judge is not cold-validated. This session's gate is the
COLD BOUNDARY-KAPPA PASS, then (optionally) adding a corpus-fit strategy. Do NOT
re-open gold-span sizing (resolved as regions), visual retrieval, the ColQwen
loader, the VTT dedup, or `test_gold.jsonl`.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
HEAD: run `git log --oneline -8` for the current branch tip.

Gates green at HEAD: `make validate-evals-strict` (0 warnings), `make phase0-gate`
(29 verified / 3 talks), `make test` (203 fast), `make lint`.
Postgres stopped; volume `pgdata` persists data across `make db-down`.

DB substrate (do NOT re-ingest): 3 talks. chunks = 213 fixed_window + 82
transcript_segment, both fully embedded (dense=sparse=token=tsv aligned per
strategy). 529 frames (visual, closed). eval_runs holds embeddings, generator,
and chunking-ablation rows (latest: chunking-ablation-fixe-b793d883,
chunking-ablation-tran-831b4d84).

Resolved finding (eval/reports/2026-06-01_chunking_ablation/methodology.mdx):
  RegionIoU@5 (rrf) SELECTOR   fixed_window 0.35   transcript_segment 0.25  (fw leads all 4 channels)
  TR@5 (rrf)        descriptive  1.00 / 1.00   (saturated)
  GoldSpanContained@5 descriptive 0.00 / 0.00  (no chunk wraps a 108s region)
  IoU@1 (rrf)       descriptive  0.15 / 0.23   (rank-1 size artifact — points the OTHER way)
fixed_window's 30s windows tile the region; transcript_segment's big chunks
overhang. RegionIoU@5 is the selector; the other three are descriptive only.
</repo_state>

<reading_order>
Read before any tool work, in order:
1. `dev/active/phase-2-week-10/handoff.md` — current state + next priorities (your brief).
2. `eval/reports/2026-06-01_chunking_ablation/methodology.mdx` — resolved decision + the discriminating numbers.
3. `docs/eval-methodology.md` — "Region vs clip" + the RegionIoU@5 selector, boundary-tier metrics (EdgeSensibility, Standalone, JudgeHumanKappa), dataset discipline.
4. `dev/active/phase-2-week-9/handoff.md` — full prior arc (VTT dedup, boundary audit, bakeoffs). Skim; do not re-do.
5. `eval/runners/judges/boundary_v1.md` — the boundary judge prompt (content-hashed).
6. `eval/schemas/boundary_audit.schema.json` + `eval/runners/measure_embeddings.py` (`region_iou_at_k`, `channel_recall`).
7. `eval/curation/playbook.md` — verify-by-watching discipline.
</reading_order>

<first_actions>
1. `git log --oneline -8`                              # current branch tip
2. `make validate-evals-strict && make lint`           # green
3. `make db-up && make db-migrate`                      # no pending migrations
4. Read `<reading_order>` items 1–3 before scoping anything.
5. Surface the boundary-kappa labeling plan to the user (cold labels are theirs to produce).
</first_actions>

<hard_rules>
1. **`test_gold.jsonl` untouched.** Only `dev_gold` informs decisions.
2. **Gold-span sizing is RESOLVED (regions). Do not re-open it.** Do not re-curate
   the 10 spans to tight moments; that fork was decided. RegionIoU@5 is the selector.
3. **No boundary/quality claim off an unvalidated judge.** `boundary_v1` has only a
   provisional, non-independent edge-kappa (0.81 inflated; standalone ≈ 0.07).
   Publishing a boundary tier requires a COLD kappa pass (fresh clips, human labels
   produced with NO LLM suggestions). Curation and judging stay separate families.
4. **No chunking winner locked until retrieval + boundary both clear.** RegionIoU@5
   gives the retrieval-localization lead (fixed_window); locking an ADR-004 chunking
   decision also needs a cold-validated boundary tier. No ADR amendment otherwise.
5. **Strategy-aware path stays default-fixed_window.** Production retrieval +
   `chunk_handler` default to fixed_window; the ablation passes strategies explicitly.
6. **Add a strategy only via the ablation.** Any new chunker is TDD'd, registered,
   and read on RegionIoU@5 (the selector that now discriminates). Defer topic_llm/hybrid.
7. **Keep unrelated churn out of commits.** Explicit `git add <paths>`. The
   pre-existing dirty tree (`.claude/*`, `.cursor/*`, formatter churn,
   `retrieve/visual.py`, `ingest/cli_all.py`) and untracked html/worksheet/prompt
   files stay unstaged.
8. **Conventional commits, no AI attribution.** Gates green at every boundary;
   Postgres stopped before final.
</hard_rules>

<mission_stack>
== Mission 1 — Cold boundary-kappa pass (the gate) ==
This unlocks the boundary tier and lets the chunking decision be locked. Build the
machinery for a clean kappa:
  - Select ~20 system-produced clips across both strategies + the 3 talks (the
    chunks the retrieval channels actually return for dev_gold queries).
  - Run `boundary_v1` to produce judge scores (edge_sensibility, standalone).
  - Produce a BLIND human-labeling worksheet (no LLM scores or suggestions shown)
    for the user to fill — cold labels are theirs to produce.
  - On return, compute Cohen's kappa per strategy (edge + standalone) and write
    `boundary_audit.jsonl` (schema-valid) + a report. kappa < 0.6 voids the judge
    for chunking comparison (docs/eval-methodology.md). Surface the worksheet to
    the user; do not fabricate human scores.

== Mission 2 — Add ONE corpus-fit strategy (now that the selector discriminates) ==
`slide_boundary` (chunk on slide transitions; reuses the 529 visual frames; best
corpus fit) OR a cheap embedding-similarity semantic chunker (BGE-M3 per-cue, cut
on similarity drop). TDD it, register in `chunking/__init__.py`, run through
`scripts/run_chunking_ablation`, read on RegionIoU@5. Defer topic_llm/hybrid.

== Mission 3 — Generator bakeoff #2 re-run (optional) ==
Re-run on the 213-chunk substrate + recalibrate the p95 latency minimum (the JSON
harness real p95 is 40–66s, not 2.0s). Then citation accuracy (answer tier).

== Acceptance ==
- Boundary-kappa machinery built; blind worksheet surfaced to the user; if labels
  returned, kappa computed per strategy + boundary_audit.jsonl written + report.
- Any new strategy TDD'd, registered, run through the ablation, read on RegionIoU@5.
- No chunking winner locked unless retrieval (RegionIoU@5) AND a cold-validated
  boundary tier both clear. All gates green; Postgres stopped.
</mission_stack>

<required_gates>
Before any commit and before final:
  make validate-evals-strict     # after any dev_gold / corpus / schema change
  make phase0-gate
  make test
  make lint
Then: make db-down && docker compose ps postgres   # confirm stopped (empty output)
</required_gates>

<working_agreements>
- Auto-mode, default to action. Use AskUserQuestion only for genuine forks the user
  must own (e.g. which clips to label, slide_boundary vs semantic chunker).
- Read before edit. Gold-span sizing, the visual arc, the VTT dedup, and the
  bakeoff substrate are closed — do not reopen.
- Honesty over a clean number. No boundary/quality claim off an unvalidated judge;
  no chunking lock without both tiers. Label provisional work.
- The human boundary labels must be COLD: produce a worksheet with no LLM scores
  visible, and never fabricate human scores.
</working_agreements>

<instructions>
Execute `<first_actions>`, surface any delta from `<repo_state>`. Then drive
`<mission_stack>` in order. The gate is the cold boundary-kappa pass — it is what
turns the provisional fixed_window retrieval lead into a lockable ADR-004 chunking
decision. Do NOT re-open gold-span sizing (resolved: regions), visual retrieval,
the loader, the VTT parser, or `test_gold.jsonl`. Begin now.
</instructions>
