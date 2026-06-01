<role>
You are a senior AI engineer joining LensGraph in a fresh session. The visual-retrieval arc is closed and trustworthy: the ColQwen2.5 loader is fixed, every frame was re-embedded, the visual eval reproduces 17/18 · 18/18 · VAG 2/10, and the `VisualAnswerGrounding` audit payload landed so both VAG positives are hand-verified (`visual-eval-13edcf554e60`, HEAD `4855f70`). Do NOT re-open any of that. Your job this session is to stand up **bakeoff #2: generator + judge selection** — the next eval-as-product centerpiece. But bakeoff #2 is NOT shovel-ready: two corpus files are empty, the judge prompt does not exist, and there is uncommitted answer-loop WIP to triage. This session is REVIEW + CURATE + BUILD, not "press run." Do NOT touch visual retrieval, the loader, or `test_gold.jsonl`.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
Baseline HEAD: `4855f70 eval: visual answer-grounding audit re-run (visual-eval-13edcf554e60)`.

Recent commits at HEAD (verified; do NOT re-litigate):
  4855f70 eval: visual answer-grounding audit re-run (visual-eval-13edcf554e60)
  d67200c feat(eval): VisualAnswerGrounding audit payload
  f9f246a docs(handoff): week-9 visual-loader close; prompt50 for VAG audit payload
  280cdb0 fix(embed,eval): repair ColQwen final-norm orphan; re-ingest frames; visual eval

Gates green at HEAD:
  make validate-evals-strict   OK (0 warnings)
  make phase0-gate             OK (29 verified non-negative across 3 talks)
  make test                    OK (152 fast, 204 deselected)
  make lint                    OK
  slow test_run_visual_eval.py OK (2 passed in 678.98s)

DB state (volume `pgdata` survives `make db-down`) — substrate is correct, do NOT
re-ingest:
  3 talks · 212 chunks · 529 frames / 514 pooled / 375,734 patches.
  eval_runs: visual_eval rows present; bakeoff #1 (embeddings) run present.

== The blockers (your first job is to see them with your own eyes) ==

Bakeoff #2 as written in `eval/config/model_candidates.yaml` cannot run
end-to-end today. Confirm each at session start:

  wc -l eval/corpora/ai_engineering_v0/{dev_gold,negative,synthesis,boundary_audit}.jsonl
    EXPECT: dev_gold 10 · negative 0 · synthesis 1 · boundary_audit 0

  1. boundary_audit.jsonl is EMPTY → judge minimum `cohens_kappa_vs_human_min:
     0.60` cannot be computed → no kappa-validated judge selection.
  2. negative.jsonl is EMPTY → generator minimum `corpus_negative_refusal_min:
     0.90` cannot be measured.
  3. eval/runners/judges/ is EMPTY → no versioned, content-hashed judge prompt.
  4. generate/ + eval/runners/{providers,minimal_generation}.py carry uncommitted
     WIP (pre-existing dirty tree) — triage before any clean generator run.
  5. dev_gold.jsonl is the generator eval set: 10 rows. Small. Even a 1-row swing
     is 10pp — report n and CI, do not over-claim.

This session is REVIEW + CURATE + BUILD. Do not publish a generator winner off an
unvalidated judge or an empty negative set.
</repo_state>

<reading_order>
Required reading before any tool work, in this order:

1. **`dev/active/phase-2-week-9/handoff.md`** — both sessions; the "Prerequisites
   blocking bakeoff #2" block is your brief in shorter form.
2. **`CLAUDE.md`** hard rules 6 (cross-family judge), 9 (no silent model defaults /
   ADR-004 selection rule), and the "Run the Answer Loop" workflow.
3. **`docs/eval-methodology.md`** + **`.claude/rules/experiment-tracking.md`** —
   the Answer tier (ClaimsSupported, CitationAccuracy, AnswerRelevancy), the
   Abstention tier (RefusalRate, FalseRefusalRate), Judge Versioning (content-hash
   the prompt; kappa < 0.6 voids the judge), and the ADR-004 v3.1 selection rule.
4. **`docs/decisions/004-*.md`** — the candidate set + selection rule you must apply
   (cheapest within `tie_break_pp` = 3pp of the leader that meets all minimums).
   The runner-up + cost differential are part of the writeup, not a footnote.
5. **`eval/config/model_candidates.yaml`** — the generator + judge options and
   their `minimums`. Generators: gemma-4-31b, qwen3-235b-a22b-instruct,
   deepseek-v3.2. Judge: cross-family vs the chosen generator.
6. **`eval/runners/run_embeddings_bakeoff.py`** + **`eval/runners/measure_embeddings.py`**
   — bakeoff #1. Copy its transactional write pattern (single transaction:
   insert_run + per-example eval_results) and its `summary`/`per_example` shape.
7. **`eval/reports/2026-05-27_embeddings_bakeoff/methodology.mdx`** — the report
   format your generator-bakeoff report must match (command, candidate IDs, config
   sha, runner-up, cost differential, modality breakdown).
8. **`generate/graph.py`, `generate/api.py`, `generate/nodes/*.py`,
   `generate/state.py`, `generate/parser.py`** — the LangGraph answer loop
   (Plan → Retrieve → Rerank → Verify → Generate → Cite). Diff each vs HEAD to see
   the uncommitted WIP. Decide commit / revise / revert per file.
9. **`eval/runners/providers.py`** + **`eval/runners/minimal_generation.py`** —
   the DeepInfra wiring and the minimal generation runner. Confirm an API key path
   and that one real generation call succeeds before building the bakeoff loop.
10. **`eval/curation/playbook.md`** — the curation discipline for the negative +
    boundary-audit rows you will add. Cross-family curation vs judging is an
    invariant (hard rule 6).

Skim only (stable): README.md · docs/architecture.md · the visual-eval reports
(do not act on them — visual retrieval is closed this session).
</reading_order>

<first_actions>
Run these in order before reading. Verify state matches `<repo_state>`; surface any delta.

1. `git log --oneline -5`                                  # confirm 4855f70 at HEAD
2. `make validate-evals-strict`                            # 0 warnings
3. `make phase0-gate`                                      # OK (29 / 3)
4. `make test`                                             # 152 passed
5. `make lint`                                             # clean
6. `make db-up && make db-migrate`                         # no pending migrations
7. `wc -l eval/corpora/ai_engineering_v0/{dev_gold,negative,synthesis,boundary_audit}.jsonl`
   # confirm the empty negative + boundary_audit blockers
8. `git diff --stat generate/ eval/runners/providers.py eval/runners/minimal_generation.py`
   # the uncommitted answer-loop WIP to triage
9. `ls eval/runners/judges/`                               # confirm empty
10. Confirm a DeepInfra key is available (env / .env). Do NOT print the key. If
    absent, that is a hard blocker — stop and ask the user how to provide it.
11. Read `<reading_order>` in order before scoping any curation or code.
</first_actions>

<hard_rules>
Non-negotiable. From CLAUDE.md, the methodology docs, ADR 004, and the week-9 handoff.

1. **`test_gold.jsonl` is untouched.** Do not load, glance at, or measure against
   it. Only `dev_gold` informs selection. The validator enforces dev-only access.

2. **Cross-family judge.** The judge model family MUST differ from the chosen
   generator family AND from the curation family (ADR 004, methodology hard rule 6).
   Anthropic/Gemma curates → a different family judges. Same-family judging inflates
   faithfulness 5–15pp. This is a contamination invariant, not a preference.

3. **No model is the default until the bakeoff runs.** Do not write "we use X" in
   any artifact before the selection commit. Apply the ADR-004 rule literally:
   cheapest candidate within 3pp of the leader that meets ALL per-component
   minimums. Record the runner-up and the cost differential in the report.

4. **Do not publish a winner off an unvalidated judge.** A generator faithfulness
   number is only a claim once the judge clears `cohens_kappa_vs_human_min: 0.60`
   on a non-empty `boundary_audit.jsonl`. If kappa cannot be computed this session,
   the generator bakeoff is PROVISIONAL — label it so; do not amend ADR 004 yet.

5. **Curation discipline.** Every committed gold/negative/boundary row is
   `verified: true` only after the curator confirms it (watch the clip for spans;
   confirm the refusal target for negatives). Draft in gitignored `*.draft.jsonl`,
   then promote. Run `make validate-evals-strict` before commit.

6. **Schema first.** If a metric or row needs a new field, update `eval/schemas/`,
   add a `valid_*`/`invalid_*` fixture pair under `eval/tests/fixtures/`, run
   `make validate-self-test` BEFORE touching consumer code. (Adding negative +
   boundary_audit rows needs NO schema change — those shapes already exist.)

7. **All measurements land in `eval_runs` + `eval_results`.** Aggregate in
   `eval_runs.summary`; per-example in `eval_results.{system_output, metrics}`.
   Single transaction per run, same atomicity contract as bakeoff #1.

8. **Judge prompt is versioned + content-hashed.** It lives under
   `eval/runners/judges/`; its sha is logged in `summary.json` and
   `per_example.jsonl` for every run (Judge Versioning, experiment-tracking).

9. **DeepInfra is primary, OpenRouter optional failover.** Adding any other
   provider requires an ADR-004 amendment. Do not add one quietly.

10. **One ML/process discipline.** The generator + judge calls are hosted (no local
    model load). If any step loads a local model (e.g. the reranker in the answer
    loop), do not run two local ML-loading processes concurrently.

11. **Keep unrelated churn out of commits.** Explicit `git add <paths>`, never
    `git add -A`. The pre-existing dirty tree (`.claude/*`, `.cursor/*`, the
    formatter churn across `test_*.py`, `retrieve/visual.py`, `embed/colqwen.py`,
    etc.) stays unstaged. The `generate/` + `providers.py` WIP is IN scope — triage
    it explicitly, do not sweep it.

12. **Conventional commits, no AI attribution.** `feat(eval):`, `eval:`, `docs:`.
    Gates green at every commit boundary. Postgres stopped before final.
</hard_rules>

<mission_stack>
Sequence the work so the scaffolding + curation land before the bakeoff, and so no
generator winner is published off an unvalidated judge. Use TeamCreate for the
independent strands (WIP triage, curation, judge prompt) and keep the runner build
+ final selection in the main context.

== Mission 1 — Triage the uncommitted answer-loop WIP ==

Diff `generate/*`, `eval/runners/providers.py`, `eval/runners/minimal_generation.py`
vs HEAD. For each: is the change correct, tested, and consistent with the LangGraph
node contract? Decide commit / revise / revert per file. Confirm `make answer
QUERY="..." GENERATOR=<id> JUDGE=<id>` runs end-to-end against DeepInfra on ONE
dev_gold query before going further. If the loop is broken, fixing it is Mission 1,
not a side quest. Commit the triaged loop in small chunks with explicit pathspecs.

== Mission 2 — Curate the missing corpus pieces ==

Decide scope with the user if needed (see working_agreements), then:
- `negative.jsonl`: curate enough corpus-scope negatives to measure
  `corpus_negative_refusal` meaningfully (target ≥10; methodology v0 target ~20).
  Curate with one model family; the judge will be a different family.
- `boundary_audit.jsonl`: curate enough judge-vs-human rows to compute Cohen's
  kappa for the chosen chunking strategy (target ≥20; kappa < 0.6 voids the judge).
  This is the gate on publishing any faithfulness claim.
- Draft → watch/confirm → `verified: true` → promote → `make validate-evals-strict`.

== Mission 3 — Write the versioned judge prompt ==

Author the cross-family judge prompt under `eval/runners/judges/<name>.md`. It
scores ClaimsSupported (claim-level grounding in retrieved spans) and
CitationAccuracy (cited timestamp actually contains the claim). Content-hash it;
wire the hash into the run summary. The judge family must differ from every
generator candidate it will score (rotate per generator, or fix one cross-family
judge and exclude its own generator option — justify the choice in the report).

== Mission 4 — Build the generator-bakeoff runner ==

New runner under `eval/runners/` (mirror `run_embeddings_bakeoff.py` +
`measure_embeddings.py`): for each generator candidate, run the answer loop over
dev_gold, capture the answer + citations + JSON parse success + p95 latency, score
faithfulness with the cross-family judge, and score abstention on the negatives.
Persist one `eval_runs` row + per-example `eval_results` per candidate in a single
transaction. Add fast tests with injected provider/judge stubs (no live API in
`make test`).

== Mission 5 — Compute kappa, select, report ==

- Compute judge↔human Cohen's kappa on `boundary_audit.jsonl`. If ≥0.60, the judge
  is valid; if <0.60, the faithfulness numbers are provisional — say so and stop
  short of an ADR amendment.
- Apply the ADR-004 selection rule: cheapest generator within 3pp of the leader
  meeting all minimums. Record runner-up + cost differential.
- Write `eval/reports/<date>_generator_bakeoff/{summary.json,per_example.jsonl,methodology.mdx}`
  in the bakeoff-#1 format. Amend ADR 004 with the result ONLY if the judge is
  validated.

== Acceptance criteria ==

- Answer-loop WIP triaged (committed / reverted, documented per file); `make answer`
  runs end-to-end on a dev_gold query.
- `negative.jsonl` and `boundary_audit.jsonl` non-empty, all rows `verified: true`,
  `make validate-evals-strict` green.
- A content-hashed cross-family judge prompt under `eval/runners/judges/`.
- A generator-bakeoff runner with fast stub tests; one real run persisted to
  `eval_runs` + `eval_results`.
- Judge kappa computed; generator selected per ADR-004 with runner-up + cost; report
  written. ADR amended only if kappa ≥ 0.60.
- All gates green; Postgres stopped.

== Out of scope ==

- Visual retrieval, the ColQwen loader, re-ingest, the VAG audit (all closed).
- `test_gold.jsonl` (locked; final writeup only).
- Chunking ablation, Phase B reranker, corpus expansion beyond the negatives +
  boundary rows this bakeoff needs.
- A second inference provider beyond DeepInfra (needs an ADR amendment).
</mission_stack>

<required_gates>
Before any commit and before final:

```bash
make validate-evals-strict     # after any corpus change
make phase0-gate
make test                      # includes the new generator-bakeoff stub tests
make lint
```

If you change `eval/runners/` production code, re-run the relevant slow test for
that runner. Then stop Postgres:

```bash
make db-down
docker compose ps postgres
```
</required_gates>

<working_agreements>
- Auto-mode default-to-action. Use AskUserQuestion only for a genuine scope tradeoff
  the user must own — e.g. "curate a full ≥20-row boundary_audit this session
  (slow, but unblocks a validated judge), or run a PROVISIONAL generator bakeoff
  now and defer kappa-validated judge selection to a follow-up?" That curation-vs-
  provisional decision is the likely fork; surface it early, do not guess.
- TeamCreate for the independent strands (Mission 1 triage, Mission 2 curation,
  Mission 3 judge prompt). Keep Missions 4–5 (runner + selection) in main context.
- Read before edit. The visual arc is verified — do not re-open it.
- Honesty over a clean number. An unvalidated judge means provisional results;
  label them. A generator winner chosen off 10 dev_gold rows is a small-n result;
  report n and the cost differential, not a triumphant headline.
- Final response must include: WIP disposition per file, corpus rows added (counts),
  judge prompt + its sha, the generator-bakeoff run id, judge kappa (or why it
  could not be computed), the selected generator + runner-up + cost differential
  (or "provisional, no selection"), gates run, and Postgres-stopped confirmation.
</working_agreements>

<execution_rhythm>
1. `<first_actions>` 1–11. Surface any delta from `<repo_state>`.
2. Mission 1: triage + commit the answer-loop WIP; prove `make answer` runs.
3. Decide the curation-vs-provisional fork with the user (working_agreements).
4. Missions 2 + 3 in parallel (curation + judge prompt) via TeamCreate.
5. Mission 4: build the runner + stub tests; `make test`.
6. Gates: `make validate-evals-strict` + `make phase0-gate` + `make lint`.
7. Mission 5: one real run, kappa, selection, report, ADR amendment (only if valid).
8. Commit in small chunks with explicit pathspecs. Re-run gates. `make db-down`.
9. STOP at the actionability-or-explanation boundary. Report state.
</execution_rhythm>

<instructions>
Execute `<first_actions>`. Then drive `<mission_stack>` in order with the
`<execution_rhythm>` and `<working_agreements>` constraints.

The session-close deliverable is one of:

- **Selected.** WIP triaged; negatives + boundary_audit curated; cross-family judge
  written and kappa ≥ 0.60; generator bakeoff run; winner selected per ADR-004 with
  runner-up + cost; report written; ADR 004 amended. ← preferred.
- **Provisional.** WIP triaged; runner built and one generator bakeoff run; but the
  judge is not yet kappa-validated (boundary_audit under-curated this session).
  Report the provisional numbers, name exactly what is missing, do NOT amend ADR
  004. ← acceptable.
- **Blocked.** A prerequisite is irreconcilable (no DeepInfra key; the answer loop
  is broken in a way that needs the user). Name the blocker and stop. ← acceptable.

Do NOT touch visual retrieval, the loader, or `test_gold.jsonl`. If a fact
contradicts the week-9 handoff or `model_candidates.yaml`, surface it explicitly
rather than working around it. Begin now.
</instructions>
