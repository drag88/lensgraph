# Handoff — Phase 2 Week 6 (Bakeoff #1: text embeddings)

**Opened:** 2026-05-27 (phase-1 week-5 closed at `8d6694a`)
**Status:** 🟢 Phase 1 shipped end-to-end (LangGraph loop, live `make answer` EXIT GATE passed, 11/11 step-35 sweep rows in `eval_runs`). Phase 2 week 6 is the **embeddings bakeoff** — the first of three sequential bakeoffs per ADR 004 v3.1.
**Authoritative design:** `docs/phase-1-design.md` rev 4 (commit `e950583`) — still the contract; only the bakeoff sequencing is phase-2 scope.
**Prior handoff:** `dev/active/phase-1-week-5/handoff.md` (CLOSED week 5; left in place as a reference for next phase-1 backfill work).

---

## TL;DR

BGE-M3 was ingested across all three channels (dense / sparse / multi-vector). Bakeoff #1 measures **TimestampRecall@5** per channel on `dev_gold.jsonl` against the candidate's ADR 004 minimum (`>= 0.75`). The candidate `bge-m3-all-channels` is the only candidate we have a complete substrate for; Voyage and Gemini are fallbacks gated on BGE-M3 failing the minimum. So week 6 is effectively a **go/no-go** measurement — pass and we lock BGE-M3 as the winner; fail and we open an ADR amendment + re-ingest with Voyage.

The scaffold (`scripts/run_embeddings_bakeoff.py`) reads synthetic measurements today; phase 2 swaps in the real sweep, then calls `db.repos.eval_runs.lock_bakeoff_winner('text_embeddings', ...)`. The phase-1 step-35a RED test (`eval/tests/test_run_embeddings_bakeoff.py`) already pins the lock contract — keep it green.

---

## Where we are at HEAD

```
8d6694a docs(handoff): drop stale "code_path filter deferred" claim + correct text_embeddings read shape
a1fa1dd fix(generate,db): cite bounds-check answer_claim_index + load_bakeoff_winner code_path filter
a37f9b6 docs(handoff): close phase-1 week-5 — slice 3 shipped, EXIT GATE passed, step 35 unlocks phase 2
1768a32 fix(eval): refresh Qwen3-235B-A22B-Instruct provider_model_id to -2507 suffix
29f3e7c feat(scripts): embeddings-bakeoff scaffold (step 35a) + dev_gold sweep driver (step 35)
3b1fe41 feat(generate,db): LangGraph nodes + graph + answer() + traces repo
f5f097b feat(eval,generate,db): bakeoff harness + shared parser + eval_runs repo
```

### Gates green at HEAD

```
make test                  84 fast tests
make lint                  clean
make phase0-gate           OK (11 verified non-negative examples across 3 talks)
make validate-evals-strict 0 warnings
slow tests (slice-3 surfaces): 29 passed
```

### Substrate state (resumes across `make db-down`/`up` via volume `pgdata`)

- 3 ingested talks (`ai_engineering_v0`): nXafozNIk3c · aie_sg_2026_d2_arize_alyx · W_CYk2ogcDI.
- 212 chunks · 212 dense_embeds · 212 sparse_embeds · 73,621 chunk_token_embeds.
- frames / frame_patches empty (no `.mp4` files staged; visual channel returns `[]` gracefully — irrelevant for embeddings TR@5 measurement which is text-channel only by minimum definition).
- `eval_runs` / `eval_results` populated by phase-1 step-35 sweep + slow-test runs. Phase 2 will append `code_path='embeddings_bakeoff'` rows; query at session start (`SELECT count(*), code_path FROM eval_runs GROUP BY code_path`) — exact totals drift across runs.

---

## What's already wired for phase 2 to consume

| Surface | What it gives you |
|---|---|
| `scripts/run_embeddings_bakeoff.py` | Scaffold: reads measurements JSON → applies ADR 004 selection rule → writes one `eval_runs` row (`code_path='embeddings_bakeoff'`) + locks winner. Synthetic measurements today; phase 2 swaps them for real. |
| `eval/tests/test_run_embeddings_bakeoff.py` | Step-35a RED — asserts the script writes `winner_locked=true` + `component='text_embeddings'` + `winner_candidate_id='bge-m3-all-channels'`. Keep it green. |
| `db.repos.eval_runs.lock_bakeoff_winner(component='text_embeddings', candidate_id='bge-m3-all-channels', run_id=...)` | The write call. Idempotent. |
| `db.repos.eval_runs.load_bakeoff_winner(component='text_embeddings', code_path='embeddings_bakeoff')` | The read call. Production callers MUST pass `code_path` (per design §5; landed in `a1fa1dd`). No production reader exists today — phase 2's ingest tuning is where one gets added. |
| `retrieve.{dense,sparse,multivec}.retrieve(conn, query, top_k=...)` | The three text channels you measure. Already used by `generate/nodes/retrieve.py` in production. |
| `retrieve.rrf.fuse({...}, top_k=30)` | RRF fusion across channel rank-lists. Returns `FusedResult`s. Used by all retrieval consumers. |
| `eval/corpora/ai_engineering_v0/dev_gold.jsonl` | 10 verified single_clip examples (+ 1 in `synthesis.jsonl`). Each has `gold_spans=[{start_sec, end_sec, video_id?}]` — your TR@5 ground truth. |

---

## Mission stack

### Mission 1 — Real embeddings measurement runner (THIS is the bakeoff)

Replace the synthetic measurements in `scripts/run_embeddings_bakeoff.py` with a real per-channel `TimestampRecall@5` sweep against `dev_gold.jsonl`.

**TR@5 definition (design + ADR 004 + eval-methodology.md):** for an example with gold span `[s, e]` on `video_id=V`, the channel `passes@5` if any of the top-5 retrieved chunks satisfies `chunk.video_id == V` AND `(chunk.start_sec, chunk.end_sec)` overlaps `(s, e)` at all (open-interval intersection, not full containment). `TR@5 = passes / total`.

**Per-channel measurement:**
- `dense`: `retrieve.dense.retrieve(conn, q, top_k=5)`
- `sparse`: `retrieve.sparse.retrieve(conn, q, top_k=5)`
- `multivec`: `retrieve.multivec.retrieve(conn, q, top_k=5)` (top-100 dense+sparse prefilter inside)
- `rrf_4ch`: fuse `{bm25, dense, sparse, multivec}` then take top-5

Skip the visual channel — minimums are TEXT only for the embeddings bakeoff. Skip BM25 as a separate row; it's not a candidate for "text embeddings" winner, but include it inside the RRF row because that's what the production loop uses.

**Selection rule** (ADR 004 v3.1): `bge-m3-all-channels` passes iff `TR@5 >= 0.75` on **at least one channel** (the design's "vector channel only" minimum — interpret as the best of dense/sparse/multivec). If it passes, lock. If it fails, write the failing-row + open an ADR amendment proposing Voyage re-ingest. Do NOT silently swap in a fallback.

**Methodology writeup**: `eval/reports/<YYYY-MM-DD>_embeddings_bakeoff/methodology.mdx` per CLAUDE.md experiment-tracking rules. Per-channel TR@5 table, minimum-check verdict, cost differential vs Voyage/Gemini (even though we're not running them — show the numbers you saved). Hash the candidate yaml + log in `summary.json`.

### Mission 2 — IF BGE-M3 fails minimums (contingency)

ADR 004 §"Revisit when": "BGE-M3 fails the embeddings minimum → promote Voyage / Gemini Embedding 2 from fallback to primary." That's an ADR amendment, not a silent swap. Process:

1. Commit the failing measurement first (so the writeup has the actual numbers).
2. Open `docs/decisions/004-model-selection.md` amendment "v3.2: Voyage promoted to primary after BGE-M3 failed embeddings minimum (TR@5=<X> < 0.75)".
3. Plumb Voyage into `embed/` (new `embed/voyage.py` module + yaml provider_model_id update).
4. Re-ingest all 3 talks via Voyage; re-run mission 1.

Expect this NOT to be needed — BGE-M3's reported public benchmarks comfortably clear 0.75. But the path exists.

### Mission 3 — Bakeoff #2 prep (generator + judge)

Only after bakeoff #1 locks. Will be design-driven later; not week-6 scope.

---

## Hard rules (carried forward from phase 1)

1. **Eval before code.** Bakeoff measurement code lands ONLY in `scripts/` and `eval/runners/`. No tweaks to `retrieve/`, `embed/`, or `chunking/` for "bakeoff convenience" — the bakeoff measures the production stack as-is.
2. **Postgres only.** All measurements land in `eval_runs` + `eval_results`. No external observability backend.
3. **No silent model defaults.** If you write a load path, it MUST pass `code_path` (e.g., `load_bakeoff_winner(component='text_embeddings', code_path='embeddings_bakeoff')`). Phase-3 `langgraph_loop` locks must not leak in.
4. **Cross-family judge discipline.** Doesn't apply to embeddings bakeoff (no judge) but reminds the reader: bakeoff #2 will need it.
5. **Locked `test_gold.jsonl`.** Measurements run on `dev_gold.jsonl` ONLY. `test_gold.jsonl` is reserved for the final writeup, untouched.
6. **All model + provider IDs from yaml.** Wire IDs (`BAAI/bge-m3`, etc.) live in `eval/config/model_candidates.yaml`. Pre-commit grep guard from phase 1 still applies (zero wire-ID matches in `generate/`, `eval/runners/minimal_generation.py`, `scripts/`, `db/repos/`).
7. **No ADR violations without an amendment** in the same commit.
8. **Conventional commits**: `eval(bakeoff): ...`, `feat(eval): ...`, `docs: ...`. No AI attribution.

---

## First commands the phase-2 session MUST run

```bash
git log --oneline -10
# Confirm 8d6694a, a1fa1dd, a37f9b6 at the top.

make validate-evals-strict
make phase0-gate
make test
make lint
# All four must pass (84 fast tests).

make db-up
make db-migrate
```

Then:

1. Read `docs/decisions/004-model-selection.md` v3.1 — selection rule + minimums table.
2. Read `docs/eval-methodology.md` — TR@5 definition + dataset discipline.
3. Read `scripts/run_embeddings_bakeoff.py` — the scaffold you'll extend.
4. Read `eval/tests/test_run_embeddings_bakeoff.py` — the contract you must keep green.
5. Run a sanity-check retrieval on one dev_gold example to confirm the channels work warm:
   ```python
   from db.conn import dsn; from retrieve import dense, sparse, multivec
   import psycopg
   with psycopg.connect(dsn()) as conn:
       q = "how does Tengyu Ma compare long context, fine-tuning, and RAG"
       print("dense top-5:", [r.chunk_id for r in dense.retrieve(conn, q, top_k=5)])
       print("sparse top-5:", [r.chunk_id for r in sparse.retrieve(conn, q, top_k=5)])
       print("multivec top-5:", [r.chunk_id for r in multivec.retrieve(conn, q, top_k=5)])
   ```

Only after those five steps: start writing the real measurement code.

---

## Cost ceiling

Bakeoff #1 is **$0** (BGE-M3 is local). The full 12-week ADR 004 budget ($18–33) is for bakeoff #2 (generator+judge) + premium triangulation. Bakeoff #1 burns local MPS time only.

If BGE-M3 fails minimums and Voyage re-ingest is needed: ~$5–15 against Voyage's API depending on corpus growth. Budget accordingly + amend ADR.

---

## What's deferred from phase 1 (still NOT phase-2 week-6 scope)

- `eval/runners/judges/` versioned prompts — phase-2 week 7 (generator+judge bakeoff).
- Async LangGraph (`ainvoke`) — phase 4 streaming.
- OpenRouter failover wiring — phase 4 hardening.
- `retrieve/api.py` public surface — phase-4 UI work.

---

## Known small-but-real items the phase-2 session may want to fold in opportunistically

- `embed/colqwen.py::encode_image_pooled`'s zero-patch defensive comment is still wrong ("HNSW NULL-pooled WHERE clause filter it downstream" — a zero vector is not NULL). Fix only if a phase-2 reason brings you into the file.
- The `make answer` first-call cold-start cost (~30s ColQwen load + ~10s BGE-M3 load) is felt every fresh process. Phase-4 perf work, not phase-2.
- No production `load_bakeoff_winner` reader for `text_embeddings` exists today (ingest doesn't auto-resolve the active embedder). Adding one is reasonable phase-2 scope if you find yourself wanting to gate ingest on the locked winner.

---

## Where to push back

Same posture as phase 1: do not silently bend the design. If the bakeoff surfaces a fact that contradicts ADR 004 (e.g., BGE-M3 multivec TR@5 destroys the budget), surface it explicitly, propose an ADR amendment in the same commit, and wait for the user's "approve amendment" signal. The 4-revision design + 3-plan-review history is the contract; deviations move through the same process.
