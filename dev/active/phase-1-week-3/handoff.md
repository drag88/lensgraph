# Handoff — Phase 1, Week 3, Step 1 (substrate)

**Written:** 2026-05-24, end of design-finalization session
**Read by:** the next session, before any implementation
**Authoritative design:** `docs/phase-1-design.md` rev 4, committed `e950583`

---

## Where we are

Two missions complete:

- **Mission 1.** Phase-0 gate flipped RED → GREEN. 11 verified gold examples (10 single_clip + 1 synthesis) across 3 talks. Commits `5dadc00` (gate green) and `b805f16` (review findings closed).
- **Mission 2 design phase.** Phase-1 architecture spec at `docs/phase-1-design.md` is stable at rev 4 after three `/cdf:plan-review` passes. Commits `0b6a2f6` (rev 2, LangGraph re-scoped into v1), `b7afc4e` (rev 3, no-silent-defaults tightened across all three model-using nodes), `e950583` (rev 4, third-pass amendments).

**Implementation has not started.** No code lives in `ingest/`, `chunking/`, `retrieve/`, `generate/`, `embed/`, `db/`, `queues/`, `api/`, `web/`.

**Gates green at HEAD (`e950583`):**

```
make validate-evals-strict   OK (0 warnings; 11 verified examples)
make phase0-gate             OK (11/3 across talks)
make test                    OK (7/7 fast tests pass)
make lint                    OK
```

---

## First commands the new session MUST run

Verify state matches this handoff before anything else:

```bash
git log --oneline -5
# Top should be: e950583 docs(phase-1): close third-pass amendments

make phase0-gate
# Must report "PHASE-0 GATE OK: 11 verified non-negative examples across 3 distinct talks"

make test && make lint
# Both must pass
```

If any of those is red, STOP and surface the delta. Do not proceed.

---

## Reading order (read before writing any code)

Per CLAUDE.md hard rule 1 and the design's discipline:

1. `docs/phase-1-design.md` — entire doc, ~1200 lines. The build sequence in §8 is the master TDD ordering.
2. `docs/roadmap.md` — for phase-1/phase-2/phase-3 boundaries.
3. `docs/decisions/004-model-selection.md` — ADR 004 v3.1 selection rule and the no-silent-defaults principle.
4. `docs/decisions/002-postgres-only.md` — why Dockerfile.postgres layers pgmq + pgvector instead of using a second store.
5. `.claude/CLAUDE.md` — hard rules; the one about "no code in ingest/chunking/retrieve/generate/embed/db/queues until phase0-gate passes" has FIRED (gate is green) so substrate work is now unlocked.
6. `eval/config/model_candidates.yaml` — source of truth for candidate IDs that GENERATOR/JUDGE/PLANNER values must match.

Skim, don't deep-read: the curation files in `eval/corpora/ai_engineering_v0/` and `eval/curation/`. They are stable.

---

## Immediate next action: week-3 step 1 (substrate)

Per design §8, step 1 ships the substrate that makes step 2's RED migration tests runnable. Strict scope — substrate ONLY, no Python code in `db/` yet.

### Files to create

1. **`Dockerfile.postgres`** (repo root). Per design §7:

   ```dockerfile
   FROM pgvector/pgvector:pg16
   RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-pgmq \
    && rm -rf /var/lib/apt/lists/*
   ```

   Fallback if `postgresql-16-pgmq` is not in PGDG apt at build time: use `quay.io/tembo/pg17-pgmq` base and verify pgvector ≥ 0.7 with `psql -c "\dx"` after launch. Documented in §7.

2. **`docker-compose.yml`** (repo root). Per design §7 — build from `Dockerfile.postgres`, expose 5432, env user/pass/db = `lensgraph`/`lensgraph`/`lensgraph`, named volume `pgdata`, healthcheck via `pg_isready`. Exact YAML in §7.

3. **`Makefile` additions** — append per design §7:

   ```
   db-up            docker compose up -d postgres
   db-down          docker compose stop postgres
   db-build         docker compose build postgres
   db-verify-ext    psql ... CREATE EXTENSION pgmq + CREATE EXTENSION vector
   db-migrate       (placeholder — points at db/migrate.py which lands in step 2)
   db-reset         drop + recreate + apply migrations
   logs             tail -f $${LENSGRAPH_LOG_PATH:-./.lensgraph/logs/workers.log}
   ```

   Do NOT add `ingest`, `workers-*`, `retrieve-test`, `answer`, `replay-dlq`, `bakeoff-prep`, `models-warm` yet — those ship in their respective weeks.

4. **`pyproject.toml` additions**:
   - `psycopg[binary,pool]>=3.2` (step 2 will use it)
   - `pydantic>=2.7` (for `AgentState`, `RetrievedChunk` BaseModel surface; lands across step 2-onwards but the dep ships now so model libraries don't sneak in later)
   - NO model libraries (`torch`, `sentence-transformers`, `colpali-engine`, `whisperx`, `transformers`) — those belong in week 4-5.
   - Pytest markers block (already has `[tool.pytest.ini_options]`):
     ```toml
     markers = ["slow: requires model load or live Postgres"]
     addopts = "-m 'not slow'"
     ```

5. **`eval/tests/test_substrate.py`** — single sanity test:

   ```python
   """Substrate sanity. Verifies pytest config + marker registration."""
   import pytest

   def test_pytest_slow_marker_registered(pytestconfig):
       """The 'slow' marker MUST be registered so @pytest.mark.slow doesn't
       emit PytestUnknownMarkWarning in subsequent test files."""
       markers = pytestconfig.getini("markers")
       assert any(m.startswith("slow:") for m in markers), (
           f"slow marker not registered; got: {markers}"
       )
   ```

   RED first: write the test before the marker entry exists in pyproject.toml; assert it fails with the marker absent. GREEN: add the marker.

6. **`.gitignore`** — append `.lensgraph/` (log directory will land here when workers start in step 3+).

### Build-time verification gate (per design §8 step 2)

These commands must all succeed before any step-2 RED test lands:

```bash
make db-build
make db-up
# wait for healthcheck
make db-verify-ext
# Both CREATE EXTENSION pgmq and CREATE EXTENSION vector must succeed.
# Failure → swap to Tembo image per §7 fallback; reverify with \dx.
make db-down
```

### Atomic commits (suggested split)

1. `feat(infra): postgres image + compose for pgvector+pgmq` — `Dockerfile.postgres` + `docker-compose.yml` + `.gitignore` addition.
2. `feat(make): db lifecycle targets` — Makefile additions for `db-build`, `db-up`, `db-down`, `db-verify-ext`, `db-migrate` (placeholder), `db-reset`, `logs`.
3. `feat(deps): psycopg + pydantic + pytest slow marker` — `pyproject.toml` additions + `eval/tests/test_substrate.py`.

All three commits must keep `make phase0-gate`, `make test`, `make lint` green.

### Stop after step 1

Do NOT start step 2 (RED migration tests + GREEN `db/migrate.py` + the 5 SQL migrations) without explicit user signal. Step 2 is meaningfully different scope (Python code, schema design, the GREEN-blocking extension verification gate). The user will say "proceed to step 2."

---

## Environment notes / known friction

- **Docker Desktop is installed at `/Applications/Docker.app`** but as of the end of this session, the daemon was not running and `/usr/local/bin/docker` was not symlinked. First-time launch of Docker Desktop should:
  - Prompt for admin password (creates the `/usr/local/bin/docker` symlink).
  - Start the engine (~30 sec).
  - After which `docker --version`, `docker info`, `docker compose version` all work from any shell.
- Build time for `make db-build` first run: ~3 min (pulls `pgvector/pgvector:pg16` base ~150 MB + apt-get install pgmq ~few MB).
- Postgres data persists in the named volume `pgdata`; `make db-reset` is the only path that drops it.

---

## Hard rules to re-respect (do not violate, even silently)

From `.claude/CLAUDE.md`, `docs/decisions/*.md`, and the design doc's binding constraints:

1. **No silent model defaults.** Pre-bakeoff, `make answer` (when it lands in week 5) MUST require explicit `GENERATOR=...` and `JUDGE=...`. Substrate work in step 1 doesn't touch this directly but don't accidentally introduce a fallback that violates it later.
2. **Postgres only.** No Redis, Celery, RabbitMQ, second vector store. Step 1's Dockerfile is the only container; do not add other services to `docker-compose.yml`.
3. **`make phase0-gate` must stay green at every commit boundary.** Re-run before each commit.
4. **`test_gold.jsonl` is locked.** Do not touch.
5. **Conventional commits.** `feat(scope):`, `fix(scope):`, `docs:`, `eval:`, `chore:`. No AI attribution lines.
6. **No new framework / datastore / provider / model without an ADR amendment first.**

---

## What this handoff does NOT cover

- Step 2 (migration tests + `db/migrate.py` + the 5 migrations) — picked up after step 1 lands.
- Step 3+ (talks repo, PGMQ wrapper, ingest fetch) — sequenced per design §8.
- Phase-2 bakeoff infrastructure — phase 1 only ships the `scripts/run_embeddings_bakeoff.py` scaffold (§8 step 35a), not the actual bakeoff run.

If you find yourself wanting to extend scope, stop and ask. The design is what was paid-for-in-blood across 3 plan-review passes; deviations need explicit user signal.

---

## Quick mental model for the new session

You are picking up `e950583` — a stable design doc, an empty implementation tree, and a green phase-0 gate. Your job is to ship `Dockerfile.postgres` + `docker-compose.yml` + Makefile additions + pyproject deps + a one-test pytest substrate, then run `make db-build && db-verify-ext` to confirm the GREEN-blocking gate. Three commits, no Python in `db/` yet, stop after.
