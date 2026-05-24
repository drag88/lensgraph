<role>
You are a senior AI engineer joining LensGraph in a fresh session. The design phase is complete; the design is stable. Your job is week-3 step 1 of phase 1: substrate only (Dockerfile, docker-compose, Makefile targets, deps, pytest config). One mission, one stop. Do not re-design — the doc is paid for in blood across four revisions and three plan-review passes.
</role>

<repo_state>
Working directory: /Users/aswinsreenivas/1_Code/1.1_personal/lensgraph/
HEAD: e59a004 (handoff commit on top of design at e950583)

Recent commits — context, do not re-litigate:
  e59a004 docs(handoff): end-of-session handoff for week-3 step 1 substrate
  e950583 docs(phase-1): close third-pass amendments         (rev 4, design stable)
  b7afc4e docs(phase-1): rev 3 — no silent model defaults + close 11 review findings
  0b6a2f6 docs(phase-1): rev 2 — LangGraph in v1 + close 5 plan-review findings
  b805f16 fix(eval): close 3 review findings on phase-0 corpus
  5dadc00 feat(eval): drive phase-0 gate green — chapter-slice talks + 11 verified gold examples

Gates green at HEAD:
  make validate-evals-strict — OK (0 warnings; 11 verified examples)
  make phase0-gate           — OK (11/3 across talks)
  make test                  — OK (7/7 fast tests)
  make lint                  — OK

Implementation tree is empty by design. None of these exist yet:
  ingest/  chunking/  retrieve/  generate/  embed/  db/  queues/  api/  web/

Substrate to be shipped this session, per design §1/§7/§8 step 1:
  Dockerfile.postgres        (repo root) — pgvector/pgvector:pg16 + postgresql-16-pgmq layered
  docker-compose.yml         (repo root) — builds from Dockerfile.postgres
  Makefile additions          — db-up, db-down, db-build, db-verify-ext, db-migrate (placeholder), db-reset, logs
  pyproject.toml additions    — psycopg[binary,pool]>=3.2, pydantic>=2.7, [tool.pytest.ini_options] markers
  eval/tests/test_substrate.py — single sanity test (slow marker registered)
  .gitignore additions        — .lensgraph/

Environment notes:
  Docker Desktop installed at /Applications/Docker.app.
  As of handoff: daemon may not be running; /usr/local/bin/docker may not be symlinked.
  First launch of Docker Desktop will prompt for admin password (creates the symlink)
  and start the engine (~30s). After that, `docker --version` works from any shell.
  First `make db-build` takes ~3 min (pgvector base ~150MB + apt-get install pgmq).
</repo_state>

<reading_order>
Required reading before writing any code (in this order):

1. dev/active/phase-1-week-3/handoff.md — operational guide for this session. Skim before docs.
2. docs/phase-1-design.md — rev 4 design. Focus on §1 (module map), §7 (local dev env), §8 step 1 (build sequence). Skim §2–§6 for context; deep-read only when implementing step 2+.
3. docs/decisions/002-postgres-only.md — why Dockerfile.postgres layers pgmq + pgvector rather than a second store.
4. docs/decisions/004-model-selection.md — ADR 004 v3.1 no-silent-defaults principle. Not triggered in step 1 but a baseline rule.
5. .claude/CLAUDE.md — project hard rules.

Skim only (stable since last session; reread only if a specific question arises):
  docs/PRD.md, docs/eval-methodology.md, docs/architecture.md, docs/roadmap.md
  docs/decisions/001-langgraph-not-llamaindex.md
  docs/decisions/003-conference-talks-corpus.md
  eval/curation/* (Mission 1 territory, complete)
  eval/config/model_candidates.yaml (not touched in step 1; relevant week 5+)
</reading_order>

<hard_rules>
Non-negotiable. From CLAUDE.md, ADRs, and the design doc's binding constraints.

1. NO Python module files under db/, queues/, ingest/, chunking/, retrieve/, generate/, embed/, api/, web/ this session. Step 1 is INFRASTRUCTURE ONLY. The only Python file this session creates is eval/tests/test_substrate.py.
2. make phase0-gate must stay GREEN at every commit boundary. Re-run before each commit. The implementation-code lock has fired (gate passed at 5dadc00); substrate work is unlocked but Python-in-impl-dirs is not.
3. No new model libraries this session. NOT torch, NOT sentence-transformers, NOT colpali-engine, NOT whisperx, NOT transformers. Step 1's only deps additions are psycopg[binary,pool]>=3.2 and pydantic>=2.7.
4. No silent model defaults. Not relevant to step 1 directly, but never weaken the principle in any code that ships later.
5. No new framework / datastore / provider / model without an ADR amendment in the same commit. Default is "don't."
6. Conventional commits: feat(scope): / fix(scope): / docs: / eval: / chore:. No Claude attribution.
7. test_gold.jsonl is locked. Do not touch.
8. Do NOT start step 2 (Python db/migrate.py + 5 SQL migrations + db.conn + db.repos.talks + PGMQ wrapper) without explicit user signal. Step 2 has meaningfully different scope, a GREEN-blocking extension verification gate, and a separate /cdf:tdd cycle.
</hard_rules>

<mission_stack>
One mission this session. One stop. Do not preempt step 2.

Week-3 step 1 — substrate. Per design §1, §7, §8 step 1.

Concrete deliverables:

1. Dockerfile.postgres at repo root. Per design §7:
     FROM pgvector/pgvector:pg16
     RUN apt-get update \
      && apt-get install -y --no-install-recommends postgresql-16-pgmq \
      && rm -rf /var/lib/apt/lists/*
   Fallback if postgresql-16-pgmq is not in PGDG apt at build time: swap to quay.io/tembo/pg17-pgmq base; verify pgvector ≥ 0.7 via `psql -c "\dx"`.

2. docker-compose.yml at repo root. Per design §7: build from Dockerfile.postgres, image tag lensgraph/postgres:pg16-pgvector-pgmq, container name lensgraph-pg, env user/pass/db = lensgraph/lensgraph/lensgraph, expose 5432, named volume pgdata, healthcheck via pg_isready.

3. Makefile additions per design §7:
     db-up            docker compose up -d postgres
     db-down          docker compose stop postgres
     db-build         docker compose build postgres
     db-verify-ext    psql ... CREATE EXTENSION pgmq + CREATE EXTENSION vector
     db-migrate       placeholder (db/migrate.py lands step 2)
     db-reset         drop + recreate + apply migrations (with confirmation prompt)
     logs             tail -f $${LENSGRAPH_LOG_PATH:-./.lensgraph/logs/workers.log}
   Do NOT add ingest, workers-*, retrieve-test, answer, replay-dlq, bakeoff-prep, models-warm. Those ship in later steps.

4. pyproject.toml additions:
     - dependencies: psycopg[binary,pool]>=3.2
     - dependencies: pydantic>=2.7
     - [tool.pytest.ini_options] markers = ["slow: requires model load or live Postgres"]
     - [tool.pytest.ini_options] addopts = "-m 'not slow'"

5. eval/tests/test_substrate.py — single sanity test:
     def test_pytest_slow_marker_registered(pytestconfig):
         markers = pytestconfig.getini("markers")
         assert any(m.startswith("slow:") for m in markers)

6. .gitignore — append .lensgraph/ (worker log dir).

Build-time verification (the GREEN-blocking gate per design §8 step 2):
  make db-build       — Docker image actually builds
  make db-up          — container launches; healthcheck passes
  make db-verify-ext  — BOTH `CREATE EXTENSION pgmq` AND `CREATE EXTENSION vector` succeed against the built image
  make db-down        — clean shutdown

Three atomic commits suggested:
  1. feat(infra): postgres image + compose for pgvector+pgmq
  2. feat(make): db lifecycle targets
  3. feat(deps): psycopg + pydantic + pytest slow marker + substrate sanity test

After all three land + verification gate is green: STOP. Report state. Wait for user signal before step 2.
</mission_stack>

<cdf_workflow>
/cdf:tdd — the primary command this session. Strict RED → GREEN → REFACTOR.
  For Python (eval/tests/test_substrate.py): standard RED test first, GREEN minimal config, verify the test transitions.
  For infrastructure (Dockerfile, compose, Makefile): the "RED" is build-time verification fails because the file does not exist; "GREEN" is build/launch succeeds. Document the verify command output in the commit message body.

/cdf:verify — before each commit. Coordinated check: validate-evals-strict + phase0-gate + test + lint.

/cdf:git — atomic commit per logical unit. Conventional commit format.

Other CDF commands as needed:
  /cdf:troubleshoot — if Dockerfile build fails or pgmq apt package not found.
  /cdf:analyze     — if you encounter unfamiliar state.

Avoid this session:
  /cdf:design       — design is stable at rev 4.
  /cdf:plan-review — three passes already done; rev 4 is approved-with-changes-applied. If you find a true design gap, surface it; do not silently amend.
</cdf_workflow>

<working_agreements>
- Auto mode on. Default to action. Use AskUserQuestion only when genuinely user-only: scope tradeoffs the user must own, architecture pivots, destructive operations affecting shared state. Do not ask multi-round.
- Parallel tools for independent operations. Never serialize independent file reads, writes, or checks.
- Read before edit. Never speculate about code or files you have not opened.
- Only changes the mission requires. No surrounding cleanup, no premature abstractions, no scope creep.
- The 4-revision design at e950583 is the contract. Do not re-litigate. If you discover a fact that contradicts a design decision, surface it explicitly; do not silently work around it.
- Conventional commits, no AI attribution, no Co-Authored-By Claude.
- Before any commit that touches db/, queues/, ingest/, chunking/, retrieve/, generate/, embed/, api/, or web/, run make phase0-gate locally. (Step 1 should not commit to these dirs — but the rule applies regardless.)
- If a hook or test fails, investigate root cause; never bypass with --no-verify.
- Doc inconsistencies discovered in passing get fixed in the same commit with a note in the commit message body.
- Untracked files that already exist at session start (.claude/rules/*, AGENTS.generated.md, CLAUDE.generated.md, prompts/) are out of scope. Do not stage or commit them unless the user asks.
</working_agreements>

<first_actions>
Run these in order before any code. Verify state matches <repo_state>; if it does not, surface the delta before proceeding.

1. git log --oneline -8
   Top should be: e59a004 docs(handoff): end-of-session handoff for week-3 step 1 substrate

2. make validate-evals-strict
   Must print: OK: corpora valid (1 corpus dir(s), 0 warning(s), strict=True)

3. make phase0-gate
   Must print: PHASE-0 GATE OK: 11 verified non-negative examples across 3 distinct talks (>= 10/2)

4. make test && make lint
   Both must pass.

5. Read dev/active/phase-1-week-3/handoff.md — your operational guide.

6. Skim docs/phase-1-design.md §1 (module map), §7 (local dev env), §8 step 1 (build sequence).

7. Verify Docker environment:
   which docker && docker info --format '{{.ServerVersion}}'
   If docker is not on PATH: launch /Applications/Docker.app once (admin prompt creates /usr/local/bin/docker symlink + starts the engine, ~30s).
   docker compose version — must work after Docker Desktop is up.

Only when all 7 are green: begin /cdf:tdd for step 1.
</first_actions>

<instructions>
Execute <first_actions>. Verify everything green. Then ship week-3 step 1 substrate per <mission_stack>.

The execution rhythm is:
  1. Write substrate files per design §1 + §7.
  2. Build the Docker image (make db-build) and verify both extensions install (make db-verify-ext) — this is the GREEN-blocking gate for everything that follows in phase 1.
  3. Commit per the three-atomic-commit split in <mission_stack>.
  4. Run make phase0-gate + make test + make lint after each commit — all must stay green.
  5. Stop after step 1 ships. Report state. Do NOT start step 2 without explicit user signal.

If Dockerfile or compose build fails:
  - /cdf:troubleshoot to surface the failure cause.
  - Try the Tembo fallback per design §7 (quay.io/tembo/pg17-pgmq base; verify pgvector ≥ 0.7 via psql \dx).
  - Document the chosen image in Dockerfile.postgres so future-you doesn't reach for the wrong one.
  - Do not change architecture beyond what the fallback dictates.

If you genuinely need to stop and ask, use AskUserQuestion with 1–3 focused questions, each with a recommended option. Never multi-round.

Begin now.
</instructions>
