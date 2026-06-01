---
description: Workflow rules, subagent strategy, verification gates, self-improvement loop
---

# Workflow

## Tool and parallel call policy

Spawn multiple subagents in the same turn when fanning out across items, reading multiple files, or running independent investigations. Skip fan-out for single-file edits or trivial reads. For multi-agent debate or implementation work, use TeamCreate + named teammates rather than ad-hoc subagents.

<use_parallel_tool_calls>
For maximum efficiency, whenever you perform multiple independent operations,
invoke all relevant tools simultaneously rather than sequentially.
</use_parallel_tool_calls>

---

## Subagent Strategy (Primary)

**Default to subagents. Main context is precious.**

The main conversation thread is your working memory. Every file read, every exploratory grep, every tangential research question consumes it permanently. Subagents work in isolated contexts and return only the result you need.

### When to Spawn a Subagent

| Task Type | Spawn? | Reason |
|-----------|--------|--------|
| Exploring an unfamiliar module | Yes | Returns a summary, not raw file contents |
| Researching a library/API | Yes | Returns verdict + key facts, not entire docs |
| Parallel analysis (multiple files/dirs) | Yes | Multiple agents, simultaneous |
| Tracing a bug across files | Yes | Agent can read 10+ files without polluting main context |
| Confirming a fact you already know | No | Just do it |
| A single targeted grep | No | Faster inline |
| Writing/editing a file | No | Must stay in main context |
| Simple single-file read | No | Faster inline |

### Dispatch Tiers

Classify the task by shape before choosing a workflow.

| Tier | Use | CDF route |
|------|-----|-----------|
| Simple | Single-file or obvious change | Direct edit or `/cdf:implement` |
| Medium | Multi-file change with known approach | `/cdf:task` for scoped breakdown, then implement in main context |
| Investigate | Bug, regression, unexplained failure | `/cdf:troubleshoot`, with codebase-navigator for multi-file tracing |
| Review | Quality, security, performance, architecture risk | `/cdf:analyze`, or `/cdf:task` with role framing when no real agent exists |
| Plan | Shape a feature before building | `/cdf:brainstorm`, `/cdf:design`, `/cdf:plan-review`, then `/cdf:approve` |
| Ship | Release execution | `/cdf:verify --mode pre-pr`, then `/cdf:ship` |

Do not recreate `/cdf:flow` or `/cdf:workflow`; Opus 4.7 handles full lifecycle plans from a clear prompt with `xhigh` effort.

### How to Spawn Well

One task per subagent. Not "analyze this module and also the one it depends on."

Give each agent:
1. A single, atomic goal
2. The specific files or directories to focus on
3. What to return (a summary, a verdict, a list — not raw file dumps)

For complex problems, throw more compute at it: spawn 3-5 agents in parallel, each covering a different angle.

### Project-Specific Spawn Patterns

**LensGraph-specific patterns** (ML / data-evaluation):
- **Schema impact tracing.** Spawn an agent to find every consumer of a given schema (`gold_example`, `talk`, `boundary_audit`, `model_candidates`) and report which call sites need updating when a field changes. Return: file:line list grouped by directory.
- **ADR conflict scan.** Spawn an agent to walk `docs/decisions/*.md` and report whether a proposed change contradicts a current ADR. Return: ADR number + the specific clause violated, or "no conflict."
- **Eval methodology cross-check.** Spawn an agent to read `docs/eval-methodology.md`, `docs/PRD.md`, and the latest `eval/reports/<date>_*/methodology.mdx` to verify a planned change does not break the anti-contamination rules (cross-family judge, locked test set, no retrieval tweaks per chunking run).
- **Corpus + fixture audit.** Spawn parallel agents — one to validate every `eval/corpora/*/talks.yaml` against schema, one to walk `eval/tests/fixtures/` and surface any invalid fixture that no longer asserts a real schema rule.

**Generic ML/data-science patterns** (use when LensGraph-specific does not fit):
- Spawn agent to trace a data pipeline from source to model input, identifying transformation steps.
- Spawn agent to analyze experiment tracking setup and compare recent run configurations.
- Spawn agent to review data validation and schema enforcement patterns.

---

## Plan Mode Default

Enter plan mode before acting on non-trivial tasks (3+ steps, or any change touching more than 2 files, or any schema change, or any commit that touches `ingest/chunking/retrieve/generate/api/web`).

What "plan mode" means here:
- Write the approach before writing any code
- Identify the 2-3 most likely failure points (anti-contamination rules are a common one)
- Get confirmation if the plan involves schema changes, ADR amendments, modifying `test_gold.jsonl`, or adding a new datastore / agent framework / inference provider

Skip plan mode for: single-file fixes, typo corrections, running the validator, lint-only changes.

If something goes sideways, stop and re-plan immediately rather than pushing through.

---

## Task Management

For anything requiring more than ~5 steps:

1. **Plan First** — write plan with checkable items before starting
2. **Verify Plan** — check in before starting implementation
3. **Track Progress** — mark items complete as you go (use TaskCreate / TaskUpdate)
4. **Explain Changes** — high-level summary at each step
5. **Document Results** — add review section after completion
6. **Capture Lessons** — update auto-memory after corrections

---

## Self-Improvement Loop

When the user corrects an approach, a fact, or a code pattern, save the lesson to auto-memory immediately. Reserve `.claude/rules/` for human-curated, durable standards. Do not write new rules files there autonomously — they create rule sprawl and conflict with existing rules.

**What to save**: key decisions, corrections, debugging insights, project patterns, architectural notes.

**Where to save**: auto-memory at `~/.claude/projects/-Users-aswinsreenivas-1-Code-1-1-personal-lensgraph/memory/`. Use Write / Edit to update `MEMORY.md` and topic files like `feedback_*.md`, `project_*.md`.

**Session continuity** — at session start, recent git history and auto-memory inject as context. No manual file management needed.

---

## Verification Before Done

Run a concrete verification step before marking a task complete. Paste the output of the verification command into the conversation. Opus 4.7 will quietly skip vague checks but executes concrete commands.

Project-specific verification gates:

- Schema or fixture change → `make validate-self-test` and paste output.
- Corpus change → `make validate-evals-strict` and paste output.
- Anything touching `ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, `web/` → `make phase0-gate` and paste output.
- Code change → `make lint` and paste output.

"It should work" is not verification. Pasted command output is.

---

## Demand Elegance (Balanced)

For non-trivial changes (more than ~20 lines, or a new function/class), pause and ask: **"Is there a more elegant way?"**

Elegance criteria:
- Fewer moving parts
- Reuses existing patterns in this codebase (single-file validator, plain dicts, JSON Schema)
- A future maintainer would not be confused

For LensGraph specifically: prefer extending `eval/validate.py` over starting a second validator. Prefer adding a fixture pair over writing a Python-only assertion. Prefer an ADR amendment over a quiet code change.

Skip this for simple, obvious fixes — don't over-engineer.

---

## Autonomous Bug Fixing

Given a bug report: fix it. Don't ask for hand-holding.

Standard approach:
1. Point at logs, errors, failing tests — then resolve them
2. Identify root cause (not just symptom)
3. Fix at root cause
4. Add a regression test (fixture pair for schema bugs, pytest case for code bugs)
5. Verify the fix

Zero context switching required from the user.

---

## Core Principles

- **Eval before code.** Every change is judged against whether it improves the eval, improves the metrics, or improves methodology rigor. Implementation code is locked behind the phase-0 gate by design.
- **Schema-first, code-second.** Update the schema and fixtures before the code.
- **Cross-family judge discipline.** Never the same family for curation and judging.
- **Lock the test set.** Treat its SHA256 as a contract.
- **One framework, one datastore.** Complexity is the budget killer for a solo 12-week project.
- **Report everything, filter later.** For review or coverage tasks, surface all findings (low-severity, uncertain, edge cases) with confidence and severity. Do not self-filter before reporting — Opus 4.7's literalism causes it to drop real issues when asked to "be conservative."
