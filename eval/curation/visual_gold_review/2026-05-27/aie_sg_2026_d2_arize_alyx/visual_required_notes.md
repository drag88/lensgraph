# Visual-Required Curation: aie_sg_2026_d2_arize_alyx

Curator: claude-opus-4-7
Date: 2026-05-27
Source talk chapter window: [516, 1494] of source m12vGjfbNlo, talk-relative [0, 978].

## Goal

Find examples where the answer is bearing on a slide and the transcript does NOT read it aloud verbatim. The existing 8 visual_gold rows are text-saturated — the 4-channel text RRF passes 8/8 — so visual lift is invisible. These three candidates pivot on slide-only content (concrete code identifiers, exact query strings, in-slide enumerations the speaker only paraphrases).

## Proposed rows

### 1. `visual-required-sally-current-plan-block` — HIGH confidence visual-required

- **Span:** chapter [314, 360] (source [830, 876]).
- **Slide:** "Plans Lives Outside Conversation History" — `agent_b_span2_source830_plan_outside_history.jpg` and `agent_b_span2_source864_plan_outside_history_alt.jpg`.
- **Slide-only content:** the literal Current Plan example with three items ("Sort LLM spans by latency", "Identify bottlenecks", "Suggest improvements"), the per-item status markers `[x]`, `[~]`, `[ ]`, the arrow annotation `← CURRENT`, and the exact call signature `todo_update(id=1, status="completed")`.
- **Why visual-required:** the transcript at lines ~995-1110 only paraphrases this — "it sees its current plan", "we're actually coaching Alex along with like when you're done, you know, call to-do update with the status completed when you finish this task." SallyAnn never names the three example tasks, never says `[x]`/`[~]`/`[ ]`, never speaks `id=1`. The text RRF cannot reach the answer.
- **Verified:** yes; both inspected frames show the same Current Plan block, identical content.

### 2. `visual-required-sally-jq-grepjson-queries` — HIGH confidence visual-required

- **Span:** chapter [546, 600] (source [1062, 1116]).
- **Slide:** "Small, composable tools == Infinite Context" — `agent_b_span3_source1066_composable_tools.jpg` and `agent_b_span3_source1102_composable_tools_alt.jpg`.
- **Slide-only content:** four exact query strings:
  - `jq '.experiments[0].rows[:5]'`
  - `jq '[.rows[] | select(.eval_score < 0.5)]'`
  - `jq '[.rows[].latency_ms] | add / length'`
  - `grep_json pattern="error"`
- **Why visual-required:** the transcript at lines ~1882-1907 only names the tool families ("Alex has access to two tools uh jq which is just like the same tool that you would use in your command line and GP JSON which is able to do reex search over serialized data"). She does not read the queries. The exact field names (`.experiments[0].rows`, `.eval_score`, `.latency_ms`), the threshold `0.5`, the slice `[:5]`, the `add / length` average idiom, and the `pattern="error"` parameter are slide-only.
- **Verified:** yes; both frames show all four queries.

### 3. `visual-required-sally-lessons-recoverable-exceptions` — MEDIUM confidence visual-required

- **Span:** chapter [590, 625] (source [1106, 1141]).
- **Slide:** "Lessons in context management" — `agent_b_span3_source1112_lessons_context_management.jpg`.
- **Slide-only content:** the exact CamelCase identifier `RecoverableExceptions` in bullet 4. Transcript at lines ~2106-2128 only paraphrases as "give good exceptions in your feedback loops" — the code-shaped identifier (which a Python/Java engineer would recognize as a named exception class) is slide-only.
- **Why visual-required (with caveat):** four of the five bullets ARE roughly spoken (token budgets, compress values, don't paper over with artificial limits, watch logs for customer data), so a text-only system could plausibly reconstruct most of the lesson list. The pivot is the exact identifier `RecoverableExceptions` (CamelCase, code-shaped) which is on-slide only. Medium confidence because a partial-credit text answer could score well unless the judge is strict about the exact identifier.
- **Verified:** yes; frame confirms the bullet list and CamelCase spelling.

## Confidence summary

| id | gold span (chapter) | pivot | text-cannot-answer confidence |
|---|---|---|---|
| `visual-required-sally-current-plan-block` | [314, 360] | Concrete 3-item plan + `[x]/[~]/[ ]` markers + `todo_update(id=1, status="completed")` call signature | **HIGH** |
| `visual-required-sally-jq-grepjson-queries` | [546, 600] | Four exact jq/grep_json query strings with field names + threshold + slice + idiom | **HIGH** |
| `visual-required-sally-lessons-recoverable-exceptions` | [590, 625] | Exact CamelCase identifier `RecoverableExceptions` | **MEDIUM** |

## Span notes

- Spans are chapter-relative (talk-relative [0, 978]); source-video offsets add 516.
- Spans are tight on the slide-display window: each leaves ~30-60s for retrieval to land while excluding adjacent transcript-saturated slides.
- All three sit inside the existing dev_gold span windows already validated by Codex's pass — no new transcript or talk metadata changes required.

## What I deliberately did NOT propose

- "Why does this happen?" / "Solution: planning" slides at [188, 250]: the attention-problem framing and "planning is the solution" are read aloud almost verbatim. Codex was right to reject this span; I see no visual-required content there.
- "The planning tools & states" slide: the three tools and four states are read aloud one-by-one. Not visual-required.
- "LargeJson" slide and "Compress value, not structure" slide: SallyAnn essentially reads each title aloud. The on-slide bash screenshot under LargeJson is too small/dark to safely cite a specific identifier without higher-resolution inspection.
- "The finish gate" slide: the error string "You must complete or mark as blocked all todos before finishing" is paraphrased closely enough in the transcript ("you need to go back and finish all of your to-do items"). Borderline; I left it for the existing `visual-sally-planning-tools-states-finish-gate` row.
