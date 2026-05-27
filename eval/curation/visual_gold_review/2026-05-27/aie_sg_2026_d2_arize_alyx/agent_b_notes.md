# Agent B Visual Gold Audit: aie_sg_2026_d2_arize_alyx

Source talk is a chapter slice of YouTube `m12vGjfbNlo`, with `source_start_sec=516`. All source-video inspection windows below add 516 seconds to the chapter-relative `dev_gold` spans.

## Decisions

| dev_gold id | Chapter span | Source span | Decision | Uncertainty |
| --- | ---: | ---: | --- | --- |
| `sally-staying-on-task-attention` | 188-250 | 704-766 | Reject | Low |
| `sally-planning-tools-states-finish-gate` | 250-380 | 766-896 | Accept | Low |
| `sally-large-json-and-composable-tools` | 460-615 | 976-1131 | Accept | Low |

## `sally-staying-on-task-attention` - Reject

Transcript evidence: SallyAnn says agents often handle the first requested item but forget later items, says this is not hallucination or capability but an attention problem, then says the solution is planning and that Alyx creates an explicit to-do before acting.

Frame evidence:
- `agent_b_span1_contact_source_700_770.jpg`
- `agent_b_span1_source718_why_happen.jpg`
- `agent_b_span1_source748_solution_planning.jpg`

Visual finding: the span shows slides titled `LESSON #1 Staying on task`, `Why does this happen?`, and `The solution: planning`. The planning solution is visible, but the answer's core explanation of the attention failure is delivered in transcript rather than shown on-screen. I do not think this row is sufficiently answer-bearing as a visual candidate.

## `sally-planning-tools-states-finish-gate` - Accept

Transcript evidence: SallyAnn names the three tools `todo_write`, `todo_update`, and `todo_read`; names the four states `pending`, `completed`, `blocked`, and `in progress`; explains `in_progress` was added after an initial pending/completed setup; says planning lives outside conversation history and is injected after system instructions on each LLM call; and describes the finish gate as an explicit structured error that prevents finishing until todos are complete.

Frame evidence:
- `agent_b_span2_contact_source_762_900.jpg`
- `agent_b_span2_source768_planning_tools_states.jpg`
- `agent_b_span2_source830_plan_outside_history.jpg`
- `agent_b_span2_source870_finish_gate.jpg`

Visual finding: accepted. The visible slides directly show the planning tools and states, a current-plan block under `Plans Lives Outside Conversation History`, and the finish-gate error: `You must complete or mark as blocked all todos before finishing`.

## `sally-large-json-and-composable-tools` - Accept

Transcript evidence: SallyAnn says one Arize experiment can be hundreds of rows or about 100,000 tokens, describes LargeJson as storing most tool data in serialized memory and giving the agent an ID for later context, says the compression rule is values not structure, names jq and GP-JSON/grep-json for regex search over serialized data, and says tool outputs have a roughly 10,000-token budget.

Frame evidence:
- `agent_b_span3_contact_source_972_1135.jpg`
- `agent_b_span3_source1002_largejson.jpg`
- `agent_b_span3_source1034_compress_value.jpg`
- `agent_b_span3_source1066_composable_tools.jpg`
- `agent_b_span3_source1112_lessons_context_management.jpg`

Visual finding: accepted. The visible slides show `LargeJson`, `Compress value, not structure`, jq/grep_json examples under `Small, composable tools == Infinite Context`, and `Hard token budgets on every tool output`.

## Network and Video Commands Used

- `yt-dlp --no-playlist --download-sections '*00:11:40-00:12:50' --force-keyframes-at-cuts -f 'bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/best[height<=720]' --merge-output-format mp4 -o '.../agent_b_span1_source_700_770.%(ext)s' 'https://www.youtube.com/watch?v=m12vGjfbNlo'`
- `yt-dlp --no-playlist --download-sections '*00:12:42-00:15:00' --force-keyframes-at-cuts -f 'bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/best[height<=720]' --merge-output-format mp4 -o '.../agent_b_span2_source_762_900.%(ext)s' 'https://www.youtube.com/watch?v=m12vGjfbNlo'`
- `yt-dlp --no-playlist --download-sections '*00:16:12-00:18:55' --force-keyframes-at-cuts -f 'bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/best[height<=720]' --merge-output-format mp4 -o '.../agent_b_span3_source_972_1135.%(ext)s' 'https://www.youtube.com/watch?v=m12vGjfbNlo'`
- `ffmpeg` contact sheets for each downloaded window with `fps=1/6`, `fps=1/12`, and `fps=1/14`.
- `ffmpeg` single-frame extracts for the evidence files listed above.
