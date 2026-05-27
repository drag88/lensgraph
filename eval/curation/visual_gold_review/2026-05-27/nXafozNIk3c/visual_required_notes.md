# nXafozNIk3c visual-REQUIRED proposal — claude-opus-4-7

Goal: break out of the text-saturated regime in `visual_gold.jsonl`. Existing 8 rows
all get 8/8 from the 4-channel text RRF because the transcript fully describes the
on-screen content. The three rows below are constructed so the screen carries the
answer and the transcript does not.

## Inputs inspected

- `transcripts/ai_engineering_v0/nXafozNIk3c.en.vtt` (segments 8:09-9:55 and 32:00-34:00).
- `eval/corpora/ai_engineering_v0/dev_gold.jsonl` (rows `google-cli-what-it-is-and-install`, `google-adk2-resume-agents`).
- `eval/corpora/ai_engineering_v0/visual_gold.jsonl` (rows `visual-google-agent-cli-getting-started-screen`, `visual-google-adk2-resume-stopped-agents-screen`).
- `eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_notes.md` and every `agent_c_frame_*.jpg` in that directory.

## Per-row rationale

### `visual-required-google-agents-cli-seven-skills-list` — HIGH

- Frame evidence: `agent_c_frame_520s_getting_started_docs.jpg`, `agent_c_frame_590s_getting_started_docs.jpg`, `agent_c_contact_sheet_490_605_every10s.jpg`. The skills table is visible in all of them and the seven IDs are readable: `google-agents-cli-workflow`, `google-agents-cli-adk-code`, `google-agents-cli-scaffold`, `google-agents-cli-eval`, `google-agents-cli-deploy`, `google-agents-cli-publish`, `google-agents-cli-observability`, each with a one-line description in the right column.
- Transcript at 9:46-9:55: "packages all these skills that you see here. It's like seven skills package which pretty much covers the entire agent development life cycle." That's the most specific the speaker ever gets. He never names a single skill ID. He never says "scaffold," "eval," "publish," "observability," "Cloud Run," "GKE," "CI/CD," "LLM-as-judge," or "Gemini Enterprise."
- Why visual-required: the eight expected claims are exact bigrams from the on-screen table and cannot be reconstructed from anything in the transcript. A text-only retriever cannot answer.

### `visual-required-google-adk-resumability-config-code` — HIGH

- Frame evidence: `agent_c_frame_2019s_resumable_config.jpg`, `agent_c_frame_2029s_resumable_code.jpg`. Both show the same dark-themed code block: `app = App(name='my_resumable_agent', root_agent=root_agent, # Set the resumability config to enable resumability. resumability_config=ResumabilityConfig(is_resumable=True),)`.
- Transcript at 33:23-33:46: speaker says you "flip a flag on top of app config" and the agent "picks up where it left off." He never says `App`, `ResumabilityConfig`, `is_resumable=True`, `my_resumable_agent`, or `root_agent`. He paraphrases the mechanism; the literal API surface is screen-only.
- Why visual-required: the expected claims are exact Python identifiers. A reranker scoring against the transcript will not find `ResumabilityConfig` anywhere.

### `visual-required-google-adk-resume-version-and-caveat` — HIGH

- Frame evidence: `agent_c_frame_2004s_resume_stopped_agents_top.jpg` (clearly shows the breadcrumb `Home > Run Agents > Agent Runtime > Resume stopped agents`, the "Supported in ADK: Python v1.16" badge, and the body text "In ADK Python 1.16 and higher, you can configure an ADK workflow to be resumable"); `agent_c_frame_2019s_resumable_config.jpg` (shows the yellow "Caution: Long Running Functions, Confirmations, Authentication" callout in full).
- Transcript across 32:19-33:50: speaker never says "1.16," never says "Caution," never mentions Long Running Functions / Confirmations / Authentication as special cases, never reads the breadcrumb. These are pure-screen artifacts.
- Why visual-required: version constraints and caution callouts are exactly the kind of documentation metadata speakers skip. The three expected claims cannot be answered from transcript.

## What I rejected and why

- **Literal install command (`pip install ...` or similar).** The frame I have for ~560s shows the skills table, not an install snippet. Without confirmed visual evidence of the exact command string I cannot mark `verified: true`. If a future curator pulls a frame in the 9:00-9:10 range showing the actual install command, that would be a fourth strong candidate.
- **Left-nav structure / "agents-cli" repo URL.** Available in frames but already adjacent to the existing `visual-google-agent-cli-getting-started-screen` row's claim space. Adding a fourth row keyed on nav structure would dilute rather than diversify.

## Output discipline

- All three rows carry `tags: ["visual-required", ...]`.
- All three carry `modality: ["screen_code"]` only — no `transcript` — to make the lift signal sharp during the 5-channel bakeoff.
- Spans are tight (15-75s) and stay inside the existing dev_gold windows.
- `verified: true` is backed by the frames named above. I have visually inspected each frame for the strings in `expected_claims`.

## Suggested next action

Promote these three rows into `eval/corpora/ai_engineering_v0/visual_gold.jsonl` only after a second pass confirms the text retriever truly misses them on `dev_gold`. If `ClaimsSupported` from the 4-channel text RRF stays under ~0.4 on these three, they have done their job: visual lift now has room to move.
