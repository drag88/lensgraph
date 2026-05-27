# nXafozNIk3c visual_gold audit - agent_c

Inputs read:
- `eval/corpora/ai_engineering_v0/talks.yaml`
- `eval/corpora/ai_engineering_v0/dev_gold.jsonl`
- `transcripts/ai_engineering_v0/nXafozNIk3c.en.vtt`
- `eval/corpora/ai_engineering_v0/visual_gold.jsonl`

Existing visual_gold status:
- `visual-google-agent-cli-getting-started-screen` already exists in `visual_gold.jsonl` for the `498-590s` Agent CLI span.
- No existing visual_gold row found for the `1939-2030s` ADK resume agents span.

## Decisions

### ACCEPT: `google-cli-what-it-is-and-install`

- Span: `498-590s` (`00:08:18-00:09:50`)
- Transcript evidence:
  - `00:08:21.840-00:08:30.800`: Agent CLI works with coding agents including Gemini CLI and Codex.
  - `00:08:38.000-00:08:53.519`: coding agents can hallucinate or create buggy ADK code; Agent CLI is a CLI plus skills package for building, evaluating, and deploying agents on Google Cloud's agent platform.
  - `00:08:55.760-00:09:18.959`: setup is one install command, then it is globally picked up by coding agents and gives access to tools, codebase, and knowledge.
- Frame evidence:
  - `eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_contact_sheet_490_605_every10s.jpg`
  - `eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_frame_520s_getting_started_docs.jpg`
  - `eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_frame_560s_install_command_docs.jpg`
  - `eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_frame_590s_getting_started_docs.jpg`
- Visual assessment: The shared screen shows the `agents-cli` Getting Started documentation in the same span, including the coding-agent setup flow, install/start-building section, and skill-area table. The hallucination/buggy-code motivation is transcript-only, but the screen contains enough answer-bearing setup and Agent CLI documentation to qualify.
- Uncertainty: Low.
- Duplicate note: Proposed row intentionally duplicates an existing visual_gold concept.

### ACCEPT: `google-adk2-resume-agents`

- Span: `1939-2030s` (`00:32:19-00:33:50`)
- Transcript evidence:
  - `00:32:30.559-00:33:20.880`: production agents face long runs, dropped connections, down services, 2 a.m. events, and workflows spanning hours or days.
  - `00:33:23.519-00:33:46.799`: resume picks up where the agent left off, is enabled by a flag on app config, tracks tools that already ran, skips them, and continues from the break.
  - `00:33:36.000-00:33:42.159`: Shubam says to flip a flag on top of app config, and the screen shows the matching resumable configuration within the span.
- Frame evidence:
  - `eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_contact_sheet_1939_2030_every10s.jpg`
  - `eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_frame_2004s_resume_stopped_agents_top.jpg`
  - `eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_frame_2019s_resumable_config.jpg`
  - `eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_frame_2029s_resumable_code.jpg`
- Visual assessment: The shared screen moves from the ADK Agent Runtime docs to the `Resume stopped agents` page inside the same span. The visible docs say interrupted workflows can be resumed after an unexpected interruption, show `Add resumable configuration`, and show code configuring resumability with `is_resumable=True`. The exact examples of 2 a.m. events and hours/days workflows are transcript-supported rather than visually listed, but the screen is answer-bearing for the resume behavior and activation.
- Uncertainty: Medium-low because the available YouTube format was 360p, but the heading and key configuration text are readable enough in extracted frames.

## Rejected rows

None. Both `nXafozNIk3c` dev_gold rows have answer-bearing visual content in the same timestamp span as the transcript answer.

## Evidence commands

- Copied existing `498-590s` frame/contact-sheet evidence into `agent_c_` filenames.
- Downloaded only the ADK resume candidate window:
  - `yt-dlp -f "bv*[height<=720][ext=mp4]/bv*[height<=720]/best[height<=720]/best" --download-sections "*00:32:19-00:34:10" --force-keyframes-at-cuts --no-playlist -o "eval/curation/visual_gold_review/2026-05-27/nXafozNIk3c/agent_c_clip_1939_2030.%(ext)s" "https://www.youtube.com/watch?v=nXafozNIk3c"`
- Checked available YouTube formats:
  - `yt-dlp -F "https://www.youtube.com/watch?v=nXafozNIk3c"`
- Extracted ADK resume evidence with `ffmpeg` into:
  - `agent_c_contact_sheet_1939_2030_every10s.jpg`
  - `agent_c_frame_1989s_resume_stopped_agents_top.jpg`
  - `agent_c_frame_1999s_resume_stopped_agents.jpg`
  - `agent_c_frame_2004s_resume_stopped_agents_top.jpg`
  - `agent_c_frame_2009s_resume_stopped_agents_top.jpg`
  - `agent_c_frame_2019s_resumable_config.jpg`
  - `agent_c_frame_2029s_resumable_code.jpg`
