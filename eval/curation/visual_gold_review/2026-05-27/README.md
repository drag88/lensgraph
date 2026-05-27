# Visual Gold Curation Notes - 2026-05-27

Purpose: evidence for the first verified `visual_gold.jsonl` rows.

Method:
- Audited all 10 `dev_gold.jsonl` single-clip examples across the three dev talks.
- Used subagents by talk so each video had an independent transcript/frame sync pass.
- Accepted a row only when the transcript window and the visible frame content refer to the same answer-bearing moment.
- Saved representative contact sheets, frame grabs, agent notes, and proposed JSONL in this repo so the curation can be inspected without relying on chat history.
- Raw `.mp4` review clips are local/generated artifacts and are ignored by git; the committed review evidence is the frame/contact-sheet layer.

Summary:
- Accepted visual rows: 8.
- Rejected transcript-only rows: 2.
- Rejected ids: `tengyu-two-reasons-to-chunk`, `sally-staying-on-task-attention`.

## `W_CYk2ogcDI`

Audit notes:
- `W_CYk2ogcDI/agent_a_notes.md`
- `W_CYk2ogcDI/agent_a_proposed_visual_gold.jsonl`

Accepted rows:
- `visual-tengyu-library-analogy-slide`
  - Gold span: `168-286`.
  - Evidence: `W_CYk2ogcDI/agent_a_frame_185s_library_analogy.jpg`, `W_CYk2ogcDI/agent_a_contact_160_295_every10s.jpg`.
  - Transcript/frame sync: the spoken comparison of long context, fine-tuning, and RAG lines up with the slide titled `Long-context vs fine-tuning vs RAG`.

- `visual-tengyu-retrieval-accuracy-plot`
  - Gold span: `305-410`.
  - Evidence: `W_CYk2ogcDI/frame_310s_accuracy_slide.jpg`, `W_CYk2ogcDI/agent_a_contact_300_610_every10s.jpg`.
  - Transcript/frame sync: Tengyu explicitly says "this plot" while the retrieval-accuracy plot is visible, then states the 80% average and headroom.

- `visual-tengyu-matryoshka-quantization-slide`
  - Gold span: `405-510`.
  - Evidence: `W_CYk2ogcDI/frame_405s_matryoshka_slide.jpg`, `W_CYk2ogcDI/agent_a_frame_420s_matryoshka_slide.jpg`.
  - Transcript/frame sync: the Matryoshka/quantization cost-reduction explanation is spoken while the matching slide is visible.

- `visual-tengyu-query-document-enhancement-slide`
  - Gold span: `503-605`.
  - Evidence: `W_CYk2ogcDI/agent_a_frame_510s_enhancing_queries_documents.jpg`, `W_CYk2ogcDI/agent_a_frame_552s_document_enrichment.jpg`.
  - Transcript/frame sync: the original dev row covered hybrid search plus query/document enrichment. The visual row is narrowed to the answer-bearing `Enhancing Queries and Documents` slide inside the span.

Rejected:
- `tengyu-two-reasons-to-chunk`
  - Reason: the visible auto-chunking slide does not show the 32k context-window limit, 100k-token split, per-query LLM cost, or lost-in-the-middle claims. Those claims are transcript-only.

## `aie_sg_2026_d2_arize_alyx`

Audit notes:
- `aie_sg_2026_d2_arize_alyx/agent_b_notes.md`
- `aie_sg_2026_d2_arize_alyx/agent_b_proposed_visual_gold.jsonl`

Accepted rows:
- `visual-sally-planning-tools-states-finish-gate`
  - Gold span: `250-380`.
  - Evidence: `aie_sg_2026_d2_arize_alyx/agent_b_span2_source768_planning_tools_states.jpg`, `aie_sg_2026_d2_arize_alyx/agent_b_span2_source830_plan_outside_history.jpg`, `aie_sg_2026_d2_arize_alyx/agent_b_span2_source870_finish_gate.jpg`.
  - Transcript/frame sync: the spoken description of planning tools, states, current-plan injection, and finish gate lines up with visible slides for each element.

- `visual-sally-large-json-composable-tools`
  - Gold span: `460-615`.
  - Evidence: `aie_sg_2026_d2_arize_alyx/agent_b_span3_source1002_largejson.jpg`, `aie_sg_2026_d2_arize_alyx/agent_b_span3_source1034_compress_value.jpg`, `aie_sg_2026_d2_arize_alyx/agent_b_span3_source1066_composable_tools.jpg`, `aie_sg_2026_d2_arize_alyx/agent_b_span3_source1112_lessons_context_management.jpg`.
  - Transcript/frame sync: the LargeJson, compression, jq/grep_json, and tool-output budget claims are spoken while the corresponding slides are visible.

Rejected:
- `sally-staying-on-task-attention`
  - Reason: visible slides show the lesson heading and planning solution, but the core attention-problem explanation is transcript-only.

## `nXafozNIk3c`

Audit notes:
- `nXafozNIk3c/agent_c_notes.md`
- `nXafozNIk3c/agent_c_proposed_visual_gold.jsonl`

Accepted rows:
- `visual-google-agent-cli-getting-started-screen`
  - Gold span: `498-590`.
  - Evidence: `nXafozNIk3c/frame_520s_getting_started_docs.jpg`, `nXafozNIk3c/frame_560s_install_command_docs.jpg`, `nXafozNIk3c/frame_590s_getting_started_docs.jpg`.
  - Transcript/frame sync: Shubam explains Agent CLI and setup while the `agents-cli` Getting Started docs are visible.

- `visual-google-adk2-resume-stopped-agents-screen`
  - Gold span: `1939-2030`.
  - Evidence: `nXafozNIk3c/agent_c_frame_2004s_resume_stopped_agents_top.jpg`, `nXafozNIk3c/agent_c_frame_2019s_resumable_config.jpg`, `nXafozNIk3c/agent_c_frame_2029s_resumable_code.jpg`.
  - Transcript/frame sync: Shubam explains resume behavior and activation while ADK Agent Runtime docs show `Resume stopped agents`, `Add resumable configuration`, and `is_resumable=True`.

## Follow-Up

These rows validate the gold side only. The visual eval still cannot produce real scores until the corresponding MP4 files are staged under `videos/ai_engineering_v0/` and the frame + ColQwen patch ingest populates `frames.pooled_embedding` and `frame_patches` for these video IDs.
