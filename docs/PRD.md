# PRD — LensGraph

## Problem

Engineers and researchers consume hours of recorded talks (NeurIPS, AI Engineer Summit, conferences, lectures, podcasts) but cannot search inside them with the precision they expect from text. Existing tools fall into two camps:

1. **Transcript search** (YouTube captions, Otter, Fireflies) — keyword-only, no semantic understanding, no slide content, citations are timestamps without context.
2. **General-purpose video assistants** (Gemini long-video, NotebookLM video) — semantic but no concept of "the exact clip that answers this question," and weak on visual modalities like slides and screen-shared code.

The specific failure mode this project targets: *"I remember someone explained this clearly in a talk last month. Find me that 30-second clip."*

## Who this is for

- **Primary persona:** AI engineers and ML practitioners who consume conference talks weekly.
- **Secondary persona:** Recruiters and hiring managers evaluating this project. They are explicit stakeholders; the project's portfolio function is part of its product definition.

## Job to be done

Given a natural-language question and a corpus of indexed talks, return:

1. The most relevant clip(s) — start and end timestamps that fully contain the answer with comprehensible lead-in.
2. A grounded answer composed only from retrieved content, with claim-level citations to specific timestamps.
3. Visible reasoning trace — every retrieval, rerank, verify, and generate step is inspectable.
4. Abstention when the corpus does not contain the answer.

## Non-goals (v0–v1)

- Real-time ingestion of live streams.
- Multi-tenant or auth (this is a single-user research tool).
- Mobile UI.
- Audio chat ("Echo" idea was a separate project; not in scope).
- Cross-language retrieval; English-only at launch.
- Diff mode across versioned talks (v2 candidate).

## Success criteria

The project is successful when all four hold simultaneously:

1. **Eval green:** `make validate-evals` passes; held-out test set scores meet targets defined in `docs/eval-methodology.md`.
2. **Public methodology:** the chunking ablation report is published as an MDX page on the deploy with reproducible commands.
3. **Demo is legible:** a recruiter watching for 60 seconds understands what was built and why it is hard.
4. **Two interview wins:** the project leads to two on-site interviews where it is referenced positively by the interviewer.

The last criterion is the actual product success metric. The technical metrics are inputs to it.

## Constraints

- **Solo build, 12 weeks part-time.** Roughly 6–10 focused hours per week.
- **Single-machine inference budget.** No multi-GPU training.
- **One datastore.** Postgres for relational + pgvector + queue (PGMQ). No Redis or separate queue.
- **One agent framework.** LangGraph. Custom ingestion. No LangChain core, no LlamaIndex.
- **Cross-model eval discipline.** Curate with one LLM family, judge with another, to limit same-model bias.

## Out of scope risks acknowledged

- Public corpus may contain copyrighted material. Transcripts are stored locally, never redistributed.
- LLM judge is imperfect; mitigated by Cohen's kappa against a manual audit, not by replacing it.
