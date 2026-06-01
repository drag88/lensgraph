<role>
You are generating a single self-contained HTML file — `docs/codebase-explorer.html` — that the project's solo author (Aswin) will open locally to (a) refresh his mental model of the LensGraph codebase and (b) actually learn the multimodal-RAG / retrieval / embeddings domain he is building in. This is a personal onboarding *and* teaching tool, not a recruiter pitch. Tone: opinionated, terse, no marketing fluff — but every piece of jargon earns a plain-language gloss the first time it appears.

Audience profile: senior data leader (Head of Data, consumer business at a telco), deep Python / SQL / Snowflake / dbt background, comfortable with statistics and pipelines. New to: vector search, embeddings, multimodal retrieval, agent frameworks, JSON Schema conditionals, pgvector internals, late-interaction models, RRF, MaxSim, ColPali / ColQwen, BGE-M3, judge contamination. Bridge from what he knows (warehouses, dbt models, SQL execution plans, analytics) toward what he is learning.
</role>

<memory_limits>
This prompt must be safe to run inside Cursor.

- Do not use `@Codebase`, whole-repo context, or broad recursive file reads.
- Do not read `.venv/`, `videos/`, `frames/`, `transcripts/`, `eval/curation/visual_gold_review/`, or the existing `docs/codebase-explorer.html` as source context.
- Keep compact notes after each source read. Do not keep full file bodies in context except for the specific code snippets that will render in the HTML.
- Run teaching/walkthrough drafting serially or with one bounded helper at a time. Do not spawn parallel subagents from Cursor.
- Replace `docs/codebase-explorer.html` once at the end after assembling the sections. Do not open or re-read the generated HTML after writing it.
</memory_limits>

<learning_mode>
Every technical term gets a one-line plain-English gloss the *first* time it appears in any section. Where a clean analogy exists, use it — prefer bridges from the data-warehouse / SQL / dbt world the reader already inhabits. A few worked examples of the bridge style:

- "JSON Schema conditional" → "the JSON equivalent of a `CHECK` constraint that fires only when another column has a specific value (e.g. `question_type = 'synthesis'` requires `gold_spans >= 2`)."
- "Embedding" → "a fixed-length vector that captures meaning; think of it as a hashed feature representation you can compute `cosine_similarity` over instead of `LIKE '%term%'`."
- "pgvector HNSW index" → "an ANN index for vector columns; the vector-search equivalent of a B-tree — fast nearest-neighbour lookup at the cost of approximate results."
- "RRF (Reciprocal Rank Fusion)" → "stitching results from N rankers by summing `1 / (k + rank)` per item; the retrieval-world version of a weighted UNION ALL with a positional discount."
- "Late interaction / MaxSim (ColPali, ColQwen)" → "instead of one vector per document, keep one vector per token/patch and score by `MAX(cosine)` per query token; like joining at the row level instead of pre-aggregating."
- "Cross-family judge" → "score model A's outputs with model B from a different lineage; same-family judging is like letting the dbt developer also grade their own data-quality tests."
- "Phase-0 gate" → "a pre-commit assertion that the eval corpus is real before any retrieval code can land; the methodological version of `NOT NULL` on the evaluation set."

Rules:
- First-mention gloss is mandatory; later mentions in the same section can be bare.
- Glosses are one sentence, ≤ 30 words. Longer explanations belong in the Concepts section (see required section 3) or in section panels.
- Analogies must be accurate, not cute. If no honest analogy exists, give the plain definition without one.
- Every jargon term in the rendered HTML becomes a `<span class="term" data-term="...">` whose tooltip is reachable by **click/focus AND hover — never hover-only** (hover-only fails touch and keyboard; NN/g). The term links to its full entry. Build a small JS registry — define each term once, render the gloss everywhere it appears.
- Do not dumb down. The reader is technical. The job is to remove unnecessary translation friction, not to remove rigor.

Teaching discipline (from learning-science research; applies to Foundations lessons AND every Walkthrough `concept` block):

- **Tag each concept adjacent vs genuinely new, and match depth to the tag.** Adjacent = a re-skin of what he already knows (HNSW ≈ an index, top-k ≈ `LIMIT`, reranker ≈ a better `ORDER BY`): state the one-clause mapping and move on. Genuinely new (late interaction, MaxSim, patch embeddings, judge contamination): full worked example. Over-explaining the familiar actively costs an expert reader learning capacity (expertise-reversal + redundancy effects). Do not define an index, a join, idempotency, or a primary key.
- **Analogies bridge on structure, not surface, and every analogy carries a "where this breaks" clause.** "An embedding is just a row of floats" is surface-true but structurally misleading — if used, immediately add the disanalogy (the dimensions are not interpretable; only relative position carries meaning).
- **Climb to the abstraction; do not stop at the analogy.** Ground concrete first, then state the domain-general idea stripped of the SQL skin. Staying concrete kills transfer.
- **Answer-first.** Lead each explanation with the takeaway, then the mechanism (Minto). Do not bury the point at the end of a walked example.
</learning_mode>

<references>
Approved "learn more" citations. These URLs were verified against live sources on 2026-05-30. Use ONLY these for external links; do not invent citations or link to SEO blogspam. Each Foundations lesson and each Walkthrough concept whose topic appears here must carry the canonical link. Render as a small "Learn more ↗" line under the relevant beat/concept.

| Topic | Canonical source | URL |
|---|---|---|
| RAG (the original idea) | Lewis et al. 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" | https://arxiv.org/abs/2005.11401 |
| Embeddings / BGE-M3 (dense+sparse+multi-vector in one model) | Chen et al. 2024, "M3-Embedding" | https://arxiv.org/abs/2402.03216 |
| BGE-M3 model card | BAAI/bge-m3 | https://huggingface.co/BAAI/bge-m3 |
| HNSW (the ANN index behind pgvector) | Malkov & Yashunin 2016 | https://arxiv.org/abs/1603.09320 |
| pgvector (Postgres vector search; halfvec + sparsevec since 0.7) | pgvector repo | https://github.com/pgvector/pgvector |
| RRF (rank fusion) | Cormack, Clarke & Buettcher, SIGIR 2009 | https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf |
| Late interaction / MaxSim (origin) | Khattab & Zaharia 2020, "ColBERT" | https://arxiv.org/abs/2004.12832 |
| ColBERTv2 (practical late interaction) | Santhanam et al. 2021 | https://arxiv.org/abs/2112.01488 |
| ColPali (visual retrieval over page images) | Faysse et al. 2024 | https://arxiv.org/abs/2407.01449 |
| colpali-engine (the library) | illuin-tech/colpali | https://github.com/illuin-tech/colpali |
| ColQwen2.5 model card | vidore/colqwen2.5-v0.2 | https://huggingface.co/vidore/colqwen2.5-v0.2 |
| ViDoRe (visual-doc retrieval benchmark) | illuin-tech/vidore-benchmark | https://github.com/illuin-tech/vidore-benchmark |
| bge-reranker-v2-m3 (cross-encoder rerank) | BAAI/bge-reranker-v2-m3 | https://huggingface.co/BAAI/bge-reranker-v2-m3 |
| PGMQ (queue on Postgres) | pgmq/pgmq | https://github.com/pgmq/pgmq |
| LangGraph (state-graph agent loop) | LangGraph Graph API docs | https://docs.langchain.com/oss/python/langgraph/graph-api |
| WhisperX (word-level timestamps) | Bain et al. 2023 | https://arxiv.org/abs/2303.00747 |
| LLM-as-judge biases | Zheng et al. 2023, "MT-Bench" | https://arxiv.org/abs/2306.05685 |
| Self-preference bias (evidence behind cross-family judge / ADR 004) | Panickssery et al. 2024, "LLM Evaluators Recognize and Favor Their Own Generations" | https://arxiv.org/abs/2404.13076 |

Two factual gotchas to surface as short inline "gotcha" callouts where the topic is taught:
- **PGMQ moved orgs.** The project is now `pgmq/pgmq`, not `tembo-io/pgmq` (the old URL only works via redirect). Link `pgmq/pgmq` directly.
- **ColQwen2.5 has a split license.** The Qwen2.5-VL backbone is under the non-commercial Qwen Research license; only the ColPali adapters are MIT. "Apache-2.0" inferred from colpali-engine is misleading for the full stack.
</references>

<scope>
Document only what exists in the repo right now. The implementation directories (`ingest/`, `chunking/`, `retrieve/`, `generate/`, `api/`, `web/`) are empty by design — note that fact once in the at-a-glance section and move on. The real surface area today is:

- `eval/schemas/` — JSON Schemas (Draft 2020-12)
- `eval/corpora/` — `talks.yaml` + JSONL files per corpus
- `eval/validate.py` — single-file validator with multiple modes
- `eval/tests/fixtures/` — `valid_*.json` and `invalid_*.json` for the self-test
- `eval/curation/` — playbook + helper scripts
- `eval/config/model_candidates.yaml` — bakeoff candidate config
- `docs/decisions/*.md` — ADRs
- `Makefile`, `pyproject.toml`
- `.claude/CLAUDE.md` and `.claude/rules/*.md` — project hard rules

Do not document planned phases or roadmap items as if they exist.
</scope>

<reading_order>
Before writing anything, read in this order:

1. `README.md`
2. `.claude/CLAUDE.md`
3. `.claude/rules/architecture.md`
4. `.claude/rules/patterns.md`
5. `.claude/rules/lensgraph-core-principles.md`
6. `eval/validate.py` (entire file)
7. Every schema in `eval/schemas/`
8. Every ADR in `docs/decisions/`
9. `Makefile`
10. A representative sample of `eval/corpora/*/talks.yaml` and `eval/tests/fixtures/`

Code snippets in the HTML must be copied verbatim from the source files at generation time, not paraphrased.
</reading_order>

<required_sections>
Nine sections in this exact order. Concepts and code explanations are NOT separated out — they live inline inside the Walkthrough stops (section 4) and the User query journey (section 5). This is deliberate; earlier versions had a Concepts dumping ground and a separate Key Functions section, which forced readers to context-switch between code and theory. Do not bring those sections back.

1. **At-a-glance** — hero treatment. One italic-serif headline, a deck paragraph that names the project + what is built today + what is empty by design (ingest/chunking/retrieve/generate/api/web), four KPI cards (Phase-0 gate status, schema count, ADR count, validator LOC), a "Gate criteria" callout with the exact `make phase0-gate` command, and the eight CLAUDE.md hard rules in a 2-column numbered grid.

2. **Foundations of multimodal RAG** — an 8-lesson learning ladder that teaches the domain from first principles. Use the `foundations-teacher` method (`.claude/skills/foundations-teacher/SKILL.md`). Produce the lessons in two small serial batches, each following the 7-beat structure (question / anchor / insight / walked_example / tradeoffs / heuristic / next). The 8-lesson spine, in order — note that visual retrieval gets its own dedicated lesson, NOT a paragraph tacked onto MaxSim:
   1. **What "search" actually means** — lexical vs semantic; where `ILIKE` / BM25 break.
   2. **Turning meaning into geometry** — embeddings, cosine similarity, BGE-M3 1024 dims, worked cosine on paper.
   3. **Finding neighbours fast** — brute force vs HNSW, the speed/recall trade.
   4. **When one channel isn't enough** — dense + sparse + multi-vector, RRF fusion with a chunk-by-chunk worked example (text only).
   5. **Late interaction and MaxSim** — the scoring mechanism, token-level, text-only. Mention generalisation to image patches only as a teaser pointing at lesson 6.
   6. **Seeing the slide itself: ColPali and ColQwen** — dedicated lesson on visual retrieval. Cover the OCR-then-embed failure mode, vision-language patch grids (~14×14), `vidore/colqwen2.5-v0.2` via `colpali-engine`, the ingestion-side trade-offs (frame sampling rate, MPS vs Modal bursts, ~196 patch vectors per frame), and a worked example where the slide gets retrieved on visual signal alone when both text channels miss.
   7. **Putting it together: the RAG pattern** — retrieve → rerank → generate under citation, end-to-end.
   8. **Knowing whether your system works** — gold examples, recall@k, faithfulness, contamination traps, the phase-0 gate.

   Each lesson renders as `<details class="lesson">` with a number badge + serif title + italic question in the summary; body shows the six labeled beats. Lesson 1 is `open` by default; 2–8 collapsed. Worked examples must use real numbers and LensGraph-flavoured queries.

3. **Architecture** — two Mermaid diagrams, both inside the full `.diagram-shell` zoom-pan-expand pattern (never bare `<pre class="mermaid">`). Set up a reusable `setupZoom(viewportId, canvasId, ctrlSuffix, title)` helper so both diagrams get independent zoom/pan/expand state without code duplication.
   - **(a) Today's eval harness** — clickable. Modules: `.claude/`, `docs/decisions/`, `eval/schemas/`, `eval/tests/fixtures/`, `eval/config/`, `eval/corpora/`, `eval/curation/`, `eval/validate.py`, `Makefile`. Clicking any node expands an inline panel showing purpose, key files, key functions, and the 1–2 invariants that govern that module.
   - **(b) Planned 5-channel retrieval pipeline** — non-clickable orientation diagram showing the query path that the system will run once phase-0 passes. Must visibly include: User query → 4 text channels grouped under `text retrieval — BAAI/bge-m3` (BM25, BGE-M3 dense, BGE-M3 sparse, BGE-M3 multi-vector) PLUS 1 visual channel grouped under `visual retrieval — vidore/colqwen2.5-v0.2` (ColQwen2.5 patch-level MaxSim) → RRF fusion (top 30) → bge-reranker-v2-m3 cross-encoder (top 8) → confidence check with retry loop → Generator (bakeoff winner via ADR 004) → cited answer. Use Mermaid `subgraph` blocks to group the two retrieval families so the visual channel is unmistakably first-class.

   The two-diagram structure exists because the eval harness diagram alone makes it look as if retrieval ends at the validator. ColPali / ColQwen2.5 must be visible at the architecture level, not buried in Foundations lesson 5 or Walkthrough stop 20.

4. **Codebase walkthrough** — the spine of the document. 22 file-by-file tour stops that fuse code + what + why + the embedded foundational concept + connections. Produce the stops in four serial batches (1–5, 6–12, 13–17, 18–22), using the `foundations-teacher` method for the inline concept at every stop. Each stop has the fixed 7-field shape: `file_path` (with line range), `name`, `purpose` (1 line), `code` (verbatim from source, ~30 lines max), `code_lang`, `what` (mechanism, 2–4 sentences), `why` (1–2 sentences citing CLAUDE.md/ADR/patterns.md), `concept` (object with `title` + `bridge` (1–2 sentences from SQL/dbt/warehouse world) + `insight` (2–4 sentences, mechanism-first) + `takehome` (1 sentence rule-of-thumb)), `connects_to` (upstream + downstream arrays). The 22 stops cover, in order: the 12 functions of `eval/validate.py` (load_validator → main); the 4 schemas (talk, gold_example, boundary_audit, model_candidates); the fixtures directory; the corpora directory; the curation playbook + clip_vtt.py; `model_candidates.yaml`; `eval/runners/providers.py`; `eval/runners/minimal_generation.py`.

   Render each stop as a full-width `<article class="walk-stop">` with: number badge + h3 title + path + purpose in header; two-column grid (code on left, what+why on right); a highlighted `<aside class="walk-concept">` block with the embedded foundational concept (bridge / insight / take-this-home); a footer with upstream + downstream `<ul>` lists. No collapsing — the stops are meant to be scrolled through.

5. **User query journey** — the missing narrative. Walks the reader through what happens when a user types a question, from keystroke to cited answer. Frame as PLANNED behaviour (the implementation directories `api/`, `web/`, `generate/`, `retrieve/` are empty by design until phase-0 passes), but write in present tense as if it runs today. Produce this section after reading only the listed source files. Required pieces:
   - **Tagline** (one short serif headline as a pull-quote, e.g. "A question goes in, an answer comes out with the 27-second clip that proves it.")
   - **POV block** (2-column intro): an opinionated POV paragraph (not a chatbot; load-bearing UX choice is no claim without a citation that resolves to a replayable clip), and a what-the-user-actually-sees paragraph.
   - **8–10 step cards**, each with: `n`, `title`, `user_sees` (what the user experiences), `behind_scenes` (which subsystem/file/framework is responsible), `timing` (latency budget pill), `ref` (planned file path + cross-link to a foundations lesson or walkthrough stop). Steps must cover: type query → API stream open → Plan node classifies → Retrieve fires 5 channels in parallel → RRF fuses → cross-encoder rerank → Verify (with loop-back on low confidence) → Generate constrained to top 8 → Cite attaches clickable pills → Trace lands in Langfuse for replay. Total budget ~2.5–3s end-to-end.
   - **Closing POV block** — why this UX matters (every claim is timestamp-cited; generator cannot free-associate; faithfulness is `ClaimsSupported` + `CitationAccuracy` on the locked test set; refusal is first-class with `RefusalRate ≥ 0.90` on corpus-negatives).
   - **"What this is NOT" callout** — 4–5 short bullets contrasting against generic patterns being rejected (not a chatbot, not a summariser, not a black box, not a search engine, not optimised for the model but for the eval).

   Render the steps as `<article class="journey-step">` with a 2-column grid (user-sees on left, behind-scenes on right), the timing as a small monospace pill in the header, and the ref line in a bg-tinted footer. The POV close gets a gold-bordered card; the anti-pattern callout uses `×` bullets in red.

6. **Data flow animations** — two animated SVG flows:
   - (a) Gold example lifecycle: `dev_gold.draft.jsonl` → curator watches clip → flipped to `verified:true` → promoted to committed corpus → `make validate-evals-strict` → CI.
   - (b) Validator pipeline: load schemas → load corpora → schema check → semantic check → SHA drift → fixture self-test → phase-0 gate.
   - Use CSS keyframes or requestAnimationFrame. No animation libraries. Active step pulses gold. Under `@media (prefers-reduced-motion: reduce)` the flow renders as a static labeled diagram with no pulsing or auto-advance.

7. **ADR index** — collapsible list of all ADRs with Status, one-line Decision, and "Revisit when" trigger.

8. **Search & navigation** — top-bar fuzzy search over section titles, lesson titles, walkthrough stop names, journey step titles, ADR titles, file paths. Press `/` to focus. Pure client-side, no dependencies (~30-line fuzzy matcher). Indexed kinds: `section`, `lesson`, `stop`, `journey`, `adr`, `file`.

9. **Directory browser** — a tabbed view, one tab per top-level documented directory (`eval/schemas/`, `eval/corpora/`, `eval/validate.py`, `eval/tests/fixtures/`, `eval/curation/`, `eval/config/`, `docs/decisions/`, `.claude/`). Each tab shows the file tree with a one-line purpose; click-to-expand shows a content preview.
</required_sections>

<bounded_generation>
Three sections need extra teaching care because their failure mode is definitions without a point of view. In Cursor, do this as bounded serial generation, not parallel subagents.

- **Section 2 (Foundations)**: use the `foundations-teacher` method. Produce two serial batches covering lessons 1–4 and 5–8. Each follows the 7-beat structure from `.claude/skills/foundations-teacher/SKILL.md`.

- **Section 4 (Walkthrough)**: produce four serial batches: stops 1–5, 6–12, 13–17, 18–22. Each uses the `foundations-teacher` method for the inline `concept` block embedded at every stop.

- **Section 5 (User query journey)**: read `docs/architecture.md`, `docs/PRD.md` (if present), `.claude/rules/architecture.md`, and `eval/config/model_candidates.yaml`, then write the section_intro + 8–10 steps + pov_close + anti_pattern_callout per the section spec. POV must be opinionated and grounded in LensGraph's specific choices (5 channels including ColQwen visual, cross-family judge, locked test set, refusal as first-class).

For all three sections: the forbidden-phrase list applies — no "comprehensive", "robust", "powerful", "seamless", "enable", "leverage", "ensure", "allow", "delve", "navigate", "empower", "unlock"; no emoji; opinionated whiteboard tone.

Each batch returns structured JSON only. The integration step converts markdown (backticks → `<code>`, fenced blocks → `<pre><code class="language-…">`), assembles the section markup, wires anchors and the fuzzy search index. Do not generate HTML inside the teaching batches.
</bounded_generation>

<output_constraints>
- One file: `docs/codebase-explorer.html`. Inline CSS and JS. Only external deps are Mermaid + Prism via CDN.
- No build step. Opens by double-clicking.
- Dark mode by default. Light/dark toggle in the corner.
- Themable via CSS custom properties declared on `:root` — at minimum `--bg`, `--fg`, `--muted`, `--accent`, `--border`, `--surface`, `--code-bg`. Light-mode overrides live in `:root[data-theme="light"]`. Every color reference in the stylesheet reads from these variables. No hex literals scattered through component styles.
- Persistent left sidebar (sticky, ~240px, centered with the rest of the layout via max-width 1720px) with anchor links to all nine primary sections plus the directory-browser tab list. Active section highlighted on scroll via IntersectionObserver. Collapsible on narrower screens (< 1100px). Layout responds at 1400px / 1900px / 2400px breakpoints — content max-width scales with viewport so wide screens are not wasted.
- Optimized for a 27" monitor; mobile-unfriendly is fine.
- No emoji anywhere. No marketing language. No "comprehensive", "powerful", "seamless".
- Footer: `Generated YYYY-MM-DD from commit <short-sha>. Regenerate by invoking the codebase-explorer skill/rule.`
- Replace the date and commit SHA by reading the working tree at generation time. If git is unavailable, use `unknown`.

Interaction and accessibility (from artifact-design research — matklad, Nicky Case, NN/g, Diátaxis):

- **No stale source links.** Name files/functions in prose; do not hyperlink to specific paths or line numbers (they rot). Wayfinding is the in-page fuzzy search (`/` to focus), not `<a href>` to source files. External "learn more" links to papers/docs are fine and encouraged — those are stable.
- **Interactivity only where it beats text** (Nicky Case). The interaction budget is: zoom/pan architecture diagrams, the two data-flow animations, click-glossary, search, sticky scroll-spy TOC. Do not add decorative widgets.
- **No scrolljacking.** Never override native scroll. Two-column walkthrough uses a sticky code panel (pure CSS `position: sticky`), not scroll-triggered visual swapping.
- **Respect `prefers-reduced-motion`.** Both data-flow animations (section 6) and any pulsing/transition must collapse to a static state under `@media (prefers-reduced-motion: reduce)`.
- **Label each section by Diátaxis mode** with a small tag in its header: Foundations = explanation; Walkthrough = tutorial; Directory + ADR index = reference; the "run the answer loop" parts of the User query journey = how-to. Keep each section in its mode; do not blend a reference dump into an explanation.
- **Cited "learn more" links are required, not optional.** Every Foundations lesson and every Walkthrough `concept` whose topic appears in the `<references>` table below carries a link to the canonical source from that table. Use only URLs from the table — do not invent citations.
</output_constraints>

<forbidden>
- Documenting planned phases as if they exist
- Adding "Getting Started" or "Contributing" sections (this file is for the author, not strangers)
- Creating any files besides `docs/codebase-explorer.html`
- Modifying any source files
- Running `make`, the validator, or any test command — read only
- Inventing function names, file paths, or schema fields not present in the actual code
- Using AI fluff: "comprehensive overview", "robust system", "powerful tool", "seamless experience"
- Using `@Codebase` or broad whole-repo context
- Reading the existing `docs/codebase-explorer.html` as source context
- Spawning parallel subagents from Cursor
</forbidden>

<completion>
When done, print exactly three lines:

1. Path to the generated HTML
2. Section count and approximate file size in KB
3. Short SHA of the commit it was generated from

Then stop. Do not run the file, do not summarize what is inside it, do not propose next steps.
</completion>
