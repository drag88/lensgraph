# Visual-required curation — W_CYk2ogcDI (Tengyu Ma, RAG in 2025)

## Why a fresh pass

The existing `visual_gold.jsonl` rows on this video (`visual-tengyu-library-analogy-slide`, `visual-tengyu-retrieval-accuracy-plot`, `visual-tengyu-matryoshka-quantization-slide`, `visual-tengyu-query-document-enhancement-slide`) all pivot on slide *content* that the speaker also reads aloud. Text retrieval over the transcript already passes them 8/8, so they cannot show visual lift.

The rows below pivot on slide *elements that the speaker never reads aloud* — axis labels, tick values, marker shapes, legend colors, named model points on a Pareto plot, and box labels inside a diagram. The transcript only gestures at the figure ("you can see that in this plot", "the trade-off on the right of the figure"). A text-only system has no shot at these answers.

## Method

1. Read `agent_a_notes.md` and the existing dev_gold + visual_gold rows for the video to understand which slides were already covered.
2. Walked the key-frame inventory (`W_CY_key_01..06.jpg`) and per-window contact sheets, then opened the highest-info frames at 310s, 370s, 420s, 445s, 470s, 590s, 1024s, 1082s.
3. For every candidate slide element, grepped the VTT to confirm the answer phrase is **not** in the transcript:
   - `NDCG`, `ndcg` — zero hits.
   - `million tokens`, `price per`, `0.01`, `0.10`, `axis` — zero hits.
   - `Voyage-3-large`, `Voyage-3.5`, `Cohere`, `OpenAI v3`, `Cohere v4` — zero hits (he only says "voyage and other companies" and "open eye").
   - `1/100`, `1/10x`, `1x` (as storage ratio), `float`, `int8`, `binary` — zero hits.
   - `BM25`, `keyword search`, `aggregat`, `re-order`, `reorder`, `Other Search` — zero hits.
4. Drafted 4 candidate rows. All bounded to the same dev_gold span windows already used on this video (305-410 / 405-510 / 460-520) so they slot cleanly into the existing retrieval test setup.

## Per-row evidence

### `visual-required-tengyu-accuracy-plot-axes` (305-410s)

**Slide pivot:** y-axis label "Retrieval Quality (NDCG@10)", x-axis label "Price per million tokens", x-axis tick values "$0.01" and "$0.10".

**Frame evidence:**
- `agent_a_frame_310s_accuracy_slide.jpg` — full slide, labels and tick values legible.
- `W_CY_key_01.jpg`, `W_CY_key_02.jpg`, `W_CY_key_03.jpg` — same plot, alternate moments inside the span.
- `agent_a_frame_370s_accuracy_slide.jpg` — confirms the plot remains on screen across the span.

**Transcript snippet that does NOT answer:**
> "you can see that in this plot you know we are averaging over about a 100 data sets and accuracy is about 80%" (VTT lines 1341-1359)

He never says "NDCG@10", never says "price per million tokens", never reads the $0.01 / $0.10 ticks. The axes carry the answer.

**Confidence:** HIGH that this is visual-required.

### `visual-required-tengyu-accuracy-plot-models` (305-410s)

**Slide pivot:** color legend (Voyage AI / Cohere / OpenAI), named points (voyage-3-large at top-right, voyage-3.5, voyage-3.5-lite, voyage-3, Voyage-3-lite, Cohere v4, OpenAI v3 large, OpenAI v3 small), Pareto-frontier interpretation.

**Frame evidence:** same set as the axes question — legend in the upper-left of the chart, named points scattered across the plot. `W_CY_key_01.jpg` is the cleanest.

**Transcript snippet that does NOT answer:**
> "voyage and other uh uh uh companies has uh uh offered is this so-called matashka learning" (lines 1477-1495); later "but voyage you know is doing a great job here because you know you can save uh 100x but still doing better than open eye" (lines 1709-1727).

He names "voyage" and "open eye" loosely but never the specific model SKUs (voyage-3-large, OpenAI v3 large, Cohere v4) or their relative positions. The Pareto-frontier story is fully visual.

**Confidence:** HIGH.

### `visual-required-tengyu-matryoshka-chart-axes` (405-510s)

**Slide pivot:** right-hand chart on the Matryoshka slide — x-axis "Relative Storage Costs" with ticks "1/100x", "1/10x", "1x"; legend "Embedding Quantization" with three shapes (circle = float, square = int8, triangle = binary).

**Frame evidence:**
- `agent_a_frame_420s_matryoshka_slide.jpg` — full slide, x-axis ticks and legend both legible.
- `W_CY_key_04.jpg` — same slide, slightly later in the span.
- `agent_a_frame_445s_matryoshka_slide.jpg` — confirms slide stays on screen.

**Transcript snippet that does NOT answer:**
> "and you can see the uh the trade-off on the right of the figure uh here. So basically you can save you know 100x you know at least 10x without losing much." (lines 1655-1679)

He says "100x" and "10x" as savings ratios but never reads the x-axis label "Relative Storage Costs", never enumerates the three quantization shapes (float / int8 / binary), and never says "Embedding Quantization". The legend is the answer source.

**Confidence:** HIGH.

### `visual-required-tengyu-hybrid-search-diagram` (460-520s)

**Slide pivot:** the "0. Hybrid Search and Rerankers" flow diagram — keyword-search box explicitly labeled "Keyword Search (BM25)", and two annotation captions ("Hybrid search combines the benefits of semantic and keyword search" / "Aggregates and re-orders the results of the initial steps").

**Frame evidence:**
- `agent_a_frame_470s_hybrid_search_rerankers.jpg` — full diagram with both annotations and the BM25 box legible.
- `W_CY_key_06.jpg` — same slide, alternate moment.

**Transcript snippet that does NOT answer:**
> "one of them is to use hybrid search and reankers you can use you know lexical search and other kind of search and then combine them with a reancher" (lines 1845-1873).

He says "hybrid search", "lexical search", and "reanker" — but never says BM25, never says "aggregates and re-orders", and never gives the exact annotation language. BM25 is the load-bearing answer token and is only on the diagram.

**Confidence:** HIGH that BM25 is visual-required. The two annotation captions are paraphrased loosely in voice, so a generous judge might give partial credit from transcript alone — that's the main reason this row is `medium` difficulty, not `easy`. If the eval harness wants a pure-signal row, drop the annotation claims and keep only the BM25 claim.

## Note on span boundaries

The hybrid-search slide is visible roughly 460-500s but the existing `tengyu-techniques-beyond-better-embeddings` dev_gold span is 503-605, which puts the diagram mostly *before* the span. I chose 460-520 so the gold span actually contains the slide. This is a minor neighborhood drift, not a new chapter. If the harness requires the dev_gold envelope to remain exactly 503-605, drop this row — the other three live fully inside existing dev_gold windows.

## What I did NOT propose

- **Domain-specific embeddings slide (~590s)** — the slide names `voyage-code-3` and the right column says "finance-3 and law-3 are coming in a few months". The speaker says "voyage also has code embedding... finance and law coming" in slightly different phrasing later in the talk (transcript window ~9:40 onward), and I could not cleanly establish that the SKU names are only on the slide. Skipped to keep precision high.
- **Auto-chunking interface (~1082s)** — the slide shows a UI mock with specific configurable parameters, but the contact-sheet resolution did not let me confidently transcribe the UI labels. A higher-res frame extraction would unlock 1-2 more rows here.
- **Flavor-of-the-month RAG list (~590s)** — visible list (Self-RAG, Golden-Retriever, Corrective RAG, Speculative RAG, GraphRAG, Iterative/recursive retrieval). Verified with `agent_a_frame_590s_domain_specific_embeddings.jpg` is actually the domain-embeddings slide; the flavor-of-the-month slide is captured elsewhere but I did not confirm whether the speaker reads all six names aloud. Skipped pending a fresh transcript pass.
