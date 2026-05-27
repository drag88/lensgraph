# Agent A Visual Gold Audit - W_CYk2ogcDI

Video: `W_CYk2ogcDI`, Tengyu Ma, `RAG in 2025: State of the Art and the Road Forward`.

Sources read:
- `eval/corpora/ai_engineering_v0/talks.yaml`
- `eval/corpora/ai_engineering_v0/dev_gold.jsonl`
- `transcripts/ai_engineering_v0/W_CYk2ogcDI.en.vtt`
- `eval/corpora/ai_engineering_v0/visual_gold.jsonl`

Downloaded evidence windows with `yt-dlp`:
- `agent_a_clip_160_295.mp4`
- `agent_a_clip_300_610.mp4`
- `agent_a_clip_980_1112.mp4`

## Decisions

### ACCEPT - `tengyu-rag-library-analogy`

Gold span: `168-286`.

Transcript evidence:
- `00:03:04-00:03:13`: long context is like scanning/skimming an entire library for a question.
- `00:03:15-00:03:26`: fine-tuning is like reading/memorizing the library and rewiring the brain.
- `00:04:07-00:04:18`: RAG is simple/modular/reliable/fast/cheap and retrieves relevant book chapters.

Frame evidence:
- `agent_a_frame_185s_library_analogy.jpg`
- `agent_a_contact_160_295_every10s.jpg`

Visual decision: accept. The slide titled `Long-context vs fine-tuning vs RAG` visibly contains the library analogies and the RAG properties, so the row is not transcript-only.

Uncertainty: low. The slide directly bears the answer.

### ACCEPT - `tengyu-retrieval-accuracy-headroom`

Gold span: `305-410`.

Transcript evidence:
- `00:06:08-00:06:20`: he says "in this plot", averaged over about 100 datasets, accuracy is about 80%, leaving about 20% headroom.
- `00:06:23-00:06:33`: he explains some datasets are around 90-95% while others are around 60%, 20%, or 30%.

Frame evidence:
- `agent_a_frame_310s_accuracy_slide.jpg`
- `agent_a_contact_300_610_every10s.jpg`

Visual decision: accept. The visual surface is the plot titled `Significant Improvements in Retrieval Accuracy Over Last 2 Years`, visible inside the gold span. This duplicates the existing `visual-tengyu-retrieval-accuracy-plot` row in `visual_gold.jsonl`.

Uncertainty: low to medium. The plot is visible in the span, but the 80% average/headroom wording is spoken in transcript rather than printed as a slide bullet.

### ACCEPT - `tengyu-matryoshka-and-quantization`

Gold span: `405-510`.

Transcript evidence:
- `00:06:50-00:06:59`: Matryoshka learning and quantization-aware training are introduced as approaches to reduce vector storage cost.
- `00:07:13-00:07:30`: he gives the 2048-dimensional to 256-dimensional prefix example and says the accuracy loss is small.
- `00:07:40-00:08:07`: he describes lower-precision quantization and the storage/accuracy tradeoff.

Frame evidence:
- `agent_a_frame_420s_matryoshka_slide.jpg`
- `agent_a_frame_445s_matryoshka_slide.jpg`
- `agent_a_contact_300_610_every10s.jpg`

Visual decision: accept. The slide titled `Matryoshka Learning & Quantization-Aware Training` is visible in the span and lists both Matryoshka and quantization under `Lower vectorDB cost`. This duplicates the existing `visual-tengyu-matryoshka-quantization-slide` row in `visual_gold.jsonl`.

Uncertainty: low.

### ACCEPT WITH CAVEAT - `tengyu-techniques-beyond-better-embeddings`

Gold span: `503-605`.

Transcript evidence:
- `00:08:29-00:08:40`: he lists hybrid search and rerankers.
- `00:08:40-00:09:06`: he describes query decomposition and document enrichment.
- `00:09:10-00:09:38`: he cites Anthropic's blog post using large models to generate additional context per chunk.

Frame evidence:
- `agent_a_frame_470s_hybrid_search_rerankers.jpg`
- `agent_a_frame_510s_enhancing_queries_documents.jpg`
- `agent_a_contact_300_610_every10s.jpg`

Visual decision: accept as a visual candidate for the query/document enhancement portion. The `Enhancing Queries and Documents` slide is visible in the dev span and lists query decomposition, document enrichment, titles/headers/categories/authors/dates, and LLM-generated context. The hybrid-search diagram is visible immediately before the current dev span at about `470s`, so the original dev row is slightly broad/misaligned for a strict visual cut.

Uncertainty: medium. If the policy requires every original expected claim to be visible inside the exact dev span, reject or split/rewrite this row. The proposed JSONL narrows the visual question to the answer-bearing `Enhancing Queries and Documents` slide.

### REJECT - `tengyu-two-reasons-to-chunk`

Gold span: `985-1108`.

Transcript evidence:
- `00:16:46-00:17:02`: reason 1 is limited embedding context length; 100k tokens must be chunked and Voyage's longest context is around 32k.
- `00:17:23-00:17:51`: reason 2 is that feeding very long retrieved documents to an LLM is expensive.
- `00:17:57-00:18:06`: long documents can cause the model to miss context in the middle, so retrieval should focus on smaller units.

Frame evidence:
- `agent_a_frame_1024s_context_aware_auto_chunking.jpg`
- `agent_a_contact_980_1112_every10s.jpg`

Visual decision: reject. The visible slide is `Context-aware and Auto-Chunking Embeddings`, showing a long document being converted into chunks/vectors with global context. It does not show the two reasons in the expected answer: the 32k embedding context limit, 100k-token splitting, per-query LLM cost, or lost-in-the-middle issue. Those answer claims are transcript-only.

Uncertainty: low.
