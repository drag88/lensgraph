# Visual retrieval redesign — research matrix

Date: 2026-05-29. Web-research-backed (HuggingFace cards, GitHub repos, papers,
ViDoRe/DocVQA leaderboards). Every row carries a link and a date in the Sources
section. This matrix exists to choose the next visual retrieval experiment for
LensGraph, given the two failures the v4 ablation isolated:

1. **Prefilter-limited at production `prefilter_k=200`** — 7/10 gold frames miss
   the pooled-cosine HNSW top-200.
2. **MaxSim demotion once prefilter is widened** — at K=500 all 10 gold frames
   enter the candidate set (ranks 136-444 of 514), then MaxSim ranks the
   gold-overlapping chunk at 60-172 of ~210. Gaps to the top-5 cutoff are tiny
   (+0.6 to +2.5) and consistent across K — a near-uniform similarity floor.

Constraints held throughout: Mac 24 GB unified RAM, one ML model loaded at a
time (ColQwen2.5 ≈ 7-8 GB on MPS), Postgres+pgvector is the only datastore,
~$18-33 total budget. Do NOT retire visual retrieval; do NOT switch to OCR-only.

## State of the art (2026-05)

Production visual-document retrieval has converged on a two-stage shape: cheap
candidate generation over a compact per-page descriptor, then an expensive
refine. The live debate is what each stage should be. On the candidate side, the
field has moved away from hand mean-pooling raw patch vectors — the canonical
recall-preserving compressions are now ColPali's `HierarchicalTokenPooler` (97.8%
nDCG retained at `pool_factor=3`), MUVERA fixed-dimensional encodings, and
trained single-vector document embedders (Cohere Embed v4, voyage-multimodal-3,
jina-embeddings-v4, nomic-embed-multimodal-3b) that emit one retrieval-tuned
vector per slide. On the refine side, the strongest results come from
purpose-built pointwise multimodal cross-encoders trained on ColPali-style hard
negatives — MonoQwen2-VL lifts ViDoRe mean nDCG@5 from 85.8 to 90.5, and the new
Apache-2.0 Qwen3-VL-Reranker-2B (Jan 2026) reports ViDoRe v3 60.8 — exactly the
discrimination MaxSim's per-token max-cosine cannot provide. A skeptical
counter-current ("Lost in OCR Translation?", arXiv:2505.05666) shows OCR-text
retrieval beating ColPali by large margins (Recall@5 0.64 vs 0.36) on text-dense
documents and flags that vision embeddings "struggle with slide layouts" —
directly relevant to LensGraph's slide+code corpus.

The implication for LensGraph is that the two diagnosed failures are separable
and both are known pipeline pathologies, not a model defect. Mean-pooling ~729
patches into one 128-d prefilter vector is a documented recall-killer (the Visual
RAG Toolkit, arXiv:2602.12510, shows degradation concentrates exactly at
R@100-R@500, where LensGraph's gold frames sit). The near-uniform MaxSim floor is
the classic multi-vector length bias plus the inability of raw patch cosine to
discriminate topically-similar dense slides. The cheapest, most reversible path
is therefore to fix the pipeline — corrected pooling for recall, a trained
reranker or OCR-text channel for ranking — before swapping the model, because a
model swap re-embeds everything and likely reproduces the same floor.

## Comparison matrix

| Option | Model + link | LensGraph integration cost | Expected eval impact | Runtime/RAM cost | Risk | Smallest next experiment |
|---|---|---|---|---|---|---|
| Pooling fix (late-interaction) | HierarchicalTokenPooler, pool_factor 2-3 ([colpali-engine](https://github.com/illuin-tech/colpali), v0.3.16, MIT) | Thin: `retrieve/visual.py` prefilter + frame-embed write path (`db/repos/frames.py`, `embed/colqwen.py`). No re-embed. | Targets **VisualFrameRecall@k** directly — recovers most of the 7/10 missed gold frames at far smaller K than 500. Indirect lift to ChunkTR/Grounding by getting gold into the candidate set. | Zero new model. numpy/torch over existing arrays. Negligible RAM. No MPS contention. | pgvector multi-vector prefilter heavier than single-vector HNSW (trivial at 514 frames). Does **not** fix the MaxSim ranking collapse. | Offline: load 514 frames' patch arrays, run `HierarchicalTokenPooler(pool_factor=3)`, re-score the 10-example slice with max-of-pooled as prefilter signal, count gold frames now in top-200. |
| Pooling fix (model-specific) | Gaussian/triangular same-length smoothing per [Visual RAG Toolkit (arXiv:2602.12510)](https://arxiv.org/html/2602.12510) | Medium: replace 128-d mean prefilter vector in `embed/colqwen.py` + `retrieve/visual.py`; raise prefilter_k≥500. One db migration for the new pooled column. | **VisualFrameRecall@k** is the direct target — toolkit shows ColPali's conv1d *degrades* ColQwen2.5; Gaussian smoothing is the documented fix. Should move recall from 3/10 toward 10/10. | Zero new model; same ColQwen2.5 ~7-8 GB MPS. Pooling is a CPU/numpy transform. | MaxSim floor may be intrinsic to a 3-talk corpus where every frame resembles every other — necessary-not-sufficient. | Offline numpy: recompute prefilter vector with Gaussian same-length smoothing, set prefilter_k=500, measure VisualFrameRecall@k vs mean-pooled baseline on 10 examples. |
| MaxSim aggregator fix | Length-normalization + low-IDF query-token filtering ([Late Interaction Dynamics](https://arxiv.org/html/2603.26259v2); Col-Bandit [arXiv:2602.02827](https://arxiv.org/abs/2602.02827)) | Thin: scorer change in `retrieve/visual.py` only. | Targets **VisualChunkTR@k / VisualAnswerGrounding@k** — divide chunk MaxSim by sqrt(#patches)/#frames so long concatenated chunks stop winning on count; drop generic query tokens that match unrelated "buy more H100s" patches. | Zero new model. Operates on per-query-token max-cosine values already computed. Negligible. | The skeptic's warning: on clean single-doc benchmarks these tweaks did nothing. Win depends on multi-frame-per-chunk concatenation being the actual culprit — verify first. | Offline replay at prefilter_k=500 (all 10 gold frames present): re-rank with MaxSim ÷ #patches, then again with low-IDF query tokens removed; measure gold-chunk rank shift vs current 60-172. |
| MUVERA FDE prefilter | Fixed-Dimensional Encoding ([arXiv:2405.19504](https://arxiv.org/abs/2405.19504), NeurIPS 2024) | Medium: build FDE per frame from existing patches, store as one pgvector float, HNSW over FDEs as prefilter; refine unchanged. ~100 lines, no new model/datastore. | Strong **VisualFrameRecall@k** candidate — FDE is purpose-built so single-vector ANN approximates multi-vector similarity (the guarantee mean pooling lacks). | Zero new model; deterministic transform over existing arrays. Negligible RAM. | More implementation surface than HierarchicalTokenPooler (already in colpali-engine). No slide-domain validation. Latency win irrelevant at 514 frames. | Implement minimal FDE offline, encode 514 frames, check whether all 10 gold land in top-200 by FDE-cosine; head-to-head against the HierarchicalTokenPooler experiment. |
| **Visual reranker (top pick)** | [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) (2026-01-08, **Apache-2.0**) | Thin: one `model_candidates.yaml` row + handler loading CrossEncoder, scoring **top-≥200 candidate FRAMES** (gold sits at 137-172), re-sort, map frames→chunks via existing stage (b). Two-pass: gen, swap, rerank. | Directly targets the MaxSim floor → **VisualChunkTR@5 / VisualAnswerGrounding@5** off the current 0/10. Caps at prefilter recall, so pair with prefilter_k=500. ViDoRe v3 60.8 (beats jina-reranker-m0 57.8). | 2B, ~6-8 GB RAM with images. **Cannot co-reside with ColQwen2.5 in 24 GB** — must unload ColQwen, load reranker. MPS unverified, ~0.3-1 s/image. | MPS support unverified; sentence-transformers CrossEncoder wrapper for a 2B VLM is new — confirm it loads vision weights, not text-only. | Offline: dump top-200 frames/example at prefilter_k=500, load reranker alone, score (question, frame_png), re-rank, recompute VisualFrameRecall@5 + VisualChunkTR@5. One model load, no co-residency. |
| Visual reranker (evidence-backed runner-up) | [MonoQwen2-VL-v0.1](https://huggingface.co/lightonai/MonoQwen2-VL-v0.1) (LightOn, **Apache-2.0**) | Thin: load Qwen2-VL-2B, merge LoRA via peft, prompt True/False, read logit diff. Slightly more setup than Qwen3's CrossEncoder. Rerank **top-≥200 frames**. | Best-documented deltas: ViDoRe v1 85.8→90.5 nDCG@5; biggest gains on dense-text/table pages (infovqa 88.1→93.2, tatdqa 69.4→79.0) — matches slide+code. Same fix mechanism for **ChunkTR/Grounding**. | 2B+LoRA, ~6-8 GB. Same one-model-at-a-time swap with ColQwen2.5. flash-attn is CUDA-only — use eager/sdpa on MPS. | Older base than Qwen3-VL-Reranker; LoRA+manual logit reading more brittle than a maintained API. MPS speed unverified. | Same offline harness as Qwen3 over the same top-200 frames/10 examples; run head-to-head in one script to pick the winner on dev_gold. |
| VLM-as-judge rerank | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Apache-2.0) local, or hosted via DeepInfra | Medium: pointwise "does this frame answer Q?" prompt over **top-≥200 frames** (gold at 137-172). Hosted handler in `eval/runners/providers.py`. Keep judge/generation roles separate (cross-family rule). | Reasonable diagnostic; lifts **VisualAnswerGrounding@5** over MaxSim but underperforms a trained reranker — general VLMs show topical yes-bias (the same "buy more H100s" failure). | 7B BF16 ~16-19 GB — needs 4-bit on MPS (slow). Hosted: ~200-400k vision tokens/query ≈ $0.04-0.08, so 10 examples ≈ $0.40-0.80. One-off only, within budget; not production. | Local 7B leaves no headroom; ~20 generative calls/query is the slowest option. Hosted use is a provider call — document under the existing DeepInfra ADR. | One-off ~$0.50: rerank top-200 frames/10 examples with hosted Qwen2.5-VL-7B (True/False prompt), map to chunks, measure VisualChunkTR@5 + VisualAnswerGrounding@5. |
| OCR-text rerank channel (hybrid) | bge-reranker-v2-m3 (MIT, already in repo) over frame OCR text — Pattern 1, [arXiv:2505.05666](https://arxiv.org/html/2505.05666v1) | Medium-low: OCR column on frames (db migration + `db/repos/frames.py`), one-time ingest OCR pass, replace MaxSim refine in `retrieve/visual.py` with reranker over (query, frame_ocr_text→chunk). OcrFn seam already exists. | **VisualFrameRecall@k unchanged** (still prefilter-gated). Likely large **VisualChunkTR@k** gain — query terms appear verbatim in slide text where MaxSim washes them out. **Grounding** rises from 0/10 if right chunk reaches top-5. | No new query-time GPU model. bge-reranker ~2.3 GB CPU, already in stack, loadable when ColQwen swapped out. Fits 24 GB. | Does not address prefilter recall — must pair with pooling fix or prefilter_k≥500. OCR quality on 1280×720 small code fonts may be poor. | Offline: OCR the 10 gold frames (Tesseract), map to chunks, score query-vs-OCR with installed bge-reranker over ~210 chunks; check if gold reaches top-5. |
| Better OCR for the text channel | [PaddleOCR-VL-0.9B](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) (Apache-2.0, MLX Apple-Silicon port) | Thin yaml row + handler via OcrFn seam in `eval/runners/measure_visual.py`. OCR only ColQwen top-k frames, feed BM25/bge-reranker. Needs explicit model load/unload vs ColQwen. | Highest local lift on candidate-text **VisualChunkTR@k / VisualAnswerGrounding@k** — markdown preserves reading order + chart/table text. OmniDocBench v1.5 91.76 on screen-photography (beats dots.ocr 85.34). | 0.9B, ~3-5 GB resident. Occupies the single model slot (load/unload vs ColQwen). MLX gives 10-20x over transformers-MPS. | Model-slot contention; trust_remote_code; VL models can hallucinate text not on the slide (grounding hazard). | Install MLX port, OCR the 7 missed + 10 gold frames, eyeball chart/code fidelity vs Tesseract, then bge-reranker over question-vs-PaddleOCR-VL text of prefilter_k=500 frames; check gold chunk climb. |
| Lightweight OCR (CPU, no slot contention) | [PP-OCRv5_mobile](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0, 5M params) | Thin yaml row + handler. paddleocr `.predict(png)` → concatenated lines matching OcrFn. Heavier dep tree (paddlepaddle CPU wheel). | Best RAM-to-accuracy ratio for a pure-text channel; cleaner lines than Tesseract → better BM25/reranker scores on **ChunkTR/Grounding**. Does not fix MaxSim demotion alone. | CPU-only, few hundred MB, ~0.5-1.5 s/frame on M-series. No MPS slot contention — coexists with ColQwen2.5. | macOS arm64 paddlepaddle install friction. Line text only — chart *meaning* lost (acceptable for term matching). | pip install CPU paddlepaddle+paddleocr in throwaway venv, OCR 10 gold frames, diff text quality vs Tesseract (count chart labels/code tokens captured). |
| Trained single-vector prefilter (hosted, free tier) | [voyage-multimodal-3](https://docs.voyageai.com/docs/multimodal-embeddings) (1024-d, proprietary hosted) | Cheapest hosted integration: yaml row + thin handler mirroring `providers.py`. Single vector drops into pgvector prefilter column. No patch store, no MaxSim. | High upside on **VisualFrameRecall@k** — a trained slide descriptor vs hand mean-pooling. ChunkTR/Grounding improve only if you re-rank by single-vector score, not MaxSim. | None local. ~$0.0002/frame; 529 frames ≈ $0.1, almost certainly inside the free tier ($0). | Hosted dependency (ADR amendment expected per tech-stack rules). No published ViDoRe v2 number — verify on the slice. Pin to `-3`, not 3.5. | Free tier: embed 514 frames + 10 queries with voyage-multimodal-3 (1024-d), temp pgvector column, recompute VisualFrameRecall@k vs pooled ColQwen prefilter. |
| Trained single-vector prefilter (local, MPS-native) | [nomic-embed-multimodal-3b](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b) / colnomic-3b (license: verify on card) | Medium: local handler. nomic-3b single-vector → prefilter column; colnomic-3b drops into existing patch+MaxSim path (same ColPali lineage as ColQwen2.5). | nomic-3b: solid **VisualFrameRecall@k** lift (MPS-native, purpose-trained). colnomic-3b: **diagnostic** — if it fixes the chunk floor in your existing MaxSim code, your code is fine and ColQwen2.5 was the weak link. ViDoRe v2 58.8 / 61.2. | 3B, ~6-7 GB. Fits single slot. MPS officially supported (`device_map='mps'` on card). | License unknown — verify (Nomic mixes Apache and NC). 58.8 dense is below ColPali-style leaders. | (1) nomic-3b single-vector → temp pgvector column, recompute VisualFrameRecall@k. (2) swap colnomic-3b into the existing patch+MaxSim path, recompute VisualChunkTR@k. |
| Model swap (open license) | [ColModernVBert](https://huggingface.co/ModernVBERT/colmodernvbert) (250M, **MIT**, 2025-10) | Full re-embed of 529 frames + yaml row + handler; prefilter/MaxSim re-fit. Not a rewrite but not free. | Speculative on this corpus — the problem is pipeline, not model quality. Swapping without fixing the aggregator likely reproduces the floor. | 250M, ~0.5-1 GB. Leaves headroom for a second model. MPS unverified (support on unmerged colpali-engine branch). | Re-embed cost, unmerged branch, unverified MPS, sub-1B generalization concerns. Poor first move. | Deferred. Only after pooling/aggregator fixes are exhausted: embed 3 talks, run 10-example slice, compare VisualFrameRecall@k against fixed-pipeline ColQwen2.5. |

## Specifically addressing the two failures

### (a) Pooled-prefilter recall miss (7/10 gold frames miss HNSW top-200 at K=200)

A mean-pooling-as-descriptor problem, not a model problem: collapsing ~729
patches of a dense slide into one 128-d vector washes out the few discriminative
patches, and the Visual RAG Toolkit (arXiv:2602.12510) shows degradation
concentrates precisely at R@100-R@500 — exactly where LensGraph's gold frames
rank (136-444). Four options fix it, all over the existing patch arrays with no
re-embedding. Cheapest and most reversible: **HierarchicalTokenPooler** (already
in colpali-engine, 97.8% nDCG retention at pool_factor=3).
**Model-specific Gaussian/triangular smoothing** is the toolkit's ColQwen2.5-correct
pooling (it warns ColPali's conv1d *degrades* ColQwen2.5 via double-smoothing).
**MUVERA FDE** is the principled alternative. A **trained single-vector embedder**
(voyage-multimodal-3 free tier, or local nomic-embed-multimodal-3b) replaces the
descriptor entirely. None of these fix ranking — they only get gold into the
candidate set. Independent of which wins, **raise prefilter_k to ≥500**, since the
diagnosis confirms all 10 gold frames enter at K=500.

### (b) MaxSim demotion once prefilter is widened (gold chunk falls to rank 60-172)

The near-uniform MaxSim floor with +0.6 to +2.5 unit gaps is multi-vector length
bias plus the inability of raw patch cosine to discriminate topically-similar
same-talk slides. Strongest fix: a **trained pointwise cross-encoder reranker** —
Qwen3-VL-Reranker-2B (Apache-2.0) or MonoQwen2-VL — trained on hard negatives to
do exactly the discrimination MaxSim cannot. **Because gold sits at ranks 137-172,
any reranker must score the top ≥200 frames, never top-100.** Cheap algorithmic
levers: **length normalization** (divide chunk MaxSim by sqrt(#patches) or
#frames) and **low-IDF query-token filtering** (drop generic tokens matching
unrelated patches), both pure scorer changes in `retrieve/visual.py`. The hybrid
alternative is an **OCR-text rerank channel**: query terms appear verbatim on the
slides, and the installed bge-reranker-v2-m3 scores question-vs-slide-text where
MaxSim washes the terms out. A general **VLM-as-judge** (Qwen2.5-VL-7B) is a
reasonable diagnostic but shows topical yes-bias, so it underperforms a trained
reranker. Caveat: the gaps are tiny *and consistent across K*, hinting the
per-token signal may be intrinsically saturated on a 3-talk corpus; if so,
aggregator tweaks fix recall-into-candidates but not ranking, and a trained
reranker or text channel is required on top.

## Recommended smallest reversible experiment

Run a **two-phase offline hybrid visual diagnostic** in `eval/runners/`, scored
against the existing metrics, writing nothing to production code or the DB until
results justify it. This implements the "Pooling fix (late-interaction)" row
first, then the "Visual reranker (top pick)" row.

**Phase A — recall fix (no new model).** Recompute the prefilter signal over the
existing 514 frames' patch arrays using `HierarchicalTokenPooler(pool_factor=3)`
(and, as a second arm, Gaussian same-length smoothing), set `prefilter_k=500`,
and measure `VisualFrameRecall@{5,10,20,200}` on the 10-example visual-required
slice. This is the smallest possible step: pure numpy/torch over arrays already
in the DB, zero model load, zero MPS contention, fully reversible (delete the
script). It is also a precondition — ranking fixes are pointless until gold
frames are in the candidate set.

**Phase B — ranking fix (one model, swapped in).** Only if Phase A lands gold
frames in the candidate set: dump the **top-200** candidate frames per example,
unload ColQwen2.5, load Qwen3-VL-Reranker-2B *alone* (respects the
one-model-at-a-time 24 GB limit), score each (question, frame_png), re-sort, map
frames→chunks via the existing stage (b), and recompute
`VisualChunkTR@{5,10,20}` + `VisualAnswerGrounding@{5,10}`. Optional comparison
arms on the same top-200: the frame-OCR-text channel (Tesseract → installed
bge-reranker-v2-m3) and a one-off hosted Qwen2.5-VL-7B judge (~$0.50).

This is a hybrid *visual* diagnostic — the visual prefilter and visual reranker
remain the spine, OCR/VLM are optional comparison arms — so it neither retires
visual retrieval nor switches to OCR-only.

### Acceptance criteria (tied to existing metrics)

- **VisualFrameRecall@200 ≥ 9/10** after Phase A (recall bug is fixed only if gold
  frames actually enter the candidate set).
- **VisualChunkTR@5 ≥ 5/10** after Phase B reranking the top-200 (gold chunk climbs
  from 60-172 into top-5 for at least half the slice).
- **VisualAnswerGrounding@5 strictly > 0/10**, target ≥4/10 — the metric that proves
  the right slide reached the generator. Any positive row requires the structured
  audit payload first (see `audit_payload_design.md`).
- If FrameRecall@200 ≥ 9/10 but ChunkTR@5 stays low, the failure is isolated to
  ranking → promotes the trained-reranker / OCR-text arm. If even the reranker
  cannot lift ChunkTR, the MaxSim floor is intrinsic to the 3-talk corpus and the
  corpus needs more talks before the visual channel can be evaluated fairly.

### Files touched if later committed

A new offline script under `eval/runners/`; an alternate prefilter pooling
function in `embed/colqwen.py` (offline-callable, not yet wired into the write
path); one `eval/config/model_candidates.yaml` row for the reranker candidate. No
db migration, no `retrieve/visual.py` production edit, no schema change until the
experiment justifies it.

## Sources

**Late-interaction / aggregator / pooling**
- colpali-engine (HierarchicalTokenPooler, ColQwen2.5) — https://github.com/illuin-tech/colpali — v0.3.16, 2026-05-12
- Late Interaction Dynamics working notes — https://arxiv.org/html/2603.26259v2 — 2026-04
- Col-Bandit (zero-shot query-time patch pruning) — https://arxiv.org/abs/2602.02827 — 2026-02-04
- MUVERA Fixed-Dimensional Encoding — https://arxiv.org/abs/2405.19504 — 2024-05 (NeurIPS 2024)
- ColModernVBert / colmodernvbert — https://huggingface.co/ModernVBERT/colmodernvbert — 2025-10-03 (arXiv 2510.01149)
- ColSmol-500M — https://huggingface.co/vidore/colSmol-500M — 2025

**Embeddings (prefilter descriptor)**
- Cohere Embed v4 (embed-v4.0) — https://docs.cohere.com/changelog/embed-multimodal-v4 — 2025-04-15
- voyage-multimodal-3 / 3.5 — https://docs.voyageai.com/docs/multimodal-embeddings — multimodal-3 2024-11
- jina-embeddings-v4 — https://huggingface.co/jinaai/jina-embeddings-v4 — 2025-06-23 (arXiv 2506.18902)
- nomic-embed-multimodal-3b / colnomic-3b — https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b — 2025
- Google Vertex AI multimodalembedding@001 — https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings — GA 2026
- Gemini Embedding 2 — https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/ — 2026-03-10 (Public Preview)
- SigLIP 2 (so400m-patch16-384) — https://huggingface.co/google/siglip2-so400m-patch16-384 — 2025-02-20
- jina-clip-v2 — https://huggingface.co/jinaai/jina-clip-v2 — 2024-12 (arXiv 2412.08802)
- OpenAI CLIP ViT-L/14-336 — https://huggingface.co/openai/clip-vit-large-patch14-336 — 2022

**OCR (frame-text channel)**
- Tesseract 5 / pytesseract — https://github.com/tesseract-ocr/tesseract — 5.5.0, 2024-11
- PP-OCRv5_mobile (PaddleOCR 3.x) — https://huggingface.co/blog/baidu/ppocrv5 ; https://github.com/PaddlePaddle/PaddleOCR — 2025
- PaddleOCR-VL-0.9B (+ Apple Silicon guide) — https://huggingface.co/PaddlePaddle/PaddleOCR-VL — 0.9B 2025-11-07
- granite-docling-258M / SmolDocling — https://huggingface.co/ibm-granite/granite-docling-258M ; https://huggingface.co/ds4sd/SmolDocling-256M-preview — 2025
- GOT-OCR2.0 — https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf — 2024-09 (arXiv 2409.01704)
- Surya OCR — https://github.com/datalab-to/surya — maintained 2025-2026
- dots.ocr — https://huggingface.co/rednote-hilab/dots.ocr — 2025-07-30
- olmOCR-2-7B — https://huggingface.co/allenai/olmOCR-2-7B-1025 — Oct 2025

**VLM rerank / judge**
- Qwen3-VL-Reranker-2B — https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B — 2026-01-08 (arXiv 2601.04720)
- MonoQwen2-VL-v0.1 (LightOn) — https://huggingface.co/lightonai/MonoQwen2-VL-v0.1 — 2024 (card updated 2025-06-20)
- jina-reranker-m0 — https://huggingface.co/jinaai/jina-reranker-m0 — 2025 (CC-BY-NC, license-blocked for public repo)
- Qwen2.5-VL-7B-Instruct — https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct — 2025-01-26
- Qwen2.5-VL-3B-Instruct — https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct — 2025-01 (Qwen Research License)
- InternVL3-8B / InternVL3.5-2B — https://huggingface.co/OpenGVLab/InternVL3-8B — 2025-04-11
- VLMs 2025 overview (SmolVLM2 / Moondream2 / MiniCPM-V) — https://huggingface.co/blog/vlms-2025 — 2025

**Hybrid patterns**
- "Lost in OCR Translation?" — https://arxiv.org/html/2505.05666v1 — 2025-05
- ColPali (ICLR 2025) — arXiv:2407.01449 — 2024-07
- Vespa vision-driven document retrieval — https://blog.vespa.ai/the-rise-of-vision-driven-document-retrieval-for-rag/ — 2024-2025
- HF cookbook: multimodal RAG (ColQwen2 + MonoQwen2-VL + Qwen2-VL) — https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_reranker_and_vlms — 2025
- Visual RAG Toolkit (ColQwen2.5 Gaussian/triangular pooling) — https://arxiv.org/html/2602.12510 — 2026

> Verify-before-trust note: several 2026-dated arXiv IDs and the
> PaddleOCR-VL/Qwen3-VL-Reranker release dates were surfaced by web search this
> session and should be re-confirmed against the primary source before the
> implementation session commits to a specific model/version.
