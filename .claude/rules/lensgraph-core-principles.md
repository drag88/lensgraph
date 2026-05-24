# LensGraph Core Principles

These are the load-bearing decisions. Every PR is judged against them.

## 1. The eval is the product

Not the demo, not the UI, not the model. The chunking ablation methodology and the per-strategy metrics on a locked test set are the artifact that justifies the project's existence.

**Why:** the differentiator vs every other portfolio RAG project. Almost no solo-built systems have a published methodology with judge-human kappa.

**How to apply:** any change that does not improve the eval, improve the metrics, or improve methodology rigor is suspect. UI polish in week 11 is fine. UI polish in week 4 is procrastination.

## 2. Schema-first, code-second

JSON Schemas are authoritative. Curation, ingestion, retrieval, and reporting code all conform to schemas. Schema changes are deliberate and reviewed.

**Why:** the alternative is freeform dicts that drift between curator, ingester, and judge.

**How to apply:** before adding a new field anywhere in the system, update the schema and fixtures. Run `make validate-self-test`. Then write code.

## 3. Cross-family judge discipline

Curate with model family A, judge with model family B. Never the same.

**Why:** LLMs are sycophantic about their own outputs. Same-family judging inflates faithfulness by 5–15 percentage points in published studies.

**How to apply:** Anthropic curates → OpenAI judges; OpenAI curates → Anthropic judges. Voyage rerankers are model-family-independent and safe to use throughout.

## 4. Lock the test set; treat it as a contract

`test_gold.jsonl` is hashed in the README. Changing it without resetting downstream metrics is methodological fraud, even on a portfolio project.

**Why:** without this discipline, every "X% improvement" claim is unverifiable.

**How to apply:** during development, only `dev_gold` informs decisions. The test set is unlocked only for final eval runs whose numbers go into the report.

## 5. One framework, one datastore

LangGraph only. Postgres only. PGMQ for queues. pgvector for embeddings.

**Why:** complexity is the budget killer for a solo 12-week project. Each additional tool doubles operational surface.

**How to apply:** if a problem seems to require adding a new tool, the right answer is usually "do it with what you have, less cleverly." Genuine exceptions go through an ADR.

## 6. Conference talks as the launch corpus

Engineering conference talks (AI Engineer Summit, NeurIPS, Strange Loop, PyData, LangChain Interrupt) are the v0–v1 corpus.

**Why:** recruiter-recognizable speakers, multimodal richness (slides + code + whiteboard), manageable copyright posture, ground truth is easy to verify.

**How to apply:** when expanding the corpus, prefer talks that satisfy these criteria. Earnings calls, podcasts without slides, and paywalled material are deferred to v2.

## 7. Public and incremental

GitHub public from day one. Weekly progress posts. Eval dashboard public.

**Why:** the project's value is partly performative — it teaches recruiters how I work, not just what I built. Private repos do not do that.

**How to apply:** do not hide work-in-progress. Do not wait until "it's done" to commit. Ship the embarrassing version of week 2 — the trajectory is the point.
