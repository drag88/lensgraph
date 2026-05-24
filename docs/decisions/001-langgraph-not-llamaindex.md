# ADR 001 — LangGraph as the agent layer, not LangChain or LlamaIndex

**Status:** Accepted
**Date:** 2026-05-24

## Context

The LensGraph query flow is cyclical. Verify can re-enter Retrieve with a refined query if confidence is below threshold. This is not a chain; it is a state machine with conditional edges. Three options were considered.

## Options

**LangChain (core).** Designed around linear chains and the LangChain Expression Language. Cyclical flows require runnable branching that fights the abstraction. The framework's direction has shifted toward LangGraph as the primary orchestration layer; LangChain core feels like legacy.

**LlamaIndex.** Strong at indexing and data ingestion abstractions, weaker at multi-step agent control flow. Its multimodal indexing for video is thinner than what is achievable by directly integrating ColPali via the `colpali-engine` library. Using both LangGraph and LlamaIndex introduces two mental models for state.

**LangGraph.** Explicitly low-level orchestration. State is a typed object. Nodes are functions. Edges are conditional. Loops are first-class. The model maps cleanly to Plan → Retrieve → Rerank → Verify → Generate → Cite with a Verify → Retrieve back-edge.

## Decision

LangGraph as the only agent framework. Custom Python for ingestion and retrieval glue. No LangChain core. No LlamaIndex.

## Consequences

- **Pro:** one mental model, clean state, easy to trace via Langfuse.
- **Pro:** custom ingestion lets ColPali integrate as a first-class retrieval channel rather than being shoehorned into LlamaIndex's `MultiModalLLM` abstraction.
- **Con:** more code to write than reaching for LlamaIndex's `VideoIndex` (which would be a worse abstraction anyway).
- **Con:** LangGraph is opinionated about state shape; refactoring node signatures later is friction.

## Revisit when

- LlamaIndex ships a column-level retrieval API competitive with ColPali patch embeddings.
- LangGraph deprecates the typed-state model in favor of something materially different.
