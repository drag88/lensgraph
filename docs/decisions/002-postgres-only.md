# ADR 002 — Postgres as the only datastore

**Status:** Accepted
**Date:** 2026-05-24

## Context

LensGraph needs three storage shapes: relational data (talks, chunks, eval runs), vector embeddings (text and visual), and a job queue (ingestion is multi-step and resumable). The default reflex is to reach for three tools.

## Options

**Postgres + Redis + Celery.** Industry-standard split. Redis for the queue, Celery for the worker abstraction, Postgres for the rest. Familiar but introduces a second backing store and a second mental model for state.

**Postgres + RQ.** Lighter than Celery, still needs Redis.

**Postgres-only with PGMQ + pgvector.** PGMQ is a Postgres-backed message queue with FOR UPDATE SKIP LOCKED semantics, transactional enqueue/dequeue, and partitioned tables. pgvector handles embeddings. The relational store is just Postgres.

## Decision

Postgres-only. PGMQ for ingestion jobs, pgvector for embeddings, plain tables for everything else.

## Consequences

- **Pro:** one backup, one connection string, one mental model for state, one local docker-compose entry. Solo-project complexity budget conserved.
- **Pro:** transactional consistency between job state and data writes (enqueue and write in one transaction). Celery + Redis cannot offer this.
- **Pro:** straightforward eval reproducibility. A single `pg_dump` snapshots the entire system state.
- **Con:** PGMQ is younger than Celery; ecosystem of monitoring tools is thinner.
- **Con:** Postgres becomes a single point of failure. Acceptable for a single-user research tool; would not fly for production multi-tenant.

## Revisit when

- Ingestion throughput needs exceed what a single Postgres instance can serve (not happening at the v0–v2 horizon).
- A team grows around the project and people want to reuse a shared Redis for caching.
