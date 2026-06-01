---
description: Installed vs planned dependencies, with provider and licensing notes
---

# Tech Stack

## Language

- Python ≥ 3.11 (set in `pyproject.toml`)

## Package Management

- `uv` (lockfile committed: `uv.lock`)
- `make install` runs `uv sync`

## Installed Dependencies (v0 eval harness)

| Package | Purpose | Notes |
|---|---|---|
| `jsonschema[format-nongpl]>=4.20` | Schema validation, Draft 2020-12 | `format-nongpl` extra is required so `date-time` and `uri` formats actually validate; the validator wires `FormatChecker` explicitly |
| `pyyaml>=6.0` | `talks.yaml` and `model_candidates.yaml` parsing | safe_load only |

## Installed Dev Dependencies

| Package | Purpose | Notes |
|---|---|---|
| `ruff>=0.6` | Format + lint | Config in `pyproject.toml`: line-length 100, rules E/F/I/B/UP/N, ignore E501 (formatter owns line length), double quotes |
| `pytest>=8.0` | Test runner | Currently only used implicitly via the validator self-test (`make validate-self-test`) |

## Planned Dependencies (per-phase, not yet installed)

### Phase 1 — ingestion + retrieval

| Component | Choice | License | Hardware |
|---|---|---|---|
| ASR | `whisperx` large-v3 | BSD-2 | Local MPS |
| Video / caption fetch | `yt-dlp` | Unlicense | Local |
| Text embeddings | `BAAI/bge-m3` via sentence-transformers or FlagEmbedding (all three channels: dense, sparse, multi-vector) | MIT | Local CPU/MPS |
| Visual embeddings | ColQwen2.5 via `colpali-engine` | Apache-2.0 | Local MPS, fall back to Modal |
| Reranker | `BAAI/bge-reranker-v2-m3` | MIT | Local CPU |
| Database | Postgres 16 + `pgvector` 0.7+ (sparsevec required) + `pgmq` | PostgreSQL | docker-compose |
| Postgres client | `psycopg[binary,pool]` | LGPL exception | — |
| Migrations | TBD (likely `alembic` or raw SQL via `psycopg`) | — | — |

### Phase 2 — bakeoff (hosted)

| Component | Candidates | Provider | License |
|---|---|---|---|
| Generator | gemma-4-31b, qwen3-235b-a22b-instruct, deepseek-v3.2 | DeepInfra (primary), OpenRouter (failover) | Open-weight |
| Judge | Cross-family vs chosen generator (gemma / qwen / deepseek) | DeepInfra | Open-weight |
| Cheap extraction | gemma-4-e4b, qwen3-8b (local first); fall back to chosen hosted generator | Local then DeepInfra | Open-weight |
| Premium triangulation (test_gold only) | claude-sonnet-4-6, gpt-5.5 | Anthropic, OpenAI | Proprietary |

### Phase 3 — agent loop

| Component | Choice | License |
|---|---|---|
| Agent framework | `langgraph` | MIT |
| Tracing | `langfuse` (self-host or cloud) | MIT |

### Phase 4 — surface

| Component | Choice |
|---|---|
| API | `fastapi` + `uvicorn` |
| UI | Next.js 15 + shadcn/ui + Tailwind |

## Inference Provider Policy

- **DeepInfra is primary.** It is the only host carrying the full open-weight candidate set (Gemma 4 31B, Qwen3-235B, DeepSeek V3.2).
- **OpenRouter is optional failover** — used only if DeepInfra is down or pricing changes mid-bakeoff.
- **Groq is excluded** until they add Gemma 4 31B, DeepSeek V3.2, and Qwen3-235B. Documented in `eval/config/model_candidates.yaml` under `inference_providers.excluded`.
- **Adding a provider requires an ADR amendment to 004.** Do not add a provider quietly.

## Forbidden by Decision

| Package / Tool | Reason | Decision |
|---|---|---|
| `langchain` (core) | Cyclical agent loop is awkward in LCEL; LangGraph supersedes | ADR 001 |
| `llama-index` | Multimodal video indexing is weaker than direct ColPali integration | ADR 001 |
| `redis` | Single-datastore invariant; PGMQ covers the queue need | ADR 002 |
| `celery`, `rq`, `rabbitmq` | Same | ADR 002 |
| Any second vector DB | pgvector handles dense + sparse + multi-vector | ADR 002 |

## CI

- `.github/workflows/ci.yml` on every push/PR to `main`:
  - `uv sync --frozen`
  - `make validate-evals` (schemas + corpora + fixture self-test)
  - `make lint`
- `make phase0-gate` is intentionally **not** in CI. It is a local commit gate (CI cannot scope it per-directory). Discipline lives in CLAUDE.md hard rule 1.
