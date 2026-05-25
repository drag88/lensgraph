.PHONY: help install validate-evals validate-evals-strict validate-self-test phase0-gate test fmt lint clean db-up db-down db-build db-verify-ext db-migrate db-reset logs ingest ingest-all bakeoff-prep

CORPUS ?= ai_engineering_v0

help:
	@echo "LensGraph make targets:"
	@echo "  install                  uv sync (set up env)"
	@echo "  validate-evals           validate schemas + corpora + run self-test (CI)"
	@echo "  validate-evals-strict    same, but fail on missing transcript files (commit gate)"
	@echo "  validate-self-test       only run fixture self-test (fast schema iteration)"
	@echo "  phase0-gate              fail unless >= 10 verified non-negative examples across >= 2 talks (commit gate before implementation code)"
	@echo "  test                     run pytest under eval/tests/"
	@echo "  fmt                      ruff format"
	@echo "  lint                     ruff check"
	@echo "  db-up                    docker compose up -d postgres"
	@echo "  db-down                  docker compose stop postgres"
	@echo "  db-build                 docker compose build postgres"
	@echo "  db-verify-ext            create pgmq + vector extensions (idempotent)"
	@echo "  db-migrate               apply pending raw-SQL migrations"
	@echo "  db-reset                 drop volume + recreate postgres (destructive, prompts)"
	@echo "  logs                     tail worker log ($$LENSGRAPH_LOG_PATH or default)"
	@echo "  ingest                   ingest one video by id (VIDEO_ID=<id>) via local transcripts"
	@echo "  ingest-all               ingest a whole corpus (CORPUS=<name>) via local transcripts"
	@echo "  bakeoff-prep             report chunks/embeds readiness for corpus (CORPUS=<name>)"

install:
	uv sync

validate-evals:
	uv run python eval/validate.py

validate-evals-strict:
	uv run python eval/validate.py --strict

validate-self-test:
	uv run python eval/validate.py --self-test

phase0-gate:
	uv run python eval/validate.py --strict --min-gold 10 --min-talks 2

test:
	uv run pytest eval/tests/ -q

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

db-up:
	docker compose up -d postgres

db-down:
	docker compose stop postgres

db-build:
	docker compose build postgres

db-verify-ext:
	docker compose exec -T postgres psql -U lensgraph -d lensgraph -c "CREATE EXTENSION IF NOT EXISTS pgmq; CREATE EXTENSION IF NOT EXISTS vector;"

db-migrate:
	uv run python -m db.migrate

db-reset:
	@read -p "This will DROP the postgres volume and all data. Type 'yes' to continue: " ans; \
	 [ "$$ans" = "yes" ] || { echo "aborted"; exit 1; }
	docker compose down -v
	docker compose up -d postgres
	@echo "Waiting for postgres to become healthy..."
	@until docker compose exec -T postgres pg_isready -U lensgraph -d lensgraph >/dev/null 2>&1; do sleep 1; done
	POSTGRES_DSN="postgresql://lensgraph:lensgraph@localhost:5432/lensgraph" $(MAKE) db-migrate
	@echo "db-reset: complete — migrations applied"

logs:
	tail -f $${LENSGRAPH_LOG_PATH:-./.lensgraph/logs/workers.log}

ingest:
	uv run python -m ingest.cli $(VIDEO_ID)

ingest-all:
	uv run python -m ingest.cli_all --corpus $(CORPUS)

bakeoff-prep:
	uv run python -m scripts.bakeoff_prep --corpus $(CORPUS)
