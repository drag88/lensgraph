.PHONY: help install validate-evals validate-evals-strict validate-self-test phase0-gate test fmt lint clean db-up db-down db-build db-verify-ext db-migrate db-reset logs

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
	@echo "  db-migrate               apply db migrations (placeholder until step 2)"
	@echo "  db-reset                 drop volume + recreate postgres (destructive, prompts)"
	@echo "  logs                     tail worker log ($$LENSGRAPH_LOG_PATH or default)"

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
	@echo "db-migrate: db/migrate.py lands in step 2 — not yet implemented"

db-reset:
	@read -p "This will DROP the postgres volume and all data. Type 'yes' to continue: " ans; \
	 [ "$$ans" = "yes" ] || { echo "aborted"; exit 1; }
	docker compose down -v
	docker compose up -d postgres
	@echo "db-reset: volume recreated; db-migrate is a placeholder"

logs:
	tail -f $${LENSGRAPH_LOG_PATH:-./.lensgraph/logs/workers.log}
