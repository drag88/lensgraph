.PHONY: help install validate-evals validate-evals-strict validate-self-test phase0-gate test fmt lint clean

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
