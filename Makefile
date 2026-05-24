.PHONY: help install validate-evals validate-evals-strict validate-self-test fmt lint clean

help:
	@echo "LensGraph make targets:"
	@echo "  install                  uv sync (set up env)"
	@echo "  validate-evals           validate schemas + corpora + run self-test (CI)"
	@echo "  validate-evals-strict    same, but fail on missing transcript files (commit gate)"
	@echo "  validate-self-test       only run fixture self-test (fast schema iteration)"
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

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
