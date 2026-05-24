.PHONY: help install validate-evals validate-self-test fmt lint clean

help:
	@echo "LensGraph make targets:"
	@echo "  install              uv sync (set up env)"
	@echo "  validate-evals       validate schemas + corpora"
	@echo "  validate-self-test   run fixture-based schema tests"
	@echo "  fmt                  ruff format"
	@echo "  lint                 ruff check"

install:
	uv sync

validate-evals:
	uv run python eval/validate.py

validate-self-test:
	uv run python eval/validate.py --self-test

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
