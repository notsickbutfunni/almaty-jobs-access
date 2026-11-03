# Thin convenience wrapper around uv.
# Optional: on Windows without make, run the uv commands directly.
.PHONY: help install lint format test sample

help:        ## List available targets
	@grep -E '^[a-z-]+:.*##' Makefile

install:     ## Create .venv and install all dependencies
	uv sync --all-groups

lint:        ## Check formatting and lint rules
	uv run ruff format --check .
	uv run ruff check .

format:      ## Auto-format code and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

test:        ## Run the test suite (offline, sample data)
	uv run pytest

sample:      ## Regenerate the synthetic sample dataset
	uv run almaty-access sample
