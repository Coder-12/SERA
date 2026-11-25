PYTHON ?= python
PIP := $(PYTHON) -m pip
FORMAT_PATHS := sera scripts tests
# Design §11 introduces strict checks incrementally for the new package scopes.
LINT_PATHS := sera/agents sera/orchestrator sera/evaluation

.PHONY: init dev install format format-check lint test test-live test-all clean help

init:
	$(PIP) install -e .

dev:
	$(PIP) install -r requirements-dev.txt

install:
	$(PIP) install -e . -r requirements-dev.txt

format:
	$(PYTHON) -m isort $(FORMAT_PATHS)
	$(PYTHON) -m black $(FORMAT_PATHS)

format-check:
	$(PYTHON) -m isort --check-only $(FORMAT_PATHS)
	$(PYTHON) -m black --check $(FORMAT_PATHS)

lint:
	$(PYTHON) -m ruff check $(LINT_PATHS)
	$(PYTHON) -m mypy $(LINT_PATHS)

test:
	$(PYTHON) -m pytest -q

test-live:
	$(PYTHON) -m pytest -q tests/integration/test_full_integration_live.py

test-all: test

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov .coverage
	rm -rf data/*
	rm -rf downloads/*

help:
	@echo "Commands:"
	@echo "  make init         - Install the package and runtime dependencies"
	@echo "  make dev          - Install development tools"
	@echo "  make install      - Install the package and development tools"
	@echo "  make format       - Run isort and Black"
	@echo "  make format-check - Check isort and Black"
	@echo "  make lint         - Run Ruff and strict mypy on the approved scopes"
	@echo "  make test         - Run the full suite, including live tests"
	@echo "  make clean        - Remove generated files and local runtime data"
