# -----------------------------
# Makefile for SERA (Research Agent)
# -----------------------------

PYTHON := python
PIP := pip

# Default virtual environment name
ENV := .venv

# -----------------------------
# Setup Commands
# -----------------------------

init:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -r requirements-dev.txt

install:
	make init
	make dev

# -----------------------------
# Code Quality
# -----------------------------

format:
	black sera/ tests/
	isort sera/ tests/

lint:
	ruff check sera/ tests/
	flake8 sera/
	mypy sera/

# -----------------------------
# Testing
# -----------------------------

test:
	pytest -q

test-live:
	pytest -q tests/test_full_integration_live.py

test-all:
	pytest -q

# -----------------------------
# Utilities
# -----------------------------

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov .coverage
	rm -rf data/*
	rm -rf downloads/*

# -----------------------------
# Help
# -----------------------------
help:
	@echo "Commands:"
	@echo "  make init        - Install production dependencies"
	@echo "  make dev         - Install development tools"
	@echo "  make install     - Install everything"
	@echo "  make format      - Run black + isort"
	@echo "  make lint        - Run ruff, flake8, mypy"
	@echo "  make test        - Run all tests"
	@echo "  make clean       - Cleanup build artifacts"