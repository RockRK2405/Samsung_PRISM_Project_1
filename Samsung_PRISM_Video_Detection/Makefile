# =====================================================================
# Samsung PRISM — Video Detection Module — developer shortcuts
# =====================================================================
# Convenience targets for the common workflows. Nothing here is
# load-bearing — every command it wraps can be run directly.
# =====================================================================

PYTHON      ?= python3.11
VENV        ?= .venv
VENV_PY     := $(VENV)/bin/python
VENV_PIP    := $(VENV)/bin/pip

.PHONY: help venv install install-dev lint format typecheck test clean

help:
	@echo "Samsung PRISM — Video Detection Module"
	@echo ""
	@echo "  make venv         Create the virtual environment at $(VENV)/"
	@echo "  make install      Install runtime dependencies into $(VENV)/"
	@echo "  make install-dev  Install runtime + dev dependencies"
	@echo "  make lint         Run ruff (lint only)"
	@echo "  make format       Run ruff --fix + ruff format"
	@echo "  make typecheck    Run mypy over src/"
	@echo "  make test         Run pytest"
	@echo "  make clean        Remove caches (does NOT delete .venv or data)"

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip setuptools wheel

install: venv
	$(VENV_PIP) install -r requirements.txt

install-dev: venv
	$(VENV_PIP) install -r requirements.txt -r requirements-dev.txt

lint:
	$(VENV)/bin/ruff check src tests scripts

format:
	$(VENV)/bin/ruff check --fix src tests scripts
	$(VENV)/bin/ruff format src tests scripts

typecheck:
	$(VENV)/bin/mypy src

test:
	$(VENV)/bin/pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache"  -exec rm -rf {} +
	find . -type d -name ".ruff_cache"  -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
