MAIN_PROGRAM := src/__main__.py
VENV         := .venv
STAMP        := $(VENV)/.install-stamp

FLAKE8_FLAGS += --color always
FLAKE8_FLAGS += --exclude .mypy_cache,.pytest_cache,.ruff_cache,.venv
MYPY_FLAGS += --color-output
MYPY_FLAGS += --warn-return-any
MYPY_FLAGS += --warn-unused-ignores
MYPY_FLAGS += --ignore-missing-imports
MYPY_FLAGS += --disallow-untyped-defs
MYPY_FLAGS += --check-untyped-defs
RUFF_FORMAT_CHECK_FLAGS += --color always
RUFF_CHECK_FLAGS += --color always
TY_CHECK_FLAGS += --color always

RM := rm -rf

all: install run

install: 
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb $(MAIN_PROGRAM) 
	
test:
	uv run --with pytest pytest

clean:
	uvx ruff clean
	$(RM) .mypy_cache/
	$(RM) .pytest_cache/
	$(RM) .ruff_cache/
	$(RM) $(VENV)/
	find . -type d -name "__pycache__" -exec $(RM) {} +

lint:
	uvx flake8 $(FLAKE8_FLAGS) .
	uvx mypy $(MYPY_FLAGS) .

lint-strict: install
	uv run flake8 $(FLAKE8_FLAGS) .
	uv run mypy $(MYPY_FLAGS) --strict .
	uv run ruff format --check $(RUFF_FORMAT_CHECK_FLAGS)
	uv run ruff check $(RUFF_CHECK_FLAGS)
	uv run ty check $(TY_CHECK_FLAGS)

format:
	uvx ruff check --fix --select=I001
	uvx ruff format

.PHONY: all install run debug test clean lint lint-strict format build