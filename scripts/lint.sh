#!/usr/bin/env bash
#
# Run linters
#
set -x
uv lock --check
uv run mypy python
uv run mypy tests
uv run ruff check --fix
uv run pylint python
uv run pylint performance
uv run ruff format python/