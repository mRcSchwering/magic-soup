#!/usr/bin/env bash
#
# Run linters
#
set -e
uv lock --check
uv run --no-sync mypy python
uv run --no-sync mypy tests
uv run --no-sync ruff check --fix