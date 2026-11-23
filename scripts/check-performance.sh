#!/usr/bin/env bash
#
# Tools for checking performance. Use:
#
# bash scripts/check-performance.sh --help
#
set -e

uv run --no-sync python -m performance "$@"
