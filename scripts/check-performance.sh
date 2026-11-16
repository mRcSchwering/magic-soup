#!/usr/bin/env bash
#
# Tools for checking performance. Use:
#
# bash scripts/check-performance.sh
#

set -e

uv run python -m performance "$@"
