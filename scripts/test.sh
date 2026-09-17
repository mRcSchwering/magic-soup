#!/usr/bin/env bash
#
# Run pytest test suite
#
set -e

uv run --locked pytest "$@"
