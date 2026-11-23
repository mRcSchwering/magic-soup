#!/usr/bin/env bash
#
# Builds and serves RTD documentation
#
# Use:
#
# bash scripts/serve-docs.sh
#
set -e

uv run --no-sync mkdocs serve