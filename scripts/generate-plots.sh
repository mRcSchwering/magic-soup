#!/usr/bin/env bash
#
# Generate all plots
#
# Use:
#
# bash scripts/generate-plots.sh
#

set -e

uv run python -m docs.generate_plots "$@"
