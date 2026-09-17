#!/usr/bin/env bash
#
# Installs all dependencies for local development and tests
# (project, dev, others)
#
#   bash scripts/install.sh --no-install-project  # only install python deps
#
uv sync --all-groups "$@" 