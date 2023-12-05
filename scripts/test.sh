#!/usr/bin/env bash
#
# Run pytest test suite
#
# Use:
#
# bash scripts/test.sh tests/
#

set -e

PYTHONPATH=$PYTHONPATH:./python pytest "$@"
