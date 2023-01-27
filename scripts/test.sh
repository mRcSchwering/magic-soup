#!/usr/bin/env bash
# 
# Use:
#
# bash scripts/test.sh tests/
#

set -e

PYTHONPATH=$PYTHONPATH:./src pytest "$@"
