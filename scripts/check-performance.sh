#!/usr/bin/env bash
#
# Check performance of specific functions
#
# Use:
#
# bash scripts/check-performance.sh
#

set -e

PYTHONPATH=$PYTHONPATH:./src python performance/check.py
