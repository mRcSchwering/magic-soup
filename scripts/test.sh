#!/usr/bin/env bash
# 
# Use:
#
# bash scripts/test.sh
#

set -e

PYTHONPATH=$PYTHONPATH:./src pytest tests/
