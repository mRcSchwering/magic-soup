#!/usr/bin/env bash
# 
# Use:
#
# bash scripts/serve-docs.sh
#

set -e

PYTHONPATH=$PYTHONPATH:./src mkdocs serve