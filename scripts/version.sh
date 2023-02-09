#!/usr/bin/env bash
# 
# Use:
#
# bash scripts/version.sh 1.0.0
#

set -e

git tag "$1" -a "$1"
git push origin --tags