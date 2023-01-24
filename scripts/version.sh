#!/usr/bin/env bash
# 
# Use:
#
# bash scripts/version.sh 1.0.0 -a "first version"
#

set -e

git tag "$@"
git push origin --tags