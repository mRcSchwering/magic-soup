#!/usr/bin/env bash
# 
# Use:
#
# bash scripts/version-docs.sh 1.0.0
#

set -e

git tag "$1"
git push origin --tags

echo "Activate new version docs on RTD"
