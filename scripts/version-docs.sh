#!/usr/bin/env bash
#
# Version docs by creating and pushing a git tag.
# Use:
#
# bash scripts/version-docs.sh 1.0.0
#

set -e

if [ $# -eq 0 ]; then
    echo "No version supplied"
    echo "Do e.g.: bash version-docs.sh 1.0.0"
    exit 1
fi

git tag "$1"
git push origin --tags

echo "Version pushed... activate new version docs on RTD"
