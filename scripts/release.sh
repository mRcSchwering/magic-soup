#!/usr/bin/env bash
# 
# Builds and releases the simulation with the current version.
# Update __init__.py before running.
#
# Use:
#
# bash scripts/release.sh
#
set -e

version=$(grep "^version = " pyproject.toml | sed 's/version = //g' | sed 's/"//g')
read -p "Release as v${version}? (y/N)" confirm
[[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

git tag "v$version"
git push origin "v$version"
