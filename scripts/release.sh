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

py_version=$(grep "^version = " pyproject.toml | sed 's/version = //g' | sed 's/"//g')
rs_version=$(grep "^version = " Cargo.toml | sed 's/version = //g' | sed 's/"//g' | sed 's/#.*//g' | sed 's/[[:space:]]//g')
read -p "Release as v${py_version} python (v${rs_version} rust binary)? (y/N)" confirm
[[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

git tag "v$py_version"
git push origin "v$py_version"
