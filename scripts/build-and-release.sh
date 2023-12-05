#!/usr/bin/env bash
# 
# Builds and releases the simulation with the current version.
# Update __init__.py before running.
#
# Use:
#
# bash scripts/build-release.sh
#

set -e

version=$(head -n 1 ./python/magicsoup/__init__.py)
read -p "Build and release as ${version}? (Y/N)" confirm
[[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

python -m build .
python -m twine upload --skip-existing dist/*