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

python -m build .
python -m twine upload --skip-existing dist/*