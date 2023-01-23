#!/usr/bin/env bash
# 
# Use:
#
# bash scripts/build-release.sh
#

set -e

python -m build .
python -m twine upload --skip-existing dist/*