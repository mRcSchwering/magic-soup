#!/usr/bin/env bash
#
# Installs all dependencies for local development and tests
# (project, dev, others)
#
#   bash scripts/install.sh --no-install-project  # only install python deps
#
set -e

echo "NVIDIA driver and supported CUDA version:"
nvidia-smi

echo "Installing project with CUDA"
uv venv
uv sync --group test --no-sources-package torch --index https://download.pytorch.org/whl/cu130
echo "!!! DONT COMMIT uv.lock !!!

echo "Is CUDA available from python torch:"
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"