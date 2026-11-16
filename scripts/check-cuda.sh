#!/usr/bin/env bash
#
# Setup magicsoup for CUDA 13 for local testing
#
set -e

echo "NVIDIA driver and supported CUDA version:"
nvidia-smi

echo "Is CUDA available from python torch:"
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"