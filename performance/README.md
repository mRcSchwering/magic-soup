## Performance

### Check

[check.py](./check.py) script for testing performance of specific functions used with 10k cells and 1k genome size.

| **code**      | **hardware**             | **spawn cells** | **replicate cells** | **advance time** | **mutations** |
|---------------|--------------------------|-----------------|---------------------|------------------|---------------|
| v0.14.1 CPU   | Intel Core i5-10210U     | (7.66+-0.51)s   | (0.37+-0.01)s       | (4.51+-0.12)s    | (7.17+-0.10)s |
| v0.14.1 CUDA  | g4dn.xlarge              | (6.64+-1.08)s   | (0.28+-0.00)s       | (0.16+-0.00)s    | (5.95+-0.22)s |
| 0476f64e CPU  | Intel Core i5-10210U     | (1.51+-0.12)s   | (0.46+-0.02)s       | (41.40+-4.58)s   | (0.10+-0.00)s |
| 4b409449 CUDA | Quadro RTX 4000          | (0.69+-0.16)s   | (0.14+-0.01)s       | (4.78+-10.23)s   | (0.16+-0.01)s |

### Simulation

[simulation.py](./simulation.py) dummy run to test performance in a more realistic setting.

| **code**      | **hardware**         | **simulated second** | **average cells**  |
|---------------|----------------------|----------------------|--------------------|
| 0476f64e CPU  | Intel Core i5-10210U | 7.49e+00s            | 7,381 +/- 1,759    |
| 4b409449 CUDA | Quadro RTX 4000      | 1.23e-01s            | 8,030 +/- 2,105    |

### Setup Test Machine

Temporary installation on Ubuntu server 22.04 with Quadro RTX4000 (8 GB GPU memory), 32 GiB RAM, and 8 vCPU:

1. Update ubuntu `sudo apt update && sudo apt upgrade -y`
2. Install drivers `sudo ubuntu-drivers install --gpgpu`
3. Install build dependencies `sudo apt install build-essential python3-dev -y`
4. Reboot
5. Check drivers branch `dpkg -l | grep -E 'nvidia-driver|nvidia-utils|linux-modules-nvidia'`
6. Install the correct utils, e.g. `sudo apt install nvidia-utils-580-server`
7. Clone repo `git clone https://github.com/mRcSchwering/magic-soup.git`
8. Install uv `curl -LsSf https://astral.sh/uv/install.sh | sh`
9. Setup venv `cd magic-soup/ && uv venv --python 3.12`
10. Reboot
11. Install magic-soup `uv sync --group dev --no-sources-package torch --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`
12. Check driver and supported CUDA version `nvidia-smi`
13. Check CUDA works with pytorch `run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"`


