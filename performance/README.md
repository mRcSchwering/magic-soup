## Performance

Use one of the [Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) with an instance like
 (see [G4 instances](https://aws.amazon.com/ec2/instance-types/g4/) and [G5 instances](https://aws.amazon.com/ec2/instance-types/g5/)).
_E.g._ **Deep Learning AMI GPU PyTorch 1.12.1 (Ubuntu 20.04) 20221114** (ami-01e8ee929409916a3) has CUDA 11 and conda installed.
After starting the instance you can initialize conda and directly install the environment.

```bash
nvidia-smi -l 1
tensorboard --host 0.0.0.0 --logdir=./performance/runs
```

### Setup NVIDIA

Setup on Ubuntu server 22.04 with Quadro RTX4000 (8 GB GPU memory), 32 GiB RAM, and 8vCPU:

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
12. Check `bash scripts/setup-cuda.sh`


### Check

[check.py](./check.py) script for testing performance of specific functions used with 10k cells and 1k genome size on CPU.

| **code**      | **hardware**         | **spawn cells** | **replicate cells** | **advance time** | **mutations** |
|---------------|----------------------|-----------------|---------------------|------------------|---------------|
| v0.14.1 CPU   | Intel Core i5-10210U | (7.66+-0.51)s   | (0.37+-0.01)s       | (4.51+-0.12)s    | (7.17+-0.10)s |
| v0.14.1 GPU   | g4dn.xlarge          | (6.64+-1.08)s   | (0.28+-0.00)s       | (0.16+-0.00)s    | (5.95+-0.22)s |
| 0476f64e CPU  | Intel Core i5-10210U | (1.51+-0.12)s   | (0.46+-0.02)s       | (41.40+-4.58)s   | (0.10+-0.00)s |


### Simulation

[simulation.py](./simulation.py) dummy run to test performance in realistic setting.


| **code**       | **hardware** | **1s (1k cells)** | **1s (40k cells)** |
|----------------|--------------|-------------------|--------------------|
| 2023-06-08 GPU | g4dn.xlarge  | 0.7s              | 4.5s               |
| 2023-11-21 GPU | g4dn.xlarge  | 0.05s             | 0.35s              |
| 2023-12-19 GPU | g4dn.xlarge  | 0.03s             | 0.30s              |

new

| **code**     | **hardware**         | **simulated second** | **average cells**  |
|--------------|----------------------|----------------------|--------------------|
| 0476f64e CPU | Intel Core i5-10210U | 7.49e+00s            | 7,381 +/- 1,759    |