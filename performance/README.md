## Performance

Some tools for checking overall simulation performance.

- [check.py](./check.py) script for testing performance of specific functions
- [run.py](./run.py) dummy run to test performance in realistic setting

### GPU Test

Use one of the [Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) with an instance like
 (see [G4 instances](https://aws.amazon.com/ec2/instance-types/g4/) and [G5 instances](https://aws.amazon.com/ec2/instance-types/g5/)).
_E.g._ **Deep Learning AMI GPU PyTorch 1.12.1 (Ubuntu 20.04) 20221114** (ami-01e8ee929409916a3) has CUDA 11 and conda installed.
After starting the instance you can initialize conda and directly install the environment.

```bash
conda init && source ~/.bashrc  # init conda
conda env create -f environment_cuda11.6.yml  # install environment
conda activate luca  # activate envrionment

nvcc --version  # check CUDA version
python -c 'import torch; print(torch.cuda.is_available())'  # check torch was compiled for it

PYTHONPATH=./src python performance/run.py --n_steps=100 --device=cuda  # start run using GPU
nvidia-smi -l 1  # monitor GPU
```
