import os

import torch

RNG_SEED = int(os.getenv("RNG_SEED", "42"))

DEVICE = os.getenv("DEVICE", "cpu")
DEBUG = bool(os.getenv("DEBUG", "False"))


IDXTYPE = torch.int32
ITYPE = torch.int8
FTYPE = torch.float32
