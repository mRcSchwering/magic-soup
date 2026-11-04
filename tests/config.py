import os

import torch

DEVICE = os.getenv("DEVICE", "cpu")

IDXTYPE = torch.int32
ITYPE = torch.int8
FTYPE = torch.float32
