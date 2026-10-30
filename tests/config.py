import os

import torch
from numpy.dtypes import StringDType  # pylint: disable=E0611

DEVICE = os.getenv("DEVICE", "cpu")

IDXTYPE = torch.int32
ITYPE = torch.int8
FTYPE = torch.float32
STYPE = StringDType()
