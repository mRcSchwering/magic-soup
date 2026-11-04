import os
from itertools import product

import torch

CODON_SIZE = 3  # in number of nucleotides
GAS_CONSTANT = 8.31446261815324  # in J/(K*mol)
ROOM_TEMP = 293  # in K

ALL_NTS = tuple("TCGA")  # "N" represents any one of these
ALL_CODONS = {"".join(d) for d in product(ALL_NTS, ALL_NTS, ALL_NTS)}

DEFAULT_DEVICE = os.getenv("DEVICE", "cpu")

# these parameters have to be aligned with Rust
ITYPE = torch.int8
FTYPE = torch.float32
IDXTYPE = torch.int32
