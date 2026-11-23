import os
import random

CODON_SIZE = 3  # in number of nucleotides
GAS_CONSTANT = 8.31446261815324  # in J/(K*mol)
ROOM_TEMP = 293  # in K

# default RNG
RNG_SEED = int(os.getenv("RNG_SEED", "42"))
DEFAULT_RNG = random.Random(RNG_SEED)

# default torch device
DEFAULT_DEVICE = os.getenv("DEVICE", "cpu")
